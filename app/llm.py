"""LLM backends. The service depends only on the `LLMBackend` protocol, so tests use a fake."""
import json
import re
import threading
import time
from typing import Dict, Iterator, List, Protocol

Messages = List[Dict[str, str]]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Qwen3 can emit a <think> block; it is never shown to the user."""
    return _THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "").strip()


class LLMBackend(Protocol):
    name: str

    def generate(self, messages: Messages, max_new_tokens: int, temperature: float) -> str: ...

    def stream(self, messages: Messages, max_new_tokens: int, temperature: float) -> Iterator[str]:
        """Yields the answer in pieces as it is generated. Reasoning tags are not stripped
        here; the service does that on the accumulated text."""
        ...


class TransformersBackend:
    """Runs the model in-process. Fine for a single instance / modest traffic."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        self._torch = torch
        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        # One generate() at a time: concurrent calls on one model instance would just contend
        # for the same device and multiply memory use.
        self._lock = threading.Lock()

    def generate(self, messages: Messages, max_new_tokens: int, temperature: float) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: answer directly; ignored by templates without it
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
        with self._lock, self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling,
                                         repetition_penalty=1.1,  # small models otherwise loop ("ʿalā ʿalā ʿalā …")
                                         pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return strip_reasoning(text)

    def stream(self, messages: Messages, max_new_tokens: int, temperature: float) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}

        def run() -> None:
            with self._lock, self._torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling, streamer=streamer,
                                    repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id)

        threading.Thread(target=run, daemon=True).start()
        yield from streamer


class OpenAICompatibleBackend:
    """Calls a vLLM / TGI / llama.cpp server exposing /v1/chat/completions.

    Recommended for production: e.g. `vllm serve Qwen/Qwen3-8B --port 8001` gives continuous
    batching, which the in-process backend can't.
    """

    # A rate-limited request (Groq's free tier: 8K tokens/minute) is retried once when the
    # provider asks for a short wait; longer waits fail fast so the app can say "busy".
    MAX_RETRY_WAIT = 6.0

    def __init__(self, model_id: str, base_url: str, api_key: str, provider: str = "vllm") -> None:
        import httpx

        self.name = model_id
        self.provider = provider
        self._client_args = dict(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(60.0, connect=10.0),
                                 headers={"Authorization": f"Bearer {api_key}"})
        self._client = httpx.Client(**self._client_args)

    def _retry_wait(self, response) -> float:
        """Seconds to wait before one retry of a 429, or 0 to give up."""
        if response.status_code != 429:
            return 0.0
        try:
            wait = float(response.headers.get("retry-after", "1"))
        except ValueError:
            wait = 1.0
        return wait if wait <= self.MAX_RETRY_WAIT else 0.0

    def _body(self, messages: Messages, max_new_tokens: int, temperature: float, stream: bool) -> dict:
        body = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if self.provider == "groq":
            # Groq: answer directly (no reasoning tokens -- they'd count against the free
            # tier's tokens-per-minute) and it rejects vLLM-only options.
            body["max_completion_tokens"] = max_new_tokens
            body["reasoning_effort"] = "none"
        else:
            body["max_tokens"] = max_new_tokens
            body["repetition_penalty"] = 1.1  # vLLM / mlx_lm extension
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def generate(self, messages: Messages, max_new_tokens: int, temperature: float) -> str:
        body = self._body(messages, max_new_tokens, temperature, False)
        response = self._client.post("/chat/completions", json=body)
        wait = self._retry_wait(response)
        if wait:
            time.sleep(wait)
            response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        return strip_reasoning(response.json()["choices"][0]["message"]["content"] or "")

    def stream(self, messages: Messages, max_new_tokens: int, temperature: float) -> Iterator[str]:
        """Each stream gets its own client (its own connection pool). When the app disconnects
        mid-answer, Starlette abandons this generator without closing it, and it's finalized
        later by the garbage collector on whatever thread happens to run -- if that's a thread
        already inside the shared pool's (non-reentrant) lock, every later request deadlocks.
        That froze the Render deployment after a couple of questions."""
        import httpx

        body = self._body(messages, max_new_tokens, temperature, True)
        with httpx.Client(**self._client_args) as client:
            for attempt in range(2):
                with client.stream("POST", "/chat/completions", json=body) as response:
                    wait = self._retry_wait(response) if attempt == 0 else 0.0
                    if wait:
                        response.close()
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    yield from self._pieces(response)
                    return

    @staticmethod
    def _pieces(response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece


def create_backend(backend: str, model_id: str, base_url: str, api_key: str, provider: str = "vllm") -> LLMBackend:
    if backend == "transformers":
        return TransformersBackend(model_id)
    if backend == "openai_compatible":
        return OpenAICompatibleBackend(model_id, base_url, api_key, provider)
    raise ValueError(f"Unknown QURAN_AI_LLM_BACKEND: {backend!r}")
