"""Runtime configuration, read from environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Shown verbatim by the app when a question falls outside the Quran. Product-defined wording.
OFF_TOPIC_MESSAGE = "This question is not according to the Privacy Policy integrated within the application"


@dataclass(frozen=True)
class Settings:
    # "transformers" runs the model in-process; "openai_compatible" calls a vLLM / TGI /
    # llama.cpp server exposing /v1/chat/completions (recommended for production traffic).
    llm_backend: str = field(default_factory=lambda: os.getenv("QURAN_AI_LLM_BACKEND", "transformers"))
    # Apache-2.0, strong Arabic. See README "Why not justdeen/QuranPlus".
    model_id: str = field(default_factory=lambda: os.getenv("QURAN_AI_MODEL_ID", "Qwen/Qwen3-8B"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("QURAN_AI_LLM_BASE_URL", "http://localhost:8001/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("QURAN_AI_LLM_API_KEY", "EMPTY"))
    # Which OpenAI-compatible server the "openai_compatible" backend talks to. They differ in
    # the extra request options they accept: "vllm" (also mlx_lm) takes repetition_penalty and
    # chat_template_kwargs; "groq" takes reasoning_effort instead.
    llm_provider: str = field(default_factory=lambda: os.getenv("QURAN_AI_LLM_PROVIDER", "vllm"))
    # Optional private Hugging Face dataset holding data/*.json, downloaded at startup when
    # the files aren't present -- so a public deployment doesn't publish the bundled texts.
    data_repo: str = field(default_factory=lambda: os.getenv("QURAN_AI_DATA_REPO", ""))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("QURAN_AI_MAX_NEW_TOKENS", "350")))
    # Shared secret the app sends as X-QuranAI-Key. Empty disables the check (local dev only).
    client_api_key: str = field(default_factory=lambda: os.getenv("QURAN_AI_CLIENT_KEY", ""))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("QURAN_AI_RATE_LIMIT_PER_MINUTE", "20")))
    # Only enable behind your own proxy/load balancer that overwrites X-Forwarded-For. Off by
    # default: a client could otherwise send a new value per request to dodge the rate limit.
    trust_forwarded_for: bool = field(default_factory=lambda: os.getenv("QURAN_AI_TRUST_FORWARDED_FOR", "false").lower() == "true")


def get_settings() -> Settings:
    return Settings()
