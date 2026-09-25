"""FastAPI entry point.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import hmac
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from .config import DATA_DIR, Settings, get_settings
from .llm import create_backend
from .phonetics import PhoneticsKnowledgeBase
from .quran_corpus import QuranCorpus
from .schemas import AskRequest, AskResponse
from .service import QuranAIService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quran_ai")


UNAVAILABLE = "Quran AI is temporarily unavailable"
BUSY = "Quran AI is busy right now. Please try again in a minute."


def _failure_detail(error: Exception) -> str:
    """The model API's own rate limit (Groq free tier) reads as "busy", not "down"."""
    response = getattr(error, "response", None)
    return BUSY if getattr(response, "status_code", None) == 429 else UNAVAILABLE


DATA_FILES = ["surahs.json", "quran_ar.json", "quran_en.json", "arabic_letters.json", "tajweed_rules.json"]


def ensure_data(repo: str) -> None:
    """Downloads data/*.json from a private HF dataset when they aren't bundled (the public
    Space ships code only). Uses the HF_TOKEN secret."""
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        return
    if not repo:
        raise RuntimeError(f"Missing data files {missing} and QURAN_AI_DATA_REPO is not set")
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in missing:
        hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=str(DATA_DIR))
    log.info("Downloaded %d data files from %s", len(missing), repo)


class RateLimiter:
    """Sliding one-minute window per client. In-memory: per process, reset on restart --
    put a gateway limit in front when running more than one instance."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def build_app(service: Optional[QuranAIService] = None, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    state: Dict[str, QuranAIService] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if service is not None:
            state["service"] = service
        else:
            ensure_data(settings.data_repo)
            log.info("Loading %s via %s backend", settings.model_id, settings.llm_backend)
            llm = create_backend(settings.llm_backend, settings.model_id, settings.llm_base_url,
                                 settings.llm_api_key, settings.llm_provider)
            state["service"] = QuranAIService(llm, PhoneticsKnowledgeBase(), QuranCorpus(), settings.max_new_tokens)
        yield

    app = FastAPI(title="Quran AI", version="1.0.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    def authorize(request: Request, x_quranai_key: Optional[str] = Header(default=None)) -> None:
        if settings.client_api_key and not hmac.compare_digest(x_quranai_key or "", settings.client_api_key):
            raise HTTPException(status_code=401, detail="Invalid client key")
        client = request.client.host if request.client else "unknown"
        if settings.trust_forwarded_for:
            # The right-most entry is the one our own proxy appended; left-most is client-supplied.
            forwarded = [ip.strip() for ip in request.headers.get("x-forwarded-for", "").split(",") if ip.strip()]
            client = forwarded[-1] if forwarded else client
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="Too many questions. Please wait a minute and try again.")

    @app.get("/health")
    def health() -> dict:
        svc = state.get("service")
        return {"status": "ok" if svc else "loading", "model": svc.llm.name if svc else None}

    # A sync endpoint: FastAPI runs it in a worker thread, so a slow generate() doesn't block
    # the event loop (and /health stays responsive).
    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(authorize)])
    def ask(request: AskRequest) -> AskResponse:
        svc = state.get("service")
        if svc is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        try:
            return svc.ask(request)
        except Exception as error:
            log.exception("ask failed")
            raise HTTPException(status_code=503, detail=_failure_detail(error))

    @app.post("/v1/ask/stream", dependencies=[Depends(authorize)])
    def ask_stream(request: AskRequest) -> StreamingResponse:
        """Server-Sent Events version of /v1/ask; see QuranAIService.ask_stream for events."""
        svc = state.get("service")
        if svc is None:
            raise HTTPException(status_code=503, detail="Model is still loading")

        # Starlette never closes a sync generator whose client went away, so the model kept
        # generating (spending the provider's tokens-per-minute) and the half-read stream was
        # finalized by the GC on an arbitrary thread. Step it from here instead, and close it
        # once the in-flight step is done.
        stream = svc.ask_stream(request)
        step_lock = threading.Lock()

        def step():
            with step_lock:
                return next(stream, None)

        def close():
            with step_lock:
                stream.close()

        async def events():
            try:
                while (event := await run_in_threadpool(step)) is not None:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as error:
                log.exception("ask_stream failed")
                yield f"data: {json.dumps({'type': 'error', 'detail': _failure_detail(error)})}\n\n"
            finally:
                threading.Thread(target=close, daemon=True).start()

        # X-Accel-Buffering: stop nginx-style proxies from buffering the stream into one blob.
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = build_app()
