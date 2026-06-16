from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diffusion_deep_research import decompose_fast_mode
from src.agents.agent_template import DEFAULT_DIFFUSIONGEMMA_MODEL, _get_diffusiongemma_runtime


MODEL_ID = os.environ.get("DIFFUSIONGEMMA_MODEL_ID") or DEFAULT_DIFFUSIONGEMMA_MODEL
LOAD_ON_STARTUP = os.environ.get("DIFFUSION_SERVER_LOAD_ON_STARTUP", "1").lower() not in {
    "0",
    "false",
    "no",
}

app = FastAPI(title="DiffusionGemma Model API")
generation_lock = threading.Lock()
load_lock = threading.Lock()
startup_loaded = False
startup_load_ms: float | None = None


class DecomposeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=768, ge=1, le=4096)
    mode: Literal["fast", "full"] = "fast"
    early_stop: bool | None = None


class DecomposeResponse(BaseModel):
    subqueries: list[str]
    mode: str
    request_mode: str
    elapsed_ms: float
    draft_step: int | None
    final_text: str | None = None
    backend: str
    server_elapsed_ms: float
    model_id: str


class GenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_new_tokens: int = Field(default=512, ge=1, le=8192)


class GenerateResponse(BaseModel):
    text: str
    elapsed_ms: float
    model_id: str


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def ensure_model_loaded() -> None:
    global startup_loaded, startup_load_ms
    if startup_loaded:
        return
    with load_lock:
        if startup_loaded:
            return
        start_time = time.perf_counter()
        _get_diffusiongemma_runtime(MODEL_ID)
        startup_load_ms = elapsed_ms(start_time)
        startup_loaded = True


@app.on_event("startup")
def load_model_on_startup() -> None:
    if LOAD_ON_STARTUP:
        ensure_model_loaded()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "loaded": startup_loaded,
        "startup_load_ms": startup_load_ms,
        "supports": ["/decompose", "/generate", "/v1/chat/completions"],
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "diffusion-fast-decompose",
            }
        ],
    }


def generate_text(messages: list[dict[str, Any]], max_new_tokens: int) -> tuple[str, float]:
    ensure_model_loaded()
    start_time = time.perf_counter()
    with generation_lock:
        runtime = _get_diffusiongemma_runtime(MODEL_ID)
        text = runtime.generate(messages, max_new_tokens=max_new_tokens)
    return text, elapsed_ms(start_time)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if request.messages:
        messages = request.messages
    elif request.prompt:
        messages = [{"role": "user", "content": request.prompt}]
    else:
        raise HTTPException(status_code=422, detail="Either prompt or messages is required.")

    try:
        text, generation_ms = generate_text(messages, request.max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return GenerateResponse(text=text, elapsed_ms=generation_ms, model_id=MODEL_ID)


@app.post("/v1/chat/completions")
def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=422, detail="messages must be a non-empty list.")

    max_tokens = (
        request.get("max_tokens")
        or request.get("max_completion_tokens")
        or request.get("max_new_tokens")
        or 512
    )
    try:
        max_tokens = int(max_tokens)
        text, generation_ms = generate_text(messages, max_new_tokens=max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model = str(request.get("model") or MODEL_ID)
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "generation_ms": generation_ms,
    }


@app.post("/decompose", response_model=DecomposeResponse)
def decompose(request: DecomposeRequest) -> DecomposeResponse:
    server_start = time.perf_counter()
    try:
        ensure_model_loaded()
        request_mode = request.mode
        early_stop = request.early_stop if request.early_stop is not None else request_mode == "fast"
        if request_mode == "full":
            early_stop = False
        with generation_lock:
            result = decompose_fast_mode(
                query=request.query,
                model_id=MODEL_ID,
                max_new_tokens=request.max_new_tokens,
                early_stop=early_stop,
                request_mode=request_mode,
                require_final=request_mode == "full",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = asdict(result)
    payload.update(
        {
            "backend": "server_hf_fastapi",
            "request_mode": request.mode,
            "server_elapsed_ms": elapsed_ms(server_start),
            "model_id": MODEL_ID,
        }
    )
    return DecomposeResponse(**payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "diffusion_fastapi_server:app",
        host=os.environ.get("DIFFUSION_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("DIFFUSION_SERVER_PORT", "8000")),
        reload=False,
    )
