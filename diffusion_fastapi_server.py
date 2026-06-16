from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Any

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

app = FastAPI(title="DiffusionGemma Fast Decomposition API")
generation_lock = threading.Lock()
load_lock = threading.Lock()
startup_loaded = False
startup_load_ms: float | None = None


class DecomposeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=768, ge=1, le=4096)
    early_stop: bool = True


class DecomposeResponse(BaseModel):
    subqueries: list[str]
    mode: str
    elapsed_ms: float
    draft_step: int | None
    final_text: str | None = None
    backend: str
    server_elapsed_ms: float
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


@app.post("/decompose", response_model=DecomposeResponse)
def decompose(request: DecomposeRequest) -> DecomposeResponse:
    server_start = time.perf_counter()
    try:
        ensure_model_loaded()
        with generation_lock:
            result = decompose_fast_mode(
                query=request.query,
                model_id=MODEL_ID,
                max_new_tokens=request.max_new_tokens,
                early_stop=request.early_stop,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = asdict(result)
    payload.update(
        {
            "backend": "server_hf_fastapi",
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
