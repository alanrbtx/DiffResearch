from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diffusion_deep_research import decompose_fast_mode, run_deep_research
from src.agents.agent_template import DEFAULT_DIFFUSIONGEMMA_MODEL, _get_diffusiongemma_runtime


MODEL_ID = os.environ.get("DIFFUSIONGEMMA_MODEL_ID") or DEFAULT_DIFFUSIONGEMMA_MODEL
RESEARCH_OUTPUT_DIR = Path(os.environ.get("DIFFUSION_RESEARCH_OUTPUT_DIR", "/workspace/diffresearch-api/runs"))
LOAD_ON_STARTUP = os.environ.get("DIFFUSION_SERVER_LOAD_ON_STARTUP", "1").lower() not in {
    "0",
    "false",
    "no",
}

app = FastAPI(title="DiffusionGemma Deep Research API")
generation_lock = threading.Lock()
research_lock = threading.Lock()
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


class ResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: Literal["fast", "full"] = "fast"
    max_new_tokens: int = Field(default=768, ge=1, le=4096)
    arxiv_per_query: int = Field(default=2, ge=0, le=10)
    s2_per_query: int = Field(default=2, ge=0, le=10)
    squeeze: bool = False
    relevance: bool = False


class ResearchResponse(BaseModel):
    report: str
    metadata: dict[str, Any]
    server_elapsed_ms: float
    output: str
    metadata_output: str
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
        "supports": ["/decompose", "/research"],
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


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    server_start = time.perf_counter()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{request.mode}-{uuid4().hex[:8]}"
    output_path = RESEARCH_OUTPUT_DIR / f"{run_id}-report.txt"
    metadata_path = RESEARCH_OUTPUT_DIR / f"{run_id}-metadata.json"
    try:
        ensure_model_loaded()
        args = SimpleNamespace(
            prompt=request.prompt,
            output=str(output_path),
            metadata_output=str(metadata_path),
            model_id=MODEL_ID,
            decomposition_base_url=None,
            decomposition_timeout=180.0,
            decomposition_mode=request.mode,
            agent_backend="diffusiongemma",
            agent_model=MODEL_ID,
            max_new_tokens=request.max_new_tokens,
            arxiv_per_query=request.arxiv_per_query,
            s2_per_query=request.s2_per_query,
            squeeze=request.squeeze,
            relevance=request.relevance,
            no_early_stop=False,
        )
        with research_lock, generation_lock:
            result = run_deep_research(args)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    server_elapsed_ms = elapsed_ms(server_start)
    metadata = result["metadata"]
    metadata["server"] = {
        "model_id": MODEL_ID,
        "server_elapsed_ms": server_elapsed_ms,
        "run_id": run_id,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ResearchResponse(
        report=result["review"],
        metadata=metadata,
        server_elapsed_ms=server_elapsed_ms,
        output=result["output"],
        metadata_output=result["metadata_output"],
        model_id=MODEL_ID,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "diffusion_fastapi_server:app",
        host=os.environ.get("DIFFUSION_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("DIFFUSION_SERVER_PORT", "8000")),
        reload=False,
    )
