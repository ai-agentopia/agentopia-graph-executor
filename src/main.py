"""Graph Execution Service — FastAPI entrypoint.

Port: 18792
Auth: Authorization: Bearer {GRAPH_EXECUTOR_TOKEN}
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from llm import build_llm_callable
from registry import GRAPH_NAMES, REGISTRY

app = FastAPI(title="agentopia-graph-executor", version="0.1.0")
_bearer = HTTPBearer(auto_error=False)


# ── Auth ─────────────────────────────────────────────────────────────


def _verify_token(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    expected = os.environ.get("GRAPH_EXECUTOR_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GRAPH_EXECUTOR_TOKEN not configured — service cannot authenticate requests",
        )
    if creds is None or creds.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
        )


# ── Request / Response models ─────────────────────────────────────────


class InvokeConfig(BaseModel):
    model: str = Field(default="", description="LLM model override")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)


class InvokeRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    config: InvokeConfig = Field(default_factory=InvokeConfig)


class InvokeMetadata(BaseModel):
    graph_name: str
    revision_count: int
    latency_ms: float
    model_used: str


class InvokeResponse(BaseModel):
    run_id: str
    status: str  # "completed" | "failed"
    output: dict[str, Any]
    metadata: InvokeMetadata
    errors: list[str]


# ── Routes ────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "graphs": GRAPH_NAMES}


@app.post(
    "/v1/graphs/{graph_name}/invoke",
    response_model=InvokeResponse,
    dependencies=[Depends(_verify_token)],
)
async def invoke_graph(
    graph_name: str,
    body: InvokeRequest,
) -> InvokeResponse:
    if graph_name not in REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown graph: '{graph_name}'. Available: {GRAPH_NAMES}",
        )

    handler = REGISTRY[graph_name]
    run_id = str(uuid.uuid4())
    model_used = body.config.model or os.environ.get("GRAPH_MODEL", "gpt-4o-mini")

    # Build LLM callable — required unless GRAPH_STUB_MODE=1 (test/dev only)
    llm = None
    stub_mode = os.environ.get("GRAPH_STUB_MODE", "") == "1"
    if stub_mode:
        llm = None  # explicit: caller opted into deterministic stub
    elif not os.environ.get("LLM_BASE_URL"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured: LLM_BASE_URL is required (set GRAPH_STUB_MODE=1 for test/dev only)",
        )
    else:
        try:
            llm = build_llm_callable(
                model=model_used,
                temperature=body.config.temperature,
                max_tokens=body.config.max_tokens,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM not configured: missing {exc}",
            )

    t0 = time.perf_counter()
    errors: list[str] = []
    invoke_status = "completed"
    output: dict[str, Any] = {}

    try:
        output = handler(body.input, llm)
        # Surface validation errors from graph output
        if not output.get("valid", True):
            errors = output.get("validation_errors", [])
    except Exception as exc:  # noqa: BLE001
        invoke_status = "failed"
        errors = [str(exc)]

    latency_ms = (time.perf_counter() - t0) * 1000

    return InvokeResponse(
        run_id=run_id,
        status=invoke_status,
        output=output,
        metadata=InvokeMetadata(
            graph_name=graph_name,
            revision_count=output.get("revision_count", 0),
            latency_ms=round(latency_ms, 2),
            model_used=model_used,
        ),
        errors=errors,
    )
