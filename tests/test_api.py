"""API integration tests for graph-executor service.

Uses TestClient (sync httpx). No real LLM calls — LLM_BASE_URL is unset.
AC coverage:
  AC1  — /health returns available graphs
  AC2  — authenticated planner invoke returns typed output
  AC3  — unknown graph → 404
  AC4  — auth rejects unauthorized calls
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")

# Set required env before importing app
os.environ.setdefault("GRAPH_EXECUTOR_TOKEN", "test-secret-token")
# Ensure LLM not configured (stub mode)
os.environ.pop("LLM_BASE_URL", None)

from main import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-secret-token"}
BAD_AUTH = {"Authorization": "Bearer wrong-token"}


# ── AC1: Health ───────────────────────────────────────────────────────


def test_health_returns_graphs():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "planner" in body["graphs"]


# ── AC3: Unknown graph → 404 ──────────────────────────────────────────


def test_unknown_graph_returns_404():
    resp = client.post(
        "/v1/graphs/nonexistent/invoke",
        json={"input": {}, "config": {}},
        headers=AUTH,
    )
    assert resp.status_code == 404
    assert "nonexistent" in resp.json()["detail"]


# ── AC4: Auth rejects unauthorized calls ──────────────────────────────


def test_auth_rejects_wrong_token():
    resp = client.post(
        "/v1/graphs/planner/invoke",
        json={"input": {"objective_text": "test"}, "config": {}},
        headers=BAD_AUTH,
    )
    assert resp.status_code == 401


def test_auth_rejects_missing_token():
    resp = client.post(
        "/v1/graphs/planner/invoke",
        json={"input": {"objective_text": "test"}, "config": {}},
    )
    assert resp.status_code == 401


# ── AC2: Authenticated planner invoke ─────────────────────────────────


def test_planner_invoke_returns_typed_output():
    """AC2 — stub mode (no LLM) returns a valid plan structure."""
    resp = client.post(
        "/v1/graphs/planner/invoke",
        json={
            "input": {"objective_text": "Build a REST API"},
            "config": {"temperature": 0.2, "max_tokens": 2000},
        },
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert "run_id" in body
    assert body["metadata"]["graph_name"] == "planner"
    assert isinstance(body["metadata"]["latency_ms"], float)
    assert "packets" in body["output"]
    assert len(body["output"]["packets"]) >= 1


def test_planner_invoke_metadata_fields():
    resp = client.post(
        "/v1/graphs/planner/invoke",
        json={"input": {"objective_text": "Add caching layer"}, "config": {}},
        headers=AUTH,
    )
    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert set(meta.keys()) >= {"graph_name", "revision_count", "latency_ms", "model_used"}


def test_planner_invoke_run_id_is_uuid():
    resp = client.post(
        "/v1/graphs/planner/invoke",
        json={"input": {"objective_text": "Refactor auth"}, "config": {}},
        headers=AUTH,
    )
    import uuid
    run_id = resp.json()["run_id"]
    uuid.UUID(run_id)  # raises ValueError if not valid UUID


def test_planner_invoke_no_token_env_skips_auth():
    """When GRAPH_EXECUTOR_TOKEN is unset, auth is disabled (dev mode)."""
    original = os.environ.pop("GRAPH_EXECUTOR_TOKEN", None)
    try:
        resp = client.post(
            "/v1/graphs/planner/invoke",
            json={"input": {"objective_text": "Test dev mode"}, "config": {}},
            # No auth header
        )
        assert resp.status_code == 200
    finally:
        if original:
            os.environ["GRAPH_EXECUTOR_TOKEN"] = original
