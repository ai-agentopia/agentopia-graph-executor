"""Unit tests for the reviewer shadow graph.

All tests run without I/O — LLM callable is mocked as needed.
Coverage:
  - First-round approve path (has diff + criteria)
  - First-round request-changes path (empty diff)
  - Rework-round path with prior comments
  - Rework-round reconciliation (prior issues addressed)
  - LLM callable injection — happy path
  - LLM callable injection — malformed response handling
  - Graph module has zero I/O imports
  - Graph has expected nodes
"""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from graphs.reviewer import (
    ReviewAnalysis,
    ReviewInput,
    build_reviewer_graph,
    invoke_reviewer,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _good_review_json(verdict: str = "APPROVE", criteria: list[str] | None = None) -> str:
    criteria = criteria or ["Tests pass", "Code is clean"]
    if verdict == "APPROVE":
        return json.dumps({
            "verdict": "APPROVE",
            "summary": "All criteria met",
            "findings": [],
            "criteria_met": criteria,
            "criteria_unmet": [],
        })
    return json.dumps({
        "verdict": "REQUEST_CHANGES",
        "summary": "Some criteria unmet",
        "findings": [
            {
                "file": "src/main.py",
                "line": 42,
                "body": "Missing error handling",
                "severity": "error",
                "category": "code_quality",
            }
        ],
        "criteria_met": criteria[:1],
        "criteria_unmet": criteria[1:],
    })


def _reconcile_json(addressed: list[str], remaining: list[str]) -> str:
    return json.dumps({
        "prior_issues_addressed": addressed,
        "prior_issues_remaining": remaining,
    })


# ── Zero I/O / provider imports in graph module ──────────────────────


def test_graph_module_no_io_imports():
    """graphs/reviewer.py must not import I/O or provider packages."""
    forbidden = {"httpx", "requests", "aiohttp", "anthropic", "openai", "boto3", "os"}
    import graphs.reviewer as rm

    for name, obj in inspect.getmembers(rm):
        if inspect.ismodule(obj) and obj.__name__.split(".")[0] in forbidden:
            pytest.fail(f"graphs/reviewer.py imported forbidden module: {obj.__name__}")

    src = inspect.getsource(rm)
    for pkg in forbidden:
        assert f"import {pkg}" not in src, f"Found 'import {pkg}' in graphs/reviewer.py"


# ── Graph structure ──────────────────────────────────────────────────


def test_graph_has_expected_nodes():
    graph = build_reviewer_graph()
    compiled = graph.compile()
    node_names = set(compiled.get_graph().nodes.keys())
    for expected in ("analyze", "reconcile_prior", "decide"):
        assert expected in node_names, f"Missing node: {expected}"


# ── Stub (no LLM) — first-round approve ─────────────────────────────


def test_first_round_approve():
    """Stub path: non-empty diff + criteria → APPROVE."""
    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ added a line",
            acceptance_criteria=["Tests pass", "No security issues"],
            pr_title="feat: add feature",
            pr_body="Adds a new feature",
            file_names=["src/main.py"],
        )
    )
    assert isinstance(result, ReviewAnalysis)
    assert result.verdict == "APPROVE"
    assert result.criteria_met == ["Tests pass", "No security issues"]
    assert result.criteria_unmet == []
    assert not result.is_rework
    assert result.prior_issues_addressed == []
    assert result.prior_issues_remaining == []


# ── Stub — first-round request changes (empty diff) ─────────────────


def test_first_round_request_changes_empty_diff():
    """Stub path: empty diff → REQUEST_CHANGES."""
    result = invoke_reviewer(
        ReviewInput(
            pr_diff="",
            acceptance_criteria=["Tests pass"],
            pr_title="fix: nothing",
            pr_body="",
            file_names=[],
        )
    )
    assert isinstance(result, ReviewAnalysis)
    assert result.verdict == "REQUEST_CHANGES"
    assert len(result.findings) >= 1


def test_first_round_request_changes_no_criteria():
    """Stub path: no criteria → REQUEST_CHANGES."""
    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ some change",
            acceptance_criteria=[],
            pr_title="fix: something",
            pr_body="",
            file_names=["a.py"],
        )
    )
    assert isinstance(result, ReviewAnalysis)
    assert result.verdict == "REQUEST_CHANGES"


# ── Stub — rework round ─────────────────────────────────────────────


def test_rework_round_with_prior_comments():
    """Stub path: rework round with prior comments and diff → APPROVE, prior addressed."""
    prior = ["Fix error handling in main.py", "Add tests for edge case"]
    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ fixed error handling\n+ added edge case test",
            acceptance_criteria=["Tests pass"],
            pr_title="fix: address review",
            pr_body="Addressed review comments",
            file_names=["src/main.py", "tests/test_main.py"],
            rework_round=1,
            prior_comments=prior,
        )
    )
    assert isinstance(result, ReviewAnalysis)
    assert result.verdict == "APPROVE"
    assert result.is_rework
    assert result.prior_issues_addressed == prior
    assert result.prior_issues_remaining == []


def test_rework_round_no_diff():
    """Stub path: rework round with no diff → REQUEST_CHANGES, prior remaining."""
    prior = ["Fix the bug"]
    result = invoke_reviewer(
        ReviewInput(
            pr_diff="",
            acceptance_criteria=["Tests pass"],
            pr_title="fix: attempt",
            pr_body="",
            file_names=[],
            rework_round=2,
            prior_comments=prior,
        )
    )
    assert isinstance(result, ReviewAnalysis)
    assert result.verdict == "REQUEST_CHANGES"
    assert result.is_rework
    assert result.prior_issues_remaining == prior


# ── LLM callable injection — happy path ──────────────────────────────


def test_llm_callable_approve():
    """LLM returns valid APPROVE JSON → ReviewAnalysis with APPROVE."""
    criteria = ["Tests pass", "Code is clean"]
    calls: list[Any] = []

    def mock_llm(messages):
        calls.append(messages)
        return _good_review_json("APPROVE", criteria)

    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ good code",
            acceptance_criteria=criteria,
            pr_title="feat: add",
            pr_body="body",
            file_names=["a.py"],
        ),
        llm=mock_llm,
    )
    assert result.verdict == "APPROVE"
    assert result.criteria_met == criteria
    assert len(calls) == 1


def test_llm_callable_request_changes():
    """LLM returns valid REQUEST_CHANGES JSON."""
    criteria = ["Tests pass", "Code is clean"]

    def mock_llm(messages):
        return _good_review_json("REQUEST_CHANGES", criteria)

    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ bad code",
            acceptance_criteria=criteria,
            pr_title="feat: bad",
            pr_body="body",
            file_names=["a.py"],
        ),
        llm=mock_llm,
    )
    assert result.verdict == "REQUEST_CHANGES"
    assert len(result.findings) >= 1


# ── LLM callable — malformed response ───────────────────────────────


def test_llm_callable_malformed_response():
    """LLM returns non-JSON → REQUEST_CHANGES with error finding."""

    def mock_llm(messages):
        return "this is not json {"

    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ some code",
            acceptance_criteria=["Tests pass"],
            pr_title="feat: test",
            pr_body="",
            file_names=["a.py"],
        ),
        llm=mock_llm,
    )
    assert result.verdict == "REQUEST_CHANGES"
    assert any("malformed" in f.body.lower() for f in result.findings)


# ── LLM callable — rework with reconciliation ───────────────────────


def test_llm_rework_reconciliation():
    """LLM rework path calls analyze + reconcile_prior + decide."""
    call_count = 0
    prior = ["Fix bug A", "Add test B"]

    def mock_llm(messages):
        nonlocal call_count
        call_count += 1
        # First call = analyze, second call = reconcile_prior
        if call_count == 1:
            return _good_review_json("APPROVE", ["Tests pass"])
        return _reconcile_json(
            addressed=["Fix bug A"],
            remaining=["Add test B"],
        )

    result = invoke_reviewer(
        ReviewInput(
            pr_diff="+ fixed bug A",
            acceptance_criteria=["Tests pass"],
            pr_title="fix: rework",
            pr_body="",
            file_names=["a.py"],
            rework_round=1,
            prior_comments=prior,
        ),
        llm=mock_llm,
    )
    assert call_count == 2  # analyze + reconcile_prior
    assert result.is_rework
    assert result.prior_issues_addressed == ["Fix bug A"]
    assert result.prior_issues_remaining == ["Add test B"]
