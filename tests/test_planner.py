"""Unit tests for the planner graph.

All tests run without I/O — LLM callable is mocked as needed.
AC coverage:
  AC5  — revision loop fires on malformed LLM output
  AC6  — finalize path fires on revision budget exhaustion
  AC7  — graph module has zero I/O / provider imports
  AC8  — LLM callable constructed in llm.py, not graph module
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from typing import Any

import pytest

# Ensure src/ is on path when running from repo root
sys.path.insert(0, "src")

from graphs.planner import (
    DeliveryPlan,
    PlannerInput,
    build_planner_graph,
    invoke_planner,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _good_plan_json(objective: str = "Build the thing") -> str:
    return json.dumps({
        "milestone_title": f"Deliver: {objective}",
        "issue_title": objective,
        "issue_body": f"Objective: {objective}",
        "packets": [
            {
                "title": objective,
                "objective": f"Implement {objective} with full test coverage",
                "assigned_role": "worker",
                "in_scope": [f"Implement core logic for {objective}"],
                "acceptance_criteria": [f"All integration tests pass for {objective}"],
            }
        ],
    })


def _bad_plan_json() -> str:
    """Returns a plan JSON that fails validation (generic criteria)."""
    return json.dumps({
        "milestone_title": "Deliver",
        "issue_title": "Do the thing",
        "issue_body": "body",
        "packets": [
            {
                "title": "Packet",
                "objective": "Implement the objective",
                "assigned_role": "worker",
                "in_scope": ["Implement the objective"],
                "acceptance_criteria": ["Complete the objective"],
            }
        ],
    })


def _malformed_json() -> str:
    return "not json {"


# ── AC7: Zero I/O / provider imports in graph module ─────────────────


def test_graph_module_no_io_imports():
    """AC7 — graphs/planner.py must not import I/O or provider packages."""
    forbidden = {"httpx", "requests", "aiohttp", "anthropic", "openai", "boto3", "os"}
    import graphs.planner as pm

    for name, obj in inspect.getmembers(pm):
        if inspect.ismodule(obj) and obj.__name__.split(".")[0] in forbidden:
            pytest.fail(f"graphs/planner.py imported forbidden module: {obj.__name__}")

    # Also check source-level imports via AST / raw source
    src = inspect.getsource(pm)
    for pkg in forbidden:
        assert f"import {pkg}" not in src, f"Found 'import {pkg}' in graphs/planner.py"


# ── Stub (no LLM) — deterministic path ───────────────────────────────


def test_stub_plan_valid():
    """Stub path produces valid plan when no LLM is provided."""
    result = invoke_planner(PlannerInput(objective_text="Build a login page"))
    assert isinstance(result, DeliveryPlan)
    assert result.valid
    assert len(result.packets) == 1
    assert result.packets[0].title != ""


def test_stub_max_packets_respected():
    result = invoke_planner(PlannerInput(objective_text="Do something", max_packets=1))
    assert len(result.packets) <= 1


# ── LLM callable injection — happy path ──────────────────────────────


def test_llm_callable_good_response():
    """Good LLM response → valid plan, revision_count=0."""
    objective = "Add OAuth2 login"
    calls: list[Any] = []

    def mock_llm(messages):
        calls.append(messages)
        return _good_plan_json(objective)

    result = invoke_planner(PlannerInput(objective_text=objective), llm=mock_llm)
    assert result.valid
    assert result.revision_count == 0
    assert len(calls) == 1


# ── AC5: Revision loop fires on malformed LLM output ─────────────────


def test_revision_loop_on_malformed_json():
    """AC5 — malformed JSON triggers revision loop."""
    call_count = 0

    def mock_llm(messages):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _malformed_json()
        return _good_plan_json("Fix the thing")

    result = invoke_planner(
        PlannerInput(objective_text="Fix the thing"),
        llm=mock_llm,
        max_revisions=2,
    )
    assert call_count >= 2


def test_revision_loop_on_generic_criteria():
    """AC5 — plan with generic criteria triggers revision loop."""
    call_count = 0

    def mock_llm(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _bad_plan_json()
        return _good_plan_json("Ship the feature")

    result = invoke_planner(
        PlannerInput(objective_text="Ship the feature"),
        llm=mock_llm,
        max_revisions=2,
    )
    assert call_count == 2
    assert result.revision_count >= 1


# ── AC6: Finalize path on budget exhaustion ───────────────────────────


def test_finalize_on_revision_exhaustion():
    """AC6 — always-bad LLM exhausts budget; finalize returns invalid plan."""

    def mock_llm(messages):
        return _malformed_json()

    result = invoke_planner(
        PlannerInput(objective_text="Unreachable goal"),
        llm=mock_llm,
        max_revisions=2,
    )
    assert not result.valid
    assert result.validation_errors


def test_finalize_bad_plan_with_revision_count():
    """AC6 — revision_count equals max_revisions after exhaustion."""

    def mock_llm(messages):
        return _bad_plan_json()

    result = invoke_planner(
        PlannerInput(objective_text="Never satisfied"),
        llm=mock_llm,
        max_revisions=2,
    )
    # revision_count should reflect exhausted budget
    assert result.revision_count >= 2


# ── Blocking 3: Malformed packet schema → validation, not crash ───────


def test_malformed_packet_missing_title():
    """Packet with no title triggers revision, not crash."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": [
                {
                    "objective": "Implement the full feature end to end",
                    "assigned_role": "worker",
                    "in_scope": ["Build it"],
                    "acceptance_criteria": ["Tests pass"],
                }
            ],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    # Should not crash — missing title caught by validation
    assert isinstance(result, DeliveryPlan)
    assert not result.valid or result.revision_count >= 1


def test_malformed_packet_not_a_dict():
    """Packet that is a string instead of dict → validation error, not crash."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": ["this is not a dict"],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    assert isinstance(result, DeliveryPlan)
    assert not result.valid


def test_malformed_packet_wrong_field_types():
    """Packet with wrong field types (e.g. int objective) → handled gracefully."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": [
                {
                    "title": 12345,
                    "objective": True,
                    "assigned_role": "worker",
                    "in_scope": "not a list",
                    "acceptance_criteria": 42,
                }
            ],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    assert isinstance(result, DeliveryPlan)
    assert not result.valid


def test_malformed_packet_extra_unknown_keys():
    """Packet with extra unknown keys → still parsed or caught, not crash."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": [
                {
                    "title": "Good title",
                    "objective": "Implement full feature with auth",
                    "assigned_role": "worker",
                    "in_scope": ["Build auth module"],
                    "acceptance_criteria": ["Auth tests pass"],
                    "unknown_field": "surprise",
                }
            ],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    assert isinstance(result, DeliveryPlan)
    # Either valid (extra keys ignored) or invalid (schema error), but never crash


def test_malformed_packet_objective_too_short():
    """Packet with objective under 5 chars → validation error."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": [
                {
                    "title": "Short",
                    "objective": "Hi",
                    "assigned_role": "worker",
                    "in_scope": ["Build it"],
                    "acceptance_criteria": ["Tests pass"],
                }
            ],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    assert isinstance(result, DeliveryPlan)
    assert not result.valid


def test_malformed_packet_empty_title_string():
    """Packet with empty string title → validation error."""

    def mock_llm(messages):
        return json.dumps({
            "milestone_title": "M",
            "issue_title": "Issue",
            "issue_body": "body",
            "packets": [
                {
                    "title": "",
                    "objective": "Implement the full feature end to end",
                    "assigned_role": "worker",
                    "in_scope": ["Build it"],
                    "acceptance_criteria": ["Tests pass"],
                }
            ],
        })

    result = invoke_planner(PlannerInput(objective_text="Test"), llm=mock_llm, max_revisions=1)
    assert isinstance(result, DeliveryPlan)
    assert not result.valid


# ── Graph structure ───────────────────────────────────────────────────


def test_graph_has_expected_nodes():
    graph = build_planner_graph()
    compiled = graph.compile()
    # LangGraph compiled graphs expose nodes via the graph attribute
    node_names = set(compiled.get_graph().nodes.keys())
    for expected in ("decompose", "validate", "finalize"):
        assert expected in node_names, f"Missing node: {expected}"
