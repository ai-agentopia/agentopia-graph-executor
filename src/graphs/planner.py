"""Planner graph — decomposes objective into a delivery plan.

IMPORT RULES: domain types only. Zero I/O, zero HTTP, zero provider imports.
LLM callable is injected at invocation time via PlannerState.input["_llm"].

Nodes: decompose → validate → finalize | (retry → decompose)
Max revision loops: 2 (configurable via max_revisions)
Constraint: max_packets=1 (Wave C scope rule)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# ── Domain Types ─────────────────────────────────────────────────────


class PacketPlan(BaseModel):
    """A single planned work packet."""

    title: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=5)
    assigned_role: str = "worker"
    in_scope: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class DeliveryPlan(BaseModel):
    """Output of the planner graph."""

    milestone_title: str = ""
    issue_title: str = ""
    issue_body: str = ""
    packets: list[PacketPlan] = Field(default_factory=list)
    revision_count: int = 0
    valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)


@dataclass
class PlannerInput:
    """Input consumed by invoke_planner."""

    objective_text: str = ""
    workflow_id: str = ""
    owner: str = ""
    repo: str = ""
    max_packets: int = 1


# ── Graph State ──────────────────────────────────────────────────────


class PlannerState(BaseModel):
    """Internal state threaded through every node."""

    input: dict[str, Any] = Field(default_factory=dict)
    raw_plan: dict[str, Any] = Field(default_factory=dict)
    plan: DeliveryPlan | None = None
    revision_count: int = 0
    max_revisions: int = 2
    done: bool = False


# ── Prompt helpers ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a technical planning assistant. Given an objective, produce a structured delivery plan.

Return ONLY valid JSON matching this schema:
{
  "milestone_title": "string (concise milestone label)",
  "issue_title": "string (clear, specific issue title)",
  "issue_body": "string (markdown body with context)",
  "packets": [
    {
      "title": "string (short packet title)",
      "objective": "string (min 10 chars, specific)",
      "assigned_role": "worker",
      "in_scope": ["string (specific deliverable)", ...],
      "acceptance_criteria": ["string (verifiable criterion)", ...]
    }
  ]
}

Rules:
- Produce exactly 1 packet
- in_scope must have ≥1 specific item (not just "Implement the objective")
- acceptance_criteria must have ≥1 verifiable item (not just "Complete the objective")
- issue_title must be specific and non-empty
"""


def _build_user_prompt(objective: str) -> str:
    return f"Objective: {objective}"


# ── Node functions ────────────────────────────────────────────────────


def decompose(state: PlannerState) -> dict[str, Any]:
    """Decompose objective into raw plan dict. Uses LLM callable if provided."""
    objective: str = state.input.get("objective_text", "")
    llm: Callable | None = state.input.get("_llm")

    if llm:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(objective)},
        ]
        raw_text: str = llm(messages)
        try:
            raw = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            # LLM returned non-JSON — treat as empty to trigger revision
            raw = {}
    else:
        # Deterministic stub (tests / CI without LLM)
        raw = {
            "milestone_title": f"Deliver: {objective[:50]}",
            "issue_title": objective[:100],
            "issue_body": f"Objective: {objective}",
            "packets": [
                {
                    "title": objective[:80],
                    "objective": objective,
                    "assigned_role": "worker",
                    "in_scope": [f"Implement: {objective[:60]}"],
                    "acceptance_criteria": [f"Verify completion of: {objective[:60]}"],
                }
            ],
        }

    return {"raw_plan": raw}


_GENERIC_SCOPE = {"implement the objective", "complete the objective"}


def _is_generic(items: list[str]) -> bool:
    return all(s.strip().lower() in _GENERIC_SCOPE for s in items) if items else True


def validate(state: PlannerState) -> dict[str, Any]:
    """Validate plan constraints. Returns revision signal or triggers finalize."""
    raw = state.raw_plan
    errors: list[str] = []
    max_packets: int = state.input.get("max_packets", 1)

    packets = raw.get("packets", [])

    if not raw.get("issue_title", "").strip():
        errors.append("Missing or empty issue_title")
    if not packets:
        errors.append("No packets in plan")
    if len(packets) > max_packets:
        errors.append(f"Too many packets: {len(packets)} > {max_packets}")

    for i, p in enumerate(packets[:max_packets]):
        if not isinstance(p, dict):
            errors.append(f"Packet {i}: not a dict")
            continue
        title = p.get("title")
        if not title or not isinstance(title, str) or not title.strip():
            errors.append(f"Packet {i}: missing or empty title")
        objective = p.get("objective")
        if not objective or not isinstance(objective, str) or not objective.strip():
            errors.append(f"Packet {i}: missing objective")
        elif len(objective.strip()) < 5:
            errors.append(f"Packet {i}: objective too short (min 5 chars)")
        criteria = p.get("acceptance_criteria", [])
        if not isinstance(criteria, list):
            errors.append(f"Packet {i}: acceptance_criteria is not a list")
        elif not criteria:
            errors.append(f"Packet {i}: empty acceptance_criteria")
        elif _is_generic(criteria):
            errors.append(f"Packet {i}: acceptance_criteria are too generic")
        in_scope = p.get("in_scope", [])
        if not isinstance(in_scope, list):
            errors.append(f"Packet {i}: in_scope is not a list")
        elif not in_scope:
            errors.append(f"Packet {i}: empty in_scope")
        elif _is_generic(in_scope):
            errors.append(f"Packet {i}: in_scope is too generic")

    if errors and state.revision_count < state.max_revisions:
        # Signal retry — do NOT build plan yet
        return {"revision_count": state.revision_count + 1, "plan": None}

    # Build validated packets — catch model construction errors
    valid_packets: list[PacketPlan] = []
    for i, p in enumerate(packets[:max_packets]):
        try:
            valid_packets.append(PacketPlan(**p))
        except Exception as exc:
            errors.append(f"Packet {i}: schema error — {exc}")

    plan = DeliveryPlan(
        milestone_title=raw.get("milestone_title", ""),
        issue_title=raw.get("issue_title", ""),
        issue_body=raw.get("issue_body", ""),
        packets=valid_packets,
        revision_count=state.revision_count,
        valid=len(errors) == 0,
        validation_errors=errors,
    )
    return {"plan": plan, "done": True}


def finalize(state: PlannerState) -> dict[str, Any]:
    """Best-effort plan when revision budget is exhausted and plan still None."""
    # plan should already be set by validate on budget exhaustion — this node
    # handles the edge case where validate returned None plan without setting done.
    raw = state.raw_plan
    packets = raw.get("packets", [])
    max_packets: int = state.input.get("max_packets", 1)

    valid_packets: list[PacketPlan] = []
    for p in packets[:max_packets]:
        try:
            valid_packets.append(PacketPlan(**p))
        except Exception:
            pass  # best-effort: skip malformed packets in finalize

    plan = DeliveryPlan(
        milestone_title=raw.get("milestone_title", ""),
        issue_title=raw.get("issue_title", "Untitled"),
        issue_body=raw.get("issue_body", ""),
        packets=valid_packets,
        revision_count=state.revision_count,
        valid=False,
        validation_errors=["Revision budget exhausted — plan may be incomplete"],
    )
    return {"plan": plan, "done": True}


def _route(state: PlannerState) -> str:
    """Conditional edge after validate."""
    if state.done or state.plan is not None:
        return END
    if state.revision_count >= state.max_revisions:
        return "finalize"
    return "decompose"


# ── Graph factory ─────────────────────────────────────────────────────


def build_planner_graph() -> StateGraph:
    graph = StateGraph(PlannerState)
    graph.add_node("decompose", decompose)
    graph.add_node("validate", validate)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "validate")
    graph.add_conditional_edges("validate", _route, {END: END, "decompose": "decompose", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph


# ── Public invocation helper ──────────────────────────────────────────


def invoke_planner(
    planner_input: PlannerInput,
    llm: Callable | None = None,
    max_revisions: int = 2,
) -> DeliveryPlan:
    """Compile and run the planner graph; return a DeliveryPlan."""
    graph = build_planner_graph()
    app = graph.compile()

    initial_state = PlannerState(
        input={
            "objective_text": planner_input.objective_text,
            "workflow_id": planner_input.workflow_id,
            "owner": planner_input.owner,
            "repo": planner_input.repo,
            "max_packets": planner_input.max_packets,
            "_llm": llm,
        },
        max_revisions=max_revisions,
    )

    result = app.invoke(initial_state)
    plan = result.get("plan")
    if plan is None:
        return DeliveryPlan(valid=False, validation_errors=["Plan generation failed"])
    return plan
