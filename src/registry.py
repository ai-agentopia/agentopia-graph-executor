"""Static graph registry.

Maps graph_name → invocation callable. New graphs are registered here.
The registry is the single source of truth for which graphs the service exposes.
"""

from __future__ import annotations

from typing import Any, Callable

from graphs.planner import DeliveryPlan, PlannerInput, invoke_planner
from graphs.reviewer import ReviewAnalysis, ReviewInput, invoke_reviewer


def _invoke_planner(input_data: dict[str, Any], llm: Callable | None) -> dict[str, Any]:
    planner_input = PlannerInput(
        objective_text=input_data.get("objective_text", ""),
        workflow_id=input_data.get("workflow_id", ""),
        owner=input_data.get("owner", ""),
        repo=input_data.get("repo", ""),
        max_packets=input_data.get("max_packets", 1),
    )
    plan: DeliveryPlan = invoke_planner(planner_input, llm=llm)
    return plan.model_dump()


def _invoke_reviewer(input_data: dict[str, Any], llm: Callable | None) -> dict[str, Any]:
    review_input = ReviewInput(
        pr_diff=input_data.get("pr_diff", ""),
        acceptance_criteria=input_data.get("acceptance_criteria", []),
        pr_title=input_data.get("pr_title", ""),
        pr_body=input_data.get("pr_body", ""),
        file_names=input_data.get("file_names", []),
        rework_round=input_data.get("rework_round", 0),
        prior_comments=input_data.get("prior_comments", []),
    )
    result: ReviewAnalysis = invoke_reviewer(review_input, llm=llm)
    return result.model_dump()


# graph_name → handler(input_data, llm) → dict
REGISTRY: dict[str, Callable[[dict[str, Any], Callable | None], dict[str, Any]]] = {
    "planner": _invoke_planner,
    "reviewer-shadow": _invoke_reviewer,
}

GRAPH_NAMES: list[str] = list(REGISTRY.keys())
