"""Reviewer shadow graph — analyzes PR diff against acceptance criteria.

IMPORT RULES: domain types only. Zero I/O, zero HTTP, zero provider imports.
LLM callable is injected at invocation time via ReviewerState.input["_llm"].

Nodes: analyze → (conditional) reconcile_prior | decide → decide → END
Output: reasoning-only ReviewAnalysis — no GitHub I/O, no persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# ── Domain Types ─────────────────────────────────────────────────────


class ReviewFinding(BaseModel):
    """A single review finding."""

    file: str = ""
    line: int = 0
    body: str = ""
    severity: str = Field(default="info", pattern=r"^(info|warning|error)$")
    category: str = Field(
        default="code_quality",
        pattern=r"^(criteria|code_quality|security|style)$",
    )


class ReviewAnalysis(BaseModel):
    """Output of the reviewer shadow graph."""

    verdict: str = Field(default="REQUEST_CHANGES", pattern=r"^(APPROVE|REQUEST_CHANGES)$")
    summary: str = ""
    findings: list[ReviewFinding] = Field(default_factory=list)
    criteria_met: list[str] = Field(default_factory=list)
    criteria_unmet: list[str] = Field(default_factory=list)
    is_rework: bool = False
    prior_issues_addressed: list[str] = Field(default_factory=list)
    prior_issues_remaining: list[str] = Field(default_factory=list)


@dataclass
class ReviewInput:
    """Input consumed by invoke_reviewer."""

    pr_diff: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    pr_title: str = ""
    pr_body: str = ""
    file_names: list[str] = field(default_factory=list)
    rework_round: int = 0
    prior_comments: list[str] = field(default_factory=list)


# ── Graph State ──────────────────────────────────────────────────────


class ReviewerState(BaseModel):
    """Internal state threaded through every node."""

    input: dict[str, Any] = Field(default_factory=dict)
    raw_analysis: dict[str, Any] = Field(default_factory=dict)
    reconciled: dict[str, Any] = Field(default_factory=dict)
    result: ReviewAnalysis | None = None
    done: bool = False


# ── Prompt helpers ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior code reviewer. Analyze the PR diff against the acceptance criteria.

Return ONLY valid JSON matching this schema:
{
  "verdict": "APPROVE" or "REQUEST_CHANGES",
  "summary": "string (brief reasoning summary)",
  "findings": [
    {
      "file": "string (file path)",
      "line": 0,
      "body": "string (finding description)",
      "severity": "info" | "warning" | "error",
      "category": "criteria" | "code_quality" | "security" | "style"
    }
  ],
  "criteria_met": ["string (criterion that passed)", ...],
  "criteria_unmet": ["string (criterion that failed)", ...]
}

Rules:
- APPROVE only if ALL criteria are met and no error-severity findings exist
- REQUEST_CHANGES if any criteria are unmet or error-severity findings exist
- Be specific in findings — reference file names and line numbers when possible
- Classify findings by category: criteria compliance, code quality, security, style
"""

_RECONCILE_PROMPT_TEMPLATE = """\
You are reviewing a rework submission. Compare the current diff against prior review comments.

Prior review comments:
{prior_comments}

Return ONLY valid JSON:
{{
  "prior_issues_addressed": ["string (prior comment that is now fixed)", ...],
  "prior_issues_remaining": ["string (prior comment still unresolved)", ...]
}}
"""


def _build_review_prompt(
    pr_diff: str,
    criteria: list[str],
    pr_title: str,
    pr_body: str,
    file_names: list[str],
) -> str:
    criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "- (none provided)"
    files_text = "\n".join(f"- {f}" for f in file_names) if file_names else "- (none)"
    return (
        f"PR Title: {pr_title}\n"
        f"PR Body: {pr_body}\n\n"
        f"Changed files:\n{files_text}\n\n"
        f"Acceptance Criteria:\n{criteria_text}\n\n"
        f"Diff:\n```\n{pr_diff}\n```"
    )


# ── Node functions ────────────────────────────────────────────────────


def analyze(state: ReviewerState) -> dict[str, Any]:
    """Examine diff against criteria. Uses LLM callable if provided."""
    inp = state.input
    pr_diff: str = inp.get("pr_diff", "")
    criteria: list[str] = inp.get("acceptance_criteria", [])
    pr_title: str = inp.get("pr_title", "")
    pr_body: str = inp.get("pr_body", "")
    file_names: list[str] = inp.get("file_names", [])
    llm: Callable | None = inp.get("_llm")

    if llm:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_review_prompt(pr_diff, criteria, pr_title, pr_body, file_names),
            },
        ]
        raw_text: str = llm(messages)
        try:
            raw = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            # LLM returned non-JSON — produce REQUEST_CHANGES with error finding
            raw = {
                "verdict": "REQUEST_CHANGES",
                "summary": "Failed to parse LLM analysis response",
                "findings": [
                    {
                        "file": "",
                        "line": 0,
                        "body": "LLM returned malformed response; manual review required",
                        "severity": "error",
                        "category": "code_quality",
                    }
                ],
                "criteria_met": [],
                "criteria_unmet": list(criteria),
            }
    else:
        # Deterministic stub (tests / CI without LLM)
        if pr_diff and criteria:
            raw = {
                "verdict": "APPROVE",
                "summary": "All criteria met (stub mode)",
                "findings": [],
                "criteria_met": list(criteria),
                "criteria_unmet": [],
            }
        else:
            raw = {
                "verdict": "REQUEST_CHANGES",
                "summary": "Empty diff or missing criteria (stub mode)",
                "findings": [
                    {
                        "file": "",
                        "line": 0,
                        "body": "Empty diff or no acceptance criteria provided",
                        "severity": "error",
                        "category": "criteria",
                    }
                ],
                "criteria_met": [],
                "criteria_unmet": list(criteria) if criteria else ["No criteria provided"],
            }

    return {"raw_analysis": raw}


def reconcile_prior(state: ReviewerState) -> dict[str, Any]:
    """Compare current diff against prior review comments (rework path)."""
    inp = state.input
    prior_comments: list[str] = inp.get("prior_comments", [])
    pr_diff: str = inp.get("pr_diff", "")
    llm: Callable | None = inp.get("_llm")

    if llm:
        prompt = _RECONCILE_PROMPT_TEMPLATE.format(
            prior_comments="\n".join(f"- {c}" for c in prior_comments),
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Current diff:\n```\n{pr_diff}\n```"},
        ]
        raw_text: str = llm(messages)
        try:
            reconciled = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            reconciled = {
                "prior_issues_addressed": [],
                "prior_issues_remaining": list(prior_comments),
            }
    else:
        # Deterministic stub: if diff exists, assume all prior issues addressed
        if pr_diff:
            reconciled = {
                "prior_issues_addressed": list(prior_comments),
                "prior_issues_remaining": [],
            }
        else:
            reconciled = {
                "prior_issues_addressed": [],
                "prior_issues_remaining": list(prior_comments),
            }

    return {"reconciled": reconciled}


def decide(state: ReviewerState) -> dict[str, Any]:
    """Produce final verdict and structured ReviewAnalysis."""
    raw = state.raw_analysis
    reconciled = state.reconciled
    rework_round: int = state.input.get("rework_round", 0)
    is_rework = rework_round > 0

    # Parse findings
    findings: list[ReviewFinding] = []
    for f in raw.get("findings", []):
        try:
            findings.append(ReviewFinding(**f))
        except Exception:
            pass  # skip malformed findings

    # Collect structured evidence
    criteria_unmet = raw.get("criteria_unmet", [])
    prior_remaining = reconciled.get("prior_issues_remaining", [])
    has_error_findings = any(f.severity == "error" for f in findings)

    # Normalize verdict against evidence — never APPROVE with contradicting evidence
    raw_verdict = raw.get("verdict", "REQUEST_CHANGES")
    if raw_verdict not in ("APPROVE", "REQUEST_CHANGES"):
        raw_verdict = "REQUEST_CHANGES"

    if criteria_unmet:
        verdict = "REQUEST_CHANGES"
    elif has_error_findings:
        verdict = "REQUEST_CHANGES"
    elif is_rework and prior_remaining:
        verdict = "REQUEST_CHANGES"
    else:
        verdict = raw_verdict

    result = ReviewAnalysis(
        verdict=verdict,
        summary=raw.get("summary", ""),
        findings=findings,
        criteria_met=raw.get("criteria_met", []),
        criteria_unmet=criteria_unmet,
        is_rework=is_rework,
        prior_issues_addressed=reconciled.get("prior_issues_addressed", []),
        prior_issues_remaining=prior_remaining,
    )

    return {"result": result, "done": True}


def _route_after_analyze(state: ReviewerState) -> str:
    """Conditional edge after analyze."""
    rework_round: int = state.input.get("rework_round", 0)
    if rework_round > 0:
        return "reconcile_prior"
    return "decide"


# ── Graph factory ─────────────────────────────────────────────────────


def build_reviewer_graph() -> StateGraph:
    graph = StateGraph(ReviewerState)
    graph.add_node("analyze", analyze)
    graph.add_node("reconcile_prior", reconcile_prior)
    graph.add_node("decide", decide)

    graph.set_entry_point("analyze")
    graph.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"reconcile_prior": "reconcile_prior", "decide": "decide"},
    )
    graph.add_edge("reconcile_prior", "decide")
    graph.add_edge("decide", END)

    return graph


# ── Public invocation helper ──────────────────────────────────────────


def invoke_reviewer(
    review_input: ReviewInput,
    llm: Callable | None = None,
) -> ReviewAnalysis:
    """Compile and run the reviewer graph; return a ReviewAnalysis."""
    graph = build_reviewer_graph()
    app = graph.compile()

    initial_state = ReviewerState(
        input={
            "pr_diff": review_input.pr_diff,
            "acceptance_criteria": review_input.acceptance_criteria,
            "pr_title": review_input.pr_title,
            "pr_body": review_input.pr_body,
            "file_names": review_input.file_names,
            "rework_round": review_input.rework_round,
            "prior_comments": review_input.prior_comments,
            "_llm": llm,
        },
    )

    result = app.invoke(initial_state)
    analysis = result.get("result")
    if analysis is None:
        return ReviewAnalysis(
            verdict="REQUEST_CHANGES",
            summary="Review graph failed to produce analysis",
        )
    return analysis
