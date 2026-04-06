"""LLM callable factory.

Reads configuration from environment and returns a callable that
the graph nodes can use for LLM inference.

Environment variables:
  LLM_BASE_URL         Base URL of the LLM-compatible API (required)
  LLM_API_KEY          Bearer token for the LLM API (required)
  GRAPH_MODEL          Model identifier (default: gpt-4o-mini)
  GRAPH_LLM_TIMEOUT    Request timeout in seconds (default: 60)
"""

from __future__ import annotations

import os
from typing import Callable

import httpx


def build_llm_callable(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float | None = None,
) -> Callable[[list[dict]], str]:
    """Return a callable that sends messages to an OpenAI-compatible API.

    The returned callable accepts a list of message dicts and returns the
    assistant reply as a plain string.
    """
    resolved_url = base_url or os.environ["LLM_BASE_URL"]
    resolved_key = api_key or os.environ["LLM_API_KEY"]
    resolved_model = model or os.environ.get("GRAPH_MODEL", "gpt-4o-mini")
    resolved_timeout = timeout or float(os.environ.get("GRAPH_LLM_TIMEOUT", "60"))

    def _call(messages: list[dict]) -> str:
        endpoint = resolved_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=resolved_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return _call
