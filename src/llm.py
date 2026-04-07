"""LLM callable factory.

Routes through agentopia-llm-proxy (central LLM egress gateway).
Authenticates with AGENTOPIA_RELAY_TOKEN (Bearer header).

Environment variables:
  LLM_BASE_URL         llm-proxy base URL (required in cluster runtime)
  LLM_API_KEY          AGENTOPIA_RELAY_TOKEN for llm-proxy auth (required)
  GRAPH_MODEL          Model identifier with provider prefix (default: openrouter/google/gemini-2.0-flash-001)
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

    In cluster runtime, this targets agentopia-llm-proxy with
    AGENTOPIA_RELAY_TOKEN as Bearer auth. The proxy routes to the
    appropriate provider based on the model prefix.

    The returned callable accepts a list of message dicts and returns the
    assistant reply as a plain string.
    """
    resolved_url = base_url or os.environ["LLM_BASE_URL"]
    resolved_key = api_key or os.environ["LLM_API_KEY"]
    resolved_model = model or os.environ.get("GRAPH_MODEL", "openrouter/google/gemini-2.0-flash-001")
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
