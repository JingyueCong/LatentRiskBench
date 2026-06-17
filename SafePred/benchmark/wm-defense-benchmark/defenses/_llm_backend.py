"""Shared OpenRouter/custom LLM-client plumbing for the strong baselines.

The recognized-defense baselines (strong LLM judge, Llama Guard, GuardAgent-style
trajectory guardrail) all speak the OpenAI-compatible protocol via OpenRouter, so
they follow the same credential convention as the world-model oracle:
``CUSTOM_API_KEY`` / ``CUSTOM_API_URL`` (OpenAI env vars as a fallback).

Each handler caches verdicts in-memory per instance and degrades gracefully to
"allow" when no key is configured, surfacing the reason for audit.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def make_client(model_name: str, *, temperature: float, max_tokens: int, timeout: int):
    """Build an LLMClient against OpenRouter (provider=custom). Returns (client, error)."""
    api_key = os.environ.get("CUSTOM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_url = (
        os.environ.get("CUSTOM_API_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        return None, "missing CUSTOM_API_KEY/OPENAI_API_KEY"
    try:
        from SafePred.utils.llm_client import LLMClient  # type: ignore
    except Exception:
        try:
            from utils.llm_client import LLMClient  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return None, f"LLMClient import failed: {exc}"
    try:
        client = LLMClient(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            provider="custom",  # OpenAI-compatible over requests
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return client, None
    except Exception as exc:  # noqa: BLE001
        return None, f"LLMClient init failed: {exc}"


def load_policy_list() -> List[Dict[str, Any]]:
    from policy_oracle import load_policies  # type: ignore

    benchmark_root = Path(__file__).resolve().parents[1]
    return load_policies(benchmark_root / "data" / "policies.json")


def format_policies(policies: List[Dict[str, Any]]) -> str:
    lines = []
    for p in policies:
        lines.append(
            f"- {p.get('policy_id', '?')}: {str(p.get('policy_description', '')).strip()}"
        )
    return "\n".join(lines)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def state_text(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return str(state or "")
    return "\n".join(
        str(state.get(k, ""))
        for k in ("a11y_tree_txt", "url", "page_type")
        if state.get(k)
    )
