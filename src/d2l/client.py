"""One-shot DeepSeek client for D2-L. Thinking is always disabled."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from attack_lab.benchmark_pins import MODEL_PRO, estimate_deepseek_cost_usd
from d2l.contract import (
    MAX_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    THINKING_DISABLED,
    TIMEOUT_SECONDS,
    TOP_P,
)
from d2l.errors import D2LContractError, D2LTransportError
from deepseek_config import load_deepseek_settings


def _extract_cached_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return int(cached)
    if isinstance(usage, Mapping):
        details_map = usage.get("prompt_tokens_details") or {}
        if isinstance(details_map, Mapping) and details_map.get("cached_tokens") is not None:
            return int(details_map["cached_tokens"])
    return 0


def _extract_reasoning_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    for attr in ("reasoning_tokens", "completion_tokens_details"):
        value = getattr(usage, attr, None)
        if attr == "reasoning_tokens" and value is not None:
            return int(value)
        if attr == "completion_tokens_details" and value is not None:
            nested = getattr(value, "reasoning_tokens", None)
            if nested is not None:
                return int(nested)
    if isinstance(usage, Mapping):
        if usage.get("reasoning_tokens") is not None:
            return int(usage["reasoning_tokens"])
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
            return int(details["reasoning_tokens"])
    return 0


@dataclass(frozen=True)
class D2LCompletion:
    text: str
    model: str
    requested_model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    latency_ms: float
    thinking_disabled: bool
    cost_usd: float
    system_fingerprint: str | None


class D2LClient:
    """OpenAI-compatible DeepSeek client pinned to Pro / ThinkOff / JSON."""

    def __init__(self) -> None:
        if MODEL_ID != MODEL_PRO:
            raise D2LContractError(f"D2-L model must be {MODEL_PRO}, got {MODEL_ID}.")
        if not THINKING_DISABLED:
            raise D2LContractError("D2-L requires thinking disabled.")
        settings = load_deepseek_settings()
        self.api_key = settings.api_key
        self.base_url = settings.base_url

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> D2LCompletion:
        from openai import OpenAI

        timeout = float(TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds)
        token_limit = int(MAX_TOKENS if max_tokens is None else max_tokens)
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )
        create_kwargs: dict[str, Any] = {
            "model": MODEL_ID,
            "messages": [dict(item) for item in messages],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": token_limit,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - transport boundary
            raise D2LTransportError(f"DeepSeek request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = ""
        if response.choices:
            message = response.choices[0].message
            text = (getattr(message, "content", None) or "").strip()
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = (
            int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        )
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        cached_tokens = _extract_cached_tokens(usage)
        reasoning_tokens = _extract_reasoning_tokens(usage)
        requested = MODEL_ID
        returned = str(getattr(response, "model", None) or requested)
        fingerprint = getattr(response, "system_fingerprint", None)
        cost = estimate_deepseek_cost_usd(
            model=MODEL_PRO,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        return D2LCompletion(
            text=text,
            model=returned,
            requested_model=requested,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=float(latency_ms),
            thinking_disabled=True,
            cost_usd=float(cost),
            system_fingerprint=None if fingerprint is None else str(fingerprint),
        )
