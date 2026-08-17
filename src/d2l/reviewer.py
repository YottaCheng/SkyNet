"""One-shot D2-L reviewer: application view -> frozen prompt -> parsed JSON."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from d2l.application_view import (
    application_view,
    application_view_sha256,
    assert_view_has_no_forbidden_fields,
    serialize_application_view,
)
from d2l.client import D2LClient, D2LCompletion
from d2l.contract import MAX_PARSE_RETRIES, MODEL_ID, PROMPT_VERSION
from d2l.errors import D2LParseError, D2LTransportError
from d2l.parser import parse_reviewer_output
from d2l.prompt import build_messages


@dataclass
class D2LReview:
    view_sha256: str
    consistency_risk_score: int
    reason_codes: list[str]
    summary: str
    raw_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_ms: float
    n_attempts: int
    n_parse_failures: int
    n_transport_failures: int
    cached: bool


class JsonlCache:
    """Append-only cache keyed by application-view SHA-256."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._index: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._index[str(record["view_sha256"])] = record

    def get(self, view_sha256: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._index.get(view_sha256)
            return None if payload is None else dict(payload)

    def put(self, record: Mapping[str, Any]) -> None:
        key = str(record["view_sha256"])
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._index[key] = dict(record)


def _review_from_cache(record: Mapping[str, Any]) -> D2LReview:
    return D2LReview(
        view_sha256=str(record["view_sha256"]),
        consistency_risk_score=int(record["consistency_risk_score"]),
        reason_codes=list(record["reason_codes"]),
        summary=str(record["summary"]),
        raw_text=str(record.get("raw_text") or ""),
        model=str(record.get("model") or MODEL_ID),
        prompt_tokens=int(record.get("prompt_tokens") or 0),
        completion_tokens=int(record.get("completion_tokens") or 0),
        total_tokens=int(record.get("total_tokens") or 0),
        cached_tokens=int(record.get("cached_tokens") or 0),
        reasoning_tokens=int(record.get("reasoning_tokens") or 0),
        cost_usd=float(record.get("cost_usd") or 0.0),
        latency_ms=float(record.get("latency_ms") or 0.0),
        n_attempts=int(record.get("n_attempts") or 1),
        n_parse_failures=int(record.get("n_parse_failures") or 0),
        n_transport_failures=int(record.get("n_transport_failures") or 0),
        cached=True,
    )


class D2LReviewer:
    """Pinned one-shot reviewer. No memory, tools, or feedback."""

    def __init__(self, client: D2LClient | None = None, cache: JsonlCache | None = None) -> None:
        self.client = client or D2LClient()
        self.cache = cache

    def review(self, record: Mapping[str, Any], *, use_cache: bool = True) -> D2LReview:
        view = application_view(record)
        blob = serialize_application_view(view)
        assert_view_has_no_forbidden_fields(view, blob)
        view_hash = application_view_sha256(view)
        if use_cache and self.cache is not None:
            hit = self.cache.get(view_hash)
            if hit is not None:
                return _review_from_cache(hit)

        messages = build_messages(view)
        parse_failures = 0
        transport_failures = 0
        last_error: Exception | None = None
        completion: D2LCompletion | None = None
        parsed: dict[str, Any] | None = None
        attempts = MAX_PARSE_RETRIES + 1
        for attempt in range(1, attempts + 1):
            try:
                completion = self.client.complete(messages)
            except D2LTransportError as exc:
                transport_failures += 1
                last_error = exc
                time.sleep(min(8.0, 1.5 ** attempt))
                continue
            try:
                parsed = parse_reviewer_output(completion.text)
                break
            except D2LParseError as exc:
                parse_failures += 1
                last_error = exc
                completion = completion
        if parsed is None or completion is None:
            raise D2LParseError(
                f"D2-L failed after {attempts} attempts "
                f"(parse_failures={parse_failures}, "
                f"transport_failures={transport_failures}): {last_error}"
            )
        review = D2LReview(
            view_sha256=view_hash,
            consistency_risk_score=int(parsed["consistency_risk_score"]),
            reason_codes=list(parsed["reason_codes"]),
            summary=str(parsed["summary"]),
            raw_text=completion.text,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            cached_tokens=completion.cached_tokens,
            reasoning_tokens=completion.reasoning_tokens,
            cost_usd=completion.cost_usd,
            latency_ms=completion.latency_ms,
            n_attempts=parse_failures + transport_failures + 1,
            n_parse_failures=parse_failures,
            n_transport_failures=transport_failures,
            cached=False,
        )
        if self.cache is not None:
            self.cache.put(
                {
                    "view_sha256": review.view_sha256,
                    "consistency_risk_score": review.consistency_risk_score,
                    "reason_codes": review.reason_codes,
                    "summary": review.summary,
                    "raw_text": review.raw_text,
                    "model": review.model,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_tokens": review.prompt_tokens,
                    "completion_tokens": review.completion_tokens,
                    "total_tokens": review.total_tokens,
                    "cached_tokens": review.cached_tokens,
                    "reasoning_tokens": review.reasoning_tokens,
                    "cost_usd": review.cost_usd,
                    "latency_ms": review.latency_ms,
                    "n_attempts": review.n_attempts,
                    "n_parse_failures": review.n_parse_failures,
                    "n_transport_failures": review.n_transport_failures,
                }
            )
        return review
