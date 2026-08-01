"""Inspectable proposal domains for constrained random (A0) sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from baf_data.config import FROZEN_CONFIG, DataLayerConfig

DomainKind = Literal["binary", "categorical", "float", "integer"]


class AttackLabDomainError(RuntimeError):
    """Raised when a mutable field lacks a verified sampling domain."""


@dataclass(frozen=True)
class FieldDomain:
    """One inspectable sampling domain for a mutable field."""

    field: str
    kind: DomainKind
    values: tuple[Any, ...] | None = None
    low: float | None = None
    high: float | None = None
    source: str = ""

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "binary":
            assert self.values is not None
            return int(rng.choice(self.values))
        if self.kind == "categorical":
            assert self.values is not None
            return str(rng.choice(self.values))
        if self.kind == "integer":
            assert self.low is not None and self.high is not None
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return float(rng.uniform(self.low, self.high))
        raise AttackLabDomainError(f"Unsupported domain kind {self.kind!r}.")


@dataclass(frozen=True)
class ProposalDomainSet:
    """Complete domain map for the mutable fields of one A0 episode."""

    domains: dict[str, FieldDomain]
    config_label: str
    config_path: str | None

    def require(self, field: str) -> FieldDomain:
        if field not in self.domains:
            raise AttackLabDomainError(
                f"Missing sampling domain for mutable field '{field}'. "
                "Supply it via the development-only A0 domains configuration."
            )
        return self.domains[field]

    def sample_changes(
        self,
        mutable_fields: tuple[str, ...],
        baseline: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        max_redraws: int = 8,
    ) -> dict[str, Any]:
        """Sample a full independent proposal over all mutable fields.

        Development decision: every attempt redraws all mutable fields from the
        original baseline independently (no carry-over from failed candidates).
        """
        changes: dict[str, Any] = {}
        for field in mutable_fields:
            domain = self.require(field)
            value = domain.sample(rng)
            # Prefer a value that differs from the baseline when possible.
            for _ in range(max_redraws):
                if not _equal(value, baseline.get(field)):
                    break
                value = domain.sample(rng)
            changes[field] = value
        return changes


def load_numeric_domains_config(path: Path) -> dict[str, Any]:
    """Load a development-only numeric domains JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "numeric_domains" not in payload:
        raise AttackLabDomainError(
            f"A0 domains file missing 'numeric_domains': {path}"
        )
    return payload


def build_proposal_domains(
    mutable_fields: tuple[str, ...] | list[str],
    *,
    categorical_vocabularies: Mapping[str, tuple[Any, ...]],
    numeric_domains_config: Mapping[str, Any] | None,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    config_path: str | None = None,
) -> ProposalDomainSet:
    """Assemble domains for every mutable field or raise a blocker.

    Sources, in order of preference for each field:
    1. frozen binary schema {0, 1};
    2. fitted categorical encoder vocabulary;
    3. explicit development-only numeric_domains configuration.
    """
    mutable = tuple(mutable_fields)
    kinds = {spec.name: spec.kind for spec in data_config.raw_columns}
    binary = set(data_config.binary_features)
    numeric_cfg = dict((numeric_domains_config or {}).get("numeric_domains", {}))
    label = str(
        (numeric_domains_config or {}).get(
            "label", "inline_or_default_development_domains"
        )
    )

    domains: dict[str, FieldDomain] = {}
    missing: list[str] = []

    for field in mutable:
        if field not in data_config.feature_columns:
            raise AttackLabDomainError(
                f"Mutable field '{field}' is not in the frozen feature schema."
            )
        if field in binary:
            domains[field] = FieldDomain(
                field=field,
                kind="binary",
                values=(0, 1),
                source="frozen_schema:binary_features",
            )
            continue
        if kinds[field] == "string":
            vocab = categorical_vocabularies.get(field)
            if not vocab:
                missing.append(field + " (categorical vocabulary unavailable)")
                continue
            domains[field] = FieldDomain(
                field=field,
                kind="categorical",
                values=tuple(str(v) for v in vocab),
                source="fitted_c1_onehot_vocabulary",
            )
            continue
        # numeric
        if field not in numeric_cfg:
            missing.append(field + " (numeric domain not in development config)")
            continue
        spec = numeric_cfg[field]
        kind = spec.get("kind")
        if kind not in {"float", "integer"}:
            raise AttackLabDomainError(
                f"Numeric domain for '{field}' must declare kind float|integer."
            )
        if "low" not in spec or "high" not in spec:
            raise AttackLabDomainError(
                f"Numeric domain for '{field}' must declare low and high."
            )
        domains[field] = FieldDomain(
            field=field,
            kind=kind,
            low=float(spec["low"]),
            high=float(spec["high"]),
            source=str(spec.get("source", "development_numeric_domains_config")),
        )

    if missing:
        raise AttackLabDomainError(
            "A0 cannot sample without verified domains for: "
            + ", ".join(missing)
            + ". Provide them in the development-only A0 domains configuration; "
            "do not silently invent final scientific domains."
        )

    return ProposalDomainSet(
        domains=domains,
        config_label=label,
        config_path=config_path,
    )


def _equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        if left != left and right != right:  # NaN
            return True
    except Exception:  # noqa: BLE001
        pass
    return left == right
