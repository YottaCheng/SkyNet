"""Output-path guards for the attack laboratory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

OUTPUTS_ROOT = Path("/Users/ziyaoch/ucl/dissertation/05_outputs")
ATTACK_LAB_ROOT = OUTPUTS_ROOT / "attack_lab"

DEFAULT_C1_ARTEFACT_DIR = (
    OUTPUTS_ROOT
    / "xgboost_challenge"
    / "xgboost_bounded_challenge_2026-07-30"
    / "final_month6"
)

PROTECTED_PREFIXES = (
    OUTPUTS_ROOT / "logistic_baseline",
    OUTPUTS_ROOT / "xgboost_baseline",
    OUTPUTS_ROOT / "xgboost_stability",
    OUTPUTS_ROOT / "xgboost_challenge",
)


class AttackLabPathError(RuntimeError):
    """Raised when an output path would overwrite protected artefacts."""


def new_run_directory(
    run_id: str | None = None,
    *,
    parent: Path | None = None,
) -> Path:
    """Create a uniquely named run directory under attack_lab outputs.

    When ``parent`` is supplied, the new directory is created beneath that
    parent (used for batch per-case subdirectories). Existing paths are
    never overwritten.
    """
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"attack_lab_{stamp}_{uuid.uuid4().hex[:8]}"
    root = parent if parent is not None else ATTACK_LAB_ROOT
    if parent is not None:
        assert_under_attack_lab(parent)
    path = root / run_id
    if path.exists():
        raise AttackLabPathError(f"Refusing to overwrite existing run directory: {path}")
    assert_not_protected(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def assert_under_attack_lab(path: Path) -> None:
    """Ensure batch parents remain inside the attack_lab output tree."""
    resolved = path.resolve()
    root = ATTACK_LAB_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise AttackLabPathError(
            f"Batch parent must live under {root}; got {resolved}"
        )


def assert_not_protected(path: Path) -> None:
    """Refuse writes into existing experimental artefact trees."""
    resolved = path.resolve()
    for prefix in PROTECTED_PREFIXES:
        root = prefix.resolve()
        if resolved == root or root in resolved.parents:
            raise AttackLabPathError(
                f"Refusing to write under protected artefact tree {root}: {resolved}"
            )
