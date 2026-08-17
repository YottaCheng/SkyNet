"""D2-L V2 discrete-rubric contract. Separate from frozen V1."""

from __future__ import annotations

PROMPT_VERSION = "d2l-v2.0.0-discrete-rubric-sanity-20260816"
SANITY_SEED = 202608162
LEGITIMATE_SAMPLE_N = 100
ATTACK_PER_CONDITION = 5
ATTACK_CONDITIONS: tuple[str, ...] = ("A0", "A1-Pro", "A2", "A3-Pro")
ATTACK_SAMPLE_N = ATTACK_PER_CONDITION * len(ATTACK_CONDITIONS)
SENTINEL_COUNT = 4
TARGET_REVIEW_BUDGET = 0.10
LABELS: tuple[str, ...] = ("GOOD", "UNCERTAIN", "BAD")
LABEL_TO_POINTS: dict[str, int] = {"GOOD": 0, "UNCERTAIN": 1, "BAD": 2}
MIN_TOTAL = 0
MAX_TOTAL = 16

DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("life_stage_coherence", "Life-stage coherence"),
    ("employment_housing_coherence", "Employment / housing coherence"),
    ("address_history_coherence", "Address-history coherence"),
    ("banking_payment_coherence", "Banking / payment coherence"),
    ("contact_coherence", "Contact coherence"),
    ("device_session_request_coherence", "Device / session / request coherence"),
    ("financial_application_coherence", "Financial / application coherence"),
    ("overall_cross_field_coherence", "Overall cross-field coherence"),
)
DIMENSION_IDS: tuple[str, ...] = tuple(item[0] for item in DIMENSIONS)

RUBRIC_MAX_TOKENS = 4000
SCORE_BATCH_MAX_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 2

# Pre-specified sanity gates. Not tuned from attack outcomes.
COLLAPSE_MAX_UNIQUE = 2
COLLAPSE_MODAL_SHARE = 0.80
DISCRIMINATION_MEDIAN_SHIFT = 0.0  # attack median must be strictly greater
DISCRIMINATION_REVIEW_LIFT = 0.10
SENTINEL_MAX_MATERIAL_FAILURES = 0
