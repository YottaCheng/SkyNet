"""Frozen D2-L V1 contract: model, prompt pin, schemas, and isolation rules.

Changing any constant here is a contract change and requires a new
``PROMPT_VERSION``.
"""

from __future__ import annotations

from baf_data.config import FROZEN_CONFIG

PROMPT_VERSION = "d2l-v1.0.0-app-consistency-20260816"
REVIEWER_ID = "D2-L"
REVIEWER_STATEMENT = (
    "A one-shot LLM application-consistency reviewer. It assesses whether a "
    "single decision-time application forms a coherent internal profile or "
    "contains cross-field inconsistencies that warrant additional human review. "
    "It is not a second fraud classifier."
)

MODEL_ID = "deepseek-v4-pro"
THINKING_DISABLED = True
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 400
TIMEOUT_SECONDS = 90.0
MAX_PARSE_RETRIES = 2

CALIBRATION_SAMPLE_N = 500
VALIDATION_SAMPLE_N = 500
SANITY_SAMPLE_N = 3
CALIBRATION_SAMPLE_SEED = 20260816
REVIEW_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.15)
PRIMARY_REVIEW_BUDGET = 0.10

N_ANCHORS = 50
EXPECTED_ATTACKS: dict[str, int] = {
    "A0": 26,
    "A1-Pro": 27,
    "A2": 37,
    "A3-Pro": 31,
}
EXPECTED_ATTACK_TOTAL = 121

# Frozen D2-S identity checks for the matched-budget comparison only.
FROZEN_D2S_CONTRACT_ID = "d2s-v1.0.0-pairwise8-20260816"
FROZEN_D2S_FINGERPRINT = (
    "cfd5330f096dabb1749be447ee4da4d5f498d2599f4f22c24a0b706e570bfd94"
)
FROZEN_D2S_THRESHOLDS: dict[float, float] = {
    0.05: 0.6298681955497826,
    0.10: 0.5918014572249943,
    0.15: 0.5720686786860827,
}
EXPECTED_D2S_REVIEW: dict[float, dict[str, int]] = {
    0.05: {"A0": 3, "A1-Pro": 0, "A2": 3, "A3-Pro": 2},
    0.10: {"A0": 4, "A1-Pro": 0, "A2": 9, "A3-Pro": 2},
    0.15: {"A0": 5, "A1-Pro": 3, "A2": 12, "A3-Pro": 3},
}

APPLICATION_FIELDS: tuple[str, ...] = FROZEN_CONFIG.feature_columns

OUTPUT_REQUIRED_KEYS: tuple[str, ...] = (
    "consistency_risk_score",
    "reason_codes",
    "summary",
)

# Keys that must never enter the D2-L application view or prompt.
FORBIDDEN_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "fraud_bool",
        "d1_score",
        "d1_probability",
        "d1_proba",
        "d1_risk_score",
        "d1_threshold",
        "d1_decision",
        "risk_score",
        "y_score",
        "y_true",
        "shap",
        "shap_values",
        "d2_score",
        "d2s_score",
        "relationship_scores",
        "attacker_id",
        "attacker_identity",
        "attacker_kind",
        "attacker_type",
        "attack_history",
        "changed_field_mask",
        "changed_fields",
        "edited_fields",
        "successful_query",
        "provenance",
        "reference_pool",
        "reference_pool_membership",
        "reference_id",
        "attack_success",
        "success",
        "month",
        "row_id",
        "source_row_id",
        "anchor_id",
        "condition_id",
        "record_id",
        "defender_metadata",
        "attack_metadata",
    }
)

# Substrings that must never appear in the frozen prompt.  These block
# D2-S relationship leakage, score leakage, and attacker-side hints.
FORBIDDEN_PROMPT_SUBSTRINGS: tuple[str, ...] = (
    "C01",
    "C14",
    "C13",
    "C09",
    "C03",
    "C10",
    "C11",
    "C15",
    "d2-s",
    "d2s",
    "d2_score",
    "xgboost",
    "shap",
    "fraud_bool",
    "d1_score",
    "d1 score",
    "d1 threshold",
    "attacker",
    "changed-field",
    "changed_field",
    "reference pool",
    "reference-pool",
    "pairwise",
    "youden",
)

GENERIC_REVIEW_DIMENSIONS: tuple[str, ...] = (
    "age / life-stage coherence",
    "employment / housing coherence",
    "address-history coherence",
    "banking / payment coherence",
    "contact configuration",
    "device / session / request context",
    "cross-field consistency across the whole profile",
)

# Official BAF Datasheet feature descriptions (Jesus et al., 2022), restricted
# to decision-time core fields.  Excluded fields are omitted, not described.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "income": "Annual income of the applicant in quantiles. Ranges between 0 and 1.",
    "name_email_similarity": (
        "Metric of similarity between the application email and the applicant's "
        "name. Higher values represent higher similarity. Ranges between 0 and 1."
    ),
    "prev_address_months_count": (
        "Number of months in the applicant's previous registered address, if "
        "applicable. Official missing values appear as null."
    ),
    "current_address_months_count": (
        "Number of months in the applicant's currently registered address. "
        "Official missing values appear as null."
    ),
    "customer_age": "Applicant's age in years, rounded to the decade.",
    "intended_balcon_amount": (
        "Initial transferred amount for the application. Official missing "
        "values appear as null."
    ),
    "payment_type": "Credit payment plan type. Anonymised categorical code.",
    "zip_count_4w": (
        "Number of applications within the same zip code in the last 4 weeks."
    ),
    "velocity_6h": (
        "Velocity of total applications made in the last 6 hours (average "
        "applications per hour). Negative values are valid observed values, "
        "not missing."
    ),
    "velocity_24h": (
        "Velocity of total applications made in the last 24 hours (average "
        "applications per hour)."
    ),
    "velocity_4w": (
        "Velocity of total applications made in the last 4 weeks (average "
        "applications per hour)."
    ),
    "bank_branch_count_8w": (
        "Number of total applications in the selected bank branch in the last "
        "8 weeks."
    ),
    "date_of_birth_distinct_emails_4w": (
        "Number of emails for applicants with the same date of birth in the "
        "last 4 weeks."
    ),
    "employment_status": "Employment status of the applicant. Anonymised categorical code.",
    "email_is_free": "Whether the application email domain is a free email provider (1) or not (0).",
    "housing_status": (
        "Current residential status of the applicant. Anonymised categorical code."
    ),
    "phone_home_valid": "Validity of the provided home phone number (1 valid, 0 not valid).",
    "phone_mobile_valid": "Validity of the provided mobile phone number (1 valid, 0 not valid).",
    "bank_months_count": (
        "How many months old the applicant's previous account with this bank "
        "is, if one exists. Official missing values appear as null."
    ),
    "has_other_cards": (
        "Whether the applicant has other cards from the same banking company "
        "(1 yes, 0 no)."
    ),
    "proposed_credit_limit": "Applicant's proposed credit limit.",
    "foreign_request": (
        "Whether the origin country of the request differs from the bank's "
        "country (1 yes, 0 no)."
    ),
    "source": "Application channel. Typical values include INTERNET and TELEAPP.",
    "session_length_in_minutes": (
        "Length of the user session on the banking website, in minutes. "
        "Official missing values appear as null."
    ),
    "device_os": "Operating system of the device that made the request.",
    "keep_alive_session": "User option on session keep-alive (1 yes, 0 no).",
    "device_distinct_emails_8w": (
        "Number of distinct emails on the banking website from the used device "
        "in the last 8 weeks. Official missing values appear as null."
    ),
}


def contract_payload() -> dict[str, object]:
    """JSON-serialisable snapshot of the frozen D2-L contract."""
    missing = [name for name in APPLICATION_FIELDS if name not in FIELD_DESCRIPTIONS]
    extra = sorted(set(FIELD_DESCRIPTIONS) - set(APPLICATION_FIELDS))
    if missing or extra:
        raise RuntimeError(
            f"Field dictionary mismatch: missing={missing} extra={extra}"
        )
    return {
        "reviewer_id": REVIEWER_ID,
        "prompt_version": PROMPT_VERSION,
        "statement": REVIEWER_STATEMENT,
        "model": MODEL_ID,
        "thinking_disabled": THINKING_DISABLED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "receives_d1_numeric_score": False,
        "receives_d2s_relationships": False,
        "receives_fraud_label": False,
        "one_shot": True,
        "memory": False,
        "tools": False,
        "retrieval": False,
        "application_fields": list(APPLICATION_FIELDS),
        "output_required_keys": list(OUTPUT_REQUIRED_KEYS),
        "forbidden_input_keys": sorted(FORBIDDEN_INPUT_KEYS),
        "generic_review_dimensions": list(GENERIC_REVIEW_DIMENSIONS),
        "calibration_sample_n": CALIBRATION_SAMPLE_N,
        "calibration_sample_seed": CALIBRATION_SAMPLE_SEED,
        "review_budgets": list(REVIEW_BUDGETS),
        "primary_review_budget": PRIMARY_REVIEW_BUDGET,
        "decision_rule": "REVIEW if consistency_risk_score >= threshold else CLEAR",
        "llm_does_not_choose_threshold": True,
        "month7_opened": False,
    }
