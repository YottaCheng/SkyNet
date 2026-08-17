# Attack Lab scripts

## Active (keep under `scripts/`)

- `run_dev_model_selection.py` — Flash/Pro development selection harness (pinned A1 V4.3 / A3 V2.3)
- `run_formal_a0_a3_comparison.py` — formal A0–A3 comparison harness (uses benchmark pins)
- `test_deepseek_connection.py` — connectivity smoke for the DeepSeek client

Active attackers: A0, A1 V4.3, A2, A3 V2.3.

Do not treat outputs from active development runners as sealed final evidence.

## Archive

Deprecated attacker leaves, thinking ablations, and obsolete prompt experiments:

`archive/2026-08-active-stack-cleanup/`

See that directory’s `CLEANUP_MANIFEST.md`.

Completed Month-6 development, calibration, smoke, validation, Stage-B, and
early comparison runners live under:

`scripts/archive/2026-08-development/`

See that directory’s `README.md` for role labels. Archived runners are retained
for reproducibility; they are not deleted and are not promoted back to active
unless explicitly authorised.
