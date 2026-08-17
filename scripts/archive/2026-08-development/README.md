# Archived 2026-08 development runners

Completed Month-6 development, calibration, smoke, validation, Stage-B
live-grid, and early comparison runners. Retained for reproducibility and
construct-validity diagnostics only.

Do not treat archived ASR/results as final comparable evidence for a
reference-backed identity-recombination claim without provenance controls.

## Active scripts (not in this archive)

Retained under `scripts/` (see also `scripts/README.md` and
`archive/2026-08-active-stack-cleanup/`):

- `run_dev_model_selection.py` — Flash/Pro development selection harness
- `run_formal_a0_a3_comparison.py` — formal A0–A3 comparison harness
- `test_deepseek_connection.py` — API connectivity smoke

Deprecated thinking / V4.4 / V2.4 / prompt-ablation artefacts were moved to
`archive/2026-08-active-stack-cleanup/` on 2026-08-13.

## Future formal runner base (keep complete; do not promote yet)

These two runners already encode the reusable comparison harness. After
provenance fixes, update these files rather than inventing a new runner /
orchestrator / logger:

| Script | Role |
|---|---|
| `run_formal_a0_a3_comparison.py` | **Future formal four-attacker (A0–A3) comparison base.** Config-locked preflight + `--launch`, shared `MatchOrchestrator` / `TrajectoryLogger`, multi seed×attacker×anchor, ASR@q summaries, implementation/governance/D1/anchor drift checks. |
| `run_a0_a2_paired_dev_comparison.py` | **Paired-reporting reference.** Same-anchor×seed pairing, paired outcome taxonomy, cross-seed ASR curves / ΔASR, diagnostic CSVs, markdown/txt comparison report. Logic to reuse when extending formal A0–A3 reporting. |

Do **not** move the formal runner back under active `scripts/` until a
provenance-complete formal campaign is authorised.

## Historical development runners (retain; do not delete)

Stage-B / calibration / smoke / pilot / validation:

- `run_a3_stage_b_live_grid.py`
- `run_a3_grounded_grid_preflight.py`
- `run_a3_smoke_test.py`
- `run_a3_p1_local_repair_validation.py`
- `run_a3_ranked_portfolio_validation.py`
- `run_a0_smoke_test.py`
- `run_a0_e_budget_calibration.py`
- `run_a0_formal_e_budget_calibration.py`
- `run_a0_m_curve_calibration.py`
- `run_a0_gov_v2_multiseed_calibration.py`
- `run_a1_smoke_test.py`
- `run_a1_stability_check.py`
- `run_a1_dev_evaluation.py`
- `run_a2_mech_pilot.py`
- `run_identity_composition_pilot.py`
- `audit_a0_a1_a2_k10_integration_smoke.py` — post-run audit only (reads completed smoke artefacts; no attacker rerun)

## Shared infrastructure (do not re-create)

Episode execution and logging already exist in the package:

- `attack_lab.orchestrator.MatchOrchestrator` — pluggable episode harness
- `attack_lab.logger.TrajectoryLogger` — episode trajectory / public transcript

**No new runner, orchestrator, or logger is required** for the eventual formal
A0–A3 comparison after provenance work.
