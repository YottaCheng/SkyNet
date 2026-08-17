# Active-stack cleanup manifest (pre-move)

Date: 2026-08-13  
Scope: archive deprecated attacker leaves / thinking ablations / obsolete runners.  
Active attackers retained: A0, A1 V4.3, A2, A3 V2.3.  
Not modified: prompt/schema text of retained contracts, D1, K/Q/m, dataset, Month-7 locks.

## Active files (remain in place)

| Path | Role |
|---|---|
| `src/attack_lab/attackers/a0_random.py` | A0 |
| `src/attack_lab/attackers/a1_planner.py` | A1 runtime (V4.3 dispatch kept) |
| `src/attack_lab/attackers/a1_v3_contract.py` | inheritance base |
| `src/attack_lab/attackers/a1_v4_contract.py` | inheritance base |
| `src/attack_lab/attackers/a1_v4_1_contract.py` | inheritance base |
| `src/attack_lab/attackers/a1_v4_2_contract.py` | inheritance base |
| `src/attack_lab/attackers/a1_v4_3_contract.py` | **active A1 pin** |
| `src/attack_lab/attackers/a2_search.py` | A2 |
| `src/attack_lab/attackers/a3_agent.py` | A3 runtime (V2.3 dispatch kept) |
| `src/attack_lab/attackers/a3_v2_contract.py` | inheritance base |
| `src/attack_lab/attackers/a3_v2_1_contract.py` | inheritance base |
| `src/attack_lab/attackers/a3_v2_2_contract.py` | inheritance base |
| `src/attack_lab/attackers/a3_v2_3_contract.py` | **active A3 pin** |
| `src/attack_lab/benchmark_pins.py` | retarget pins → V4.3 / V2.3 |
| `scripts/run_dev_model_selection.py` | active harness (pin strings updated) |
| `scripts/run_formal_a0_a3_comparison.py` | formal harness (uses pins) |
| `scripts/test_deepseek_connection.py` | connectivity smoke |

## Archived files (this move)

| Active path (before) | Archived path (after) |
|---|---|
| `src/attack_lab/attackers/a1_v4_4_contract.py` | `archive/2026-08-active-stack-cleanup/contracts/a1_v4_4_contract.py` |
| `src/attack_lab/attackers/a3_v2_4_contract.py` | `archive/2026-08-active-stack-cleanup/contracts/a3_v2_4_contract.py` |
| `scripts/run_dev_thinking_mode_n25.py` | `archive/.../scripts/run_dev_thinking_mode_n25.py` |
| `scripts/run_dev_a3_v2_3_thinkon_sanity.py` | `archive/.../scripts/run_dev_a3_v2_3_thinkon_sanity.py` |
| `scripts/run_dev_a3_v2_3_reasoning_effort_sanity.py` | `archive/.../scripts/run_dev_a3_v2_3_reasoning_effort_sanity.py` |
| `scripts/run_a3_prompt_ablation.py` | `archive/.../scripts/run_a3_prompt_ablation.py` |
| `tests/attack_lab/test_a1_v4_4_a3_v2_4_prompt_candidates.py` | `archive/.../tests/test_a1_v4_4_a3_v2_4_prompt_candidates.py` |
| `tests/attack_lab/test_dev_thinking_mode_pins.py` | `archive/.../tests/test_dev_thinking_mode_pins.py` |
| `tests/attack_lab/test_a3_stage_b_live_runner.py` | `archive/.../tests/test_a3_stage_b_live_runner.py` |

## Imports / wiring affected (post-move edits; no prompt text edits)

| Module | Change |
|---|---|
| `benchmark_pins.py` | Import `PROMPT_VERSION_V4_3` / `PROMPT_VERSION_A3_V2_3`; set `PINNED_A1_*` / `PINNED_A3_*` |
| `a1_planner.py` | Remove V4.4 import, `SUPPORTED_PROMPT_VERSIONS` entry, render/dispatch branches, `__all__` |
| `a3_agent.py` | Remove V2.4 import, label, planning-contract branch, v2 family membership, payload/slots branches, `__all__` |
| `scripts/run_dev_model_selection.py` | Drift checks → V4.3 / V2.3 strings |
| `tests/attack_lab/test_benchmark_pins.py` | Expected pins → V4.3 / V2.3; legacy fail uses archived V4.4 id |
| `tests/attack_lab/test_thinking_max_tokens_policy.py` | Pin assert → V2.3 (max_tokens helper stays active) |

## Explicitly not archived

- Intermediate contracts V3–V4.2 and A3 V2–V2.2 (required bases).
- Historical `scripts/archive/2026-08-development/` (already archived).
- Month-7 paths / D1 artefact / governance fingerprint constants.

## Post-move verification

- `pytest tests/attack_lab`: 311 passed
- `pytest tests`: 430 passed
- Live package no longer imports `a1_v4_4_contract` / `a3_v2_4_contract`
- Benchmark pins: A1 V4.3 / A3 V2.3
- D1 artefact id, governance fingerprint, K/Q/m, datasets, Month-7 locks unchanged

## Follow-up: attackers/ leaf-only (2026-08-13 evening)

`src/attack_lab/attackers/` now contains only:
- `__init__.py`
- `a0_random.py`
- `a1_planner.py` + `a1_v4_3_contract.py`
- `a2_search.py`
- `a3_agent.py` + `a3_v2_3_contract.py`

Intermediate inheritance contracts moved to importable archive package:
`src/attack_lab/archive/contracts/`
(a1_v3, a1_v4, a1_v4_1, a1_v4_2, a3_v2, a3_v2_1, a3_v2_2)

Active V4.3 / V2.3 import those bases from `attack_lab.archive.contracts.*`.
