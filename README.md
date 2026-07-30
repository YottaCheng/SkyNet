# 04_implementation — Dissertation Code

Code for the dissertation *Adversarial Simulation of Synthetic Identity Fraud:
Benchmarking Hybrid KYC Defenses against Autonomous LLM Agents* (SECU0045, Ziyao Cheng).

## Current contents

| Path | Purpose |
| --- | --- |
| `src/audit_baf.py` | Read-only audit of the BAF Base dataset. Produces summary tables, a JSON summary, a fraud-rate-by-month plot and `DATASET_AUDIT.md`. Performs **no** cleaning, imputation, splitting or modelling. |
| `src/baf_data/` | Deterministic data layer: raw-source integrity checks, frozen schema/sentinel/split configuration, in-memory sentinel normalisation, temporal train/dev/test split, X/y views and manifest writing. Fits **no** imputer, encoder, scaler or model. |
| `src/prepare_baf_data.py` | CLI launcher for the data layer. |
| `src/baf_models/` | Model package: four-branch preprocessing, unfitted LR scaffold, training orchestration (`training.py`), development evaluation/threshold selection (`evaluation.py`), and artifact/plot writers (`artifacts.py`). Feature groups derive from `baf_data.config`. |
| `src/run_logistic_baseline.py` | CLI: fit unweighted and balanced LR on months 0–5; evaluate month 6 only; write artefacts under `05_outputs/logistic_baseline/`. Does not score month 7. |
| `config/logistic_baseline.yaml` | Shared a-priori LR settings (`C`, `solver`, `max_iter`, `random_state`). Variant `class_weight` values are set by the training module. |
| `config/feature_handling.csv` | Human-readable feature-governance register. Runtime enforcement remains in `src/baf_data/config.py`. |
| `tests/` | Data-layer tests, LR scaffold tests and LR training/evaluation tests. |
| `logs/` | Dated development evidence and append-only decision history; not current-state instructions. |

## Data location

The BAF dataset lives on an external drive and is **not** stored in this repository:

```
<external drive>/ucl_dissertation_data/raw/baf/Base.csv
```

The raw file is immutable: permissions are read-only (0444) and every
pipeline run verifies its SHA-256 before reading and again after
completion, refusing to run on any mismatch. Nothing is ever written
inside `raw/`.

Audit outputs: `<external drive>/ucl_dissertation_data/documentation/baf_audit/`
Data-layer manifests and run log: `<external drive>/ucl_dissertation_data/splits/baf_base/`

## Frozen data-layer decisions

Executable frozen data-layer decisions live in `src/baf_data/config.py`
(`FROZEN_CONFIG`). The matching human-readable register is
`config/feature_handling.csv`; the full methodological rationale is in the
project-root `EXPERIMENT_SPEC.md`. Summary:

- `fraud_bool` is the target only; `month` is split-only. Neither enters the feature matrix.
- Temporal split: months 0–5 train, month 6 development, month 7 final test.
- Primary exclusions: `device_fraud_count` (constant), `days_since_request` (excluded from all current training), `credit_risk_score` (reserved for a possible later sensitivity experiment).
- Verified sentinel rules only: `-1 -> NaN` for `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`; `< 0 -> NaN` for `intended_balcon_amount`. Valid negative `velocity_6h` and all `credit_risk_score` values are preserved. No rows are deleted.

## Setup

```bash
conda create -n ucl python=3.11 -y
conda activate ucl
pip install -r requirements.txt
```

## Usage

Read-only audit:

```bash
python src/audit_baf.py \
  --input  /Volumes/<drive>/ucl_dissertation_data/raw/baf/Base.csv \
  --output /Volumes/<drive>/ucl_dissertation_data/documentation/baf_audit
```

Data layer (writes manifests + run log only; no cleaned CSV is persisted):

```bash
python src/prepare_baf_data.py \
  --raw    /Volumes/<drive>/ucl_dissertation_data/raw/baf/Base.csv \
  --output /Volumes/<drive>/ucl_dissertation_data/splits/baf_base
```

From model code, import the in-memory interface instead of re-reading CSVs:

```python
from baf_data import load_prepared_splits

prepared = load_prepared_splits(raw_path)
X_train, y_train = prepared.views["train"].X, prepared.views["train"].y
```

Logistic Regression development baseline (months 0–5 fit, month 6 eval only):

```bash
python src/run_logistic_baseline.py \
  --raw /Volumes/<drive>/ucl_dissertation_data/raw/baf/Base.csv \
  --config config/logistic_baseline.yaml
```

Artefacts are written outside this Git repository to
`../05_outputs/logistic_baseline/<run_id>/`.

Tests:

```bash
python -m pytest tests/ -v
```
