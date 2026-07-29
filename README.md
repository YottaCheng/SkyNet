# 04_implementation — Dissertation Code

Code for the dissertation *Adversarial Simulation of Synthetic Identity Fraud:
Benchmarking Hybrid KYC Defenses against Autonomous LLM Agents* (SECU0045, Ziyao Cheng).

## Current contents

| Path | Purpose |
| --- | --- |
| `src/audit_baf.py` | Read-only audit of the BAF Base dataset. Produces summary tables, a JSON summary, a fraud-rate-by-month plot and `DATASET_AUDIT.md`. Performs **no** cleaning, imputation, splitting or modelling. |
| `src/baf_data/` | Deterministic data layer: raw-source integrity checks, frozen schema/sentinel/split configuration, in-memory sentinel normalisation, temporal train/dev/test split, X/y views and manifest writing. Fits **no** imputer, encoder, scaler or model. |
| `src/prepare_baf_data.py` | CLI launcher for the data layer. |
| `src/baf_models/` | Model architectures. Currently the **unfitted** Logistic Regression baseline scaffold: preprocessing definition (median imputer + missing indicators + scaling for numerics, one-hot for categoricals) and a configuration-driven pipeline builder. Feature lists are derived from `baf_data.config`; nothing is fitted and no data is read. |
| `config/logistic_baseline.yaml` | Initial LR settings (a-priori values, not experimentally selected). Never stores results. |
| `config/feature_handling.csv` | Human-readable feature register (role, feature set, sentinel rule, status). |
| `tests/` | Unit tests for every data-layer module, synthetic and real-data integration tests (the latter skip automatically when the external drive is not mounted), and structural tests for the unfitted LR scaffold. |
| `config/feature_handling.csv` | Human-readable feature-governance register. Runtime enforcement remains in `src/baf_data/config.py`. |
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

Tests:

```bash
python -m pytest tests/ -v
```
