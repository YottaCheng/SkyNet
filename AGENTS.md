# Implementation Agent Instructions

## Scope

These instructions apply to all files under `04_implementation/`.

The repository may be read broadly when necessary, but writes are limited to
the files and task scope explicitly authorised by the user.

## Authoritative sources

For implementation work, use the following sources in this order:

1. The user's current explicit instruction
2. This `AGENTS.md`
3. `../EXPERIMENT_SPEC.md` for frozen experimental design
4. `../IMPLEMENTATION_STATUS.md` for verified implementation status
5. `src/baf_data/config.py` for executable feature and data-layer configuration
6. `config/feature_handling.csv` as the human-readable feature register
7. Dated files under `logs/` as historical evidence only

Do not treat plans, scaffolds or previous chat claims as implemented evidence.

## Allowed work

When explicitly authorised for the current task, implementation agents may:

- modify code under `src/`;
- modify or add tests under `tests/`;
- modify task-specific configuration under `config/`;
- create new, uniquely named experimental output directories;
- create dated local implementation notes under `logs/`;
- run tests, inspections and authorised experiments.

## Protected files and areas

Unless the user explicitly authorises the exact change, do not modify:

- files outside `04_implementation/`;
- dissertation DOCX or PDF files;
- `../UCL_REQUIREMENTS.md`;
- `../PROJECT_CONTEXT.md`;
- `../EXPERIMENT_SPEC.md`;
- `../IMPLEMENTATION_STATUS.md`;
- raw datasets;
- existing experimental output directories;
- `logs/DECISION_LOG.md`.

Reading these files is allowed when relevant.

## Experimental integrity

- Months 0–5 are development training data.
- Month 6 is development evaluation and threshold-selection data.
- Month 7 is reserved for final model-untouched evaluation.
- Month 7 must not be used for fitting, preprocessing fit, feature selection,
  hyperparameter selection, model selection, threshold selection, calibration,
  debugging or deciding which result to report.
- Do not silently change datasets, features, splits, sentinels, metrics,
  thresholds, seeds, models or experimental conditions.
- New runs must use new output directories.
- Never overwrite or mutate existing run artefacts.

## Code quality

- Prefer tested shared functions over copied model-specific implementations.
- Maintain low coupling, high cohesion and single responsibility.
- Keep dependencies directional: CLI or run scripts may depend on shared
  modules; shared modules must not depend on CLI scripts.
- Do not duplicate feature lists, configuration values or path rules.
- Preserve the existing single sources of truth.
- Use explicit validation, clear exceptions and type annotations.
- Do not use bare `except`, silent fallbacks, hidden parameter changes,
  mutable global state or speculative framework abstractions.
- Do not perform large refactors unless the task explicitly requires them.
- Shared-code changes must preserve existing LR and XGBoost behaviour and be
  protected by regression tests.

## Verification

Before reporting completion:

- run task-specific tests;
- run the full test suite whenever shared code changes;
- report the exact commands and exact results;
- verify that no month-7 metrics, scores or figures were produced;
- verify that existing outputs were not overwritten;
- distinguish code created, code tested and experiments actually executed.

## Git

Do not run `git add`, `git commit`, `git push`, `merge`, `rebase`, `tag` or
history-rewriting commands unless the user explicitly authorises that exact
operation.

Never add `Co-authored-by` or other AI authorship trailers.

## Completion report

For any task that changes code, tests, configuration or experimental outputs,
report:

1. files created and modified;
2. commands executed;
3. test results;
4. experimental evidence;
5. data-isolation confirmation;
6. unresolved risks;
7. the single recommended next action.