# Archived attack-lab modules

Historical A1/A3 prompt-contract modules that are **not** the active stack
(A0 / A1 V4.3 / A2 / A3 V2.3).

Active attacker implementations live in `attack_lab.attackers`.
These contracts remain importable as `attack_lab.archive.contracts.*` because
V4.3 / V2.3 inherit from them and regression tests still exercise the bases.

Deprecated leaf contracts V4.4 / V2.4 remain under
`04_implementation/archive/2026-08-active-stack-cleanup/contracts/` (not on the
live import path).
