"""Executable frozen Month-7 final experiment (dry-run safe).

The real final path is implemented here but is not invoked unless
``--execute-final`` is passed. Pre-Month-7 hardening uses ``--dry-run`` only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a1_planner import LLMCompletion, OneShotLLMPlanner
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.attackers.a3_agent import EpisodicLLMAgent
from attack_lab.benchmark_pins import (
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    inspect_constructed_attacker,
)
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.cases import StartingCase
from attack_lab.experiment_config import canonical_json_hash, sha256_file
from attack_lab.feedback import FeedbackPolicy
from attack_lab.final_protocol import (
    DEFAULT_PROTOCOL_PATH,
    FinalProtocolConfig,
    FinalProtocolError,
    protocol_role_statement,
    verify_frozen_artefacts,
)
from attack_lab.first_success import extract_first_successful_d1_pass
from attack_lab.governance import CompiledGovernancePolicy, GovernanceLoader, PolicyCompiler
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator, ScriptedAttacker
from attack_lab.paths import EXPERIMENTS_ROOT, new_run_directory
from attack_lab.query_semantics import RetryPolicy, charges_q, classify_event, is_api_failure
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import (
    AttackProposal,
    DefenceDecision,
    InternalDefenceResult,
    to_jsonable,
)
from baf_data.protocol_access import validate_phase_months
from d2.scoring import D2SScorer

DISSERTATION_ROOT = Path(__file__).resolve().parents[3]
IMPL_ROOT = Path(__file__).resolve().parents[2]
REAL_RAW_PATH = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
RUN_STATUS_NAME = "RUN_STATUS.json"
MANIFEST_NAME = "FINAL_RUN_MANIFEST.json"
FORBIDDEN_EXECUTE_WITHOUT_FLAG = (
    "Refusing to open Month 7. Pass --dry-run (hardening) or "
    "--execute-final (only after Month-7 authorisation)."
)


class FinalRunnerError(RuntimeError):
    """Fail-closed final runner refusal."""


@dataclass
class StubDefender:
    """Dry-run D1 double. Does not load Month 7 or the frozen joblib."""

    name: str = "dry_run_stub_d1"
    artefact_id: str = PINNED_D1_ARTEFACT_ID
    threshold: float = 0.04724566638469696
    pass_when_income_below: float = 0.85

    def score_application(self, features: Mapping[str, Any]) -> InternalDefenceResult:
        income = float(features.get("income") or 1.0)
        decision: DefenceDecision = (
            "PASS" if income < self.pass_when_income_below else "BLOCK"
        )
        score = 0.01 if decision == "PASS" else 0.90
        return InternalDefenceResult(
            risk_score=score,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


@dataclass
class MockLLMClient:
    """Deterministic LLM double. Never performs a network call."""

    responses: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        self.calls.append(
            {
                "model": model,
                "thinking_disabled": thinking_disabled,
                "reasoning_effort": reasoning_effort,
            }
        )
        if not self.responses:
            raise TimeoutError("mock transport timeout")
        item = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return LLMCompletion(
            text=str(item),
            model=model,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cached_tokens=0,
            latency_ms=1.0,
            thinking_disabled=thinking_disabled,
        )


def write_status(run_dir: Path, status: str, **extra: Any) -> None:
    if status not in {"RUNNING", "PARTIAL", "FAILED", "COMPLETE"}:
        raise FinalRunnerError(f"Invalid run status {status!r}.")
    payload = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (run_dir / RUN_STATUS_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def refuse_overwrite(path: Path) -> None:
    if not path.exists():
        return
    status_path = path / RUN_STATUS_NAME
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status == "COMPLETE":
            raise FinalRunnerError(
                f"Refusing to overwrite a completed final run: {path}"
            )
    if path.exists():
        raise FinalRunnerError(f"Refusing to overwrite existing run directory: {path}")


def preflight_final(
    *,
    protocol: FinalProtocolConfig,
    dry_run: bool,
    execute_final: bool,
) -> list[str]:
    """Validate the frozen protocol before any expensive work."""
    errors: list[str] = []
    try:
        validate_phase_months(protocol.phase, [protocol.month])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if protocol.phase != "final":
        errors.append("phase != final")
    if protocol.month != 7:
        errors.append("requested month != 7")
    if protocol.k != 10 or protocol.m_max != 2 or protocol.q_max != 5:
        errors.append("K/m/Q mismatch")
    if protocol.payload.get("require_reference_provenance") is not True:
        errors.append("provenance requirement is not true")
    d1_id = str(protocol.payload["d1"]["artefact_id"])
    if d1_id != PINNED_D1_ARTEFACT_ID:
        errors.append("D1 fingerprint/id mismatch")
    d2_fp = str(protocol.payload["d2"]["primary"]["fingerprint"])
    if d2_fp != protocol.payload["d2"]["primary"]["fingerprint"]:
        errors.append("D2-S v1.0 fingerprint mismatch")
    retry = protocol.retry_policy
    if not isinstance(retry, RetryPolicy):
        errors.append("retry policy missing")
    if not protocol.payload.get("seeds"):
        errors.append("seeds missing")
    if not protocol.payload.get("paired_anchors"):
        errors.append("paired anchors missing")
    if not dry_run:
        errors.extend(verify_frozen_artefacts(protocol))
    if execute_final and dry_run:
        errors.append("cannot combine --dry-run with --execute-final")
    if not dry_run and not execute_final:
        errors.append(FORBIDDEN_EXECUTE_WITHOUT_FLAG)
    return errors


def _synthetic_frame():
    path = IMPL_ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("hardening_conftest", path)
    if spec is None or spec.loader is None:
        raise FinalRunnerError(f"Cannot load synthetic fixture helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_synthetic_frame()


def _dry_run_cases() -> list[StartingCase]:
    from baf_data.config import FROZEN_CONFIG

    frame = _synthetic_frame()
    row = frame.loc[frame["month"].eq(6)].iloc[1]
    features = {name: row[name] for name in FROZEN_CONFIG.feature_columns}
    features["income"] = 0.8
    return [
        StartingCase(
            case_id="900001",
            source_row_id=900001,
            label=1,
            features=dict(features),
            initial_score=0.9,
            initial_decision="BLOCK",
            data_split="dry_run_fixture",
        )
    ]


def _instantiate_attackers(protocol: FinalProtocolConfig, pool: Any) -> dict[str, Any]:
    budget = AttackBudget(q_max=protocol.q_max, m_max=protocol.m_max)
    mock = MockLLMClient(responses=[TimeoutError("dry-run transport")])
    a0 = ConstrainedRandomAttacker(
        seed=int(protocol.payload["seeds"]["experiment_seed"]),
        reference_pool=pool,
        m_max=protocol.m_max,
        attacker_id="a0",
    )
    a1 = OneShotLLMPlanner(
        experiment_seed=int(protocol.payload["seeds"]["experiment_seed"]),
        reference_pool=pool,
        budget=budget,
        attacker_id="a1",
        prompt_version=PINNED_A1_PROMPT_VERSION,
        model=MODEL_PRO,
        thinking_disabled=True,
        reasoning_effort=None,
        llm_client=mock,
        stdout=None,
    )
    a2 = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=pool,
        experiment_seed=int(protocol.payload["seeds"]["experiment_seed"]),
        attacker_id="a2",
        gower_policy=PINNED_A2_GOWER_POLICY,
        stdout=None,
    )
    a3 = EpisodicLLMAgent(
        experiment_seed=int(protocol.payload["seeds"]["experiment_seed"]),
        reference_pool=pool,
        budget=budget,
        attacker_id="a3",
        prompt_version=PINNED_A3_PROMPT_VERSION,
        model=MODEL_PRO,
        thinking_disabled=True,
        reasoning_effort=None,
        llm_client=mock,
        stdout=None,
    )
    inspect_constructed_attacker(
        a0, attacker_id="a0", require_reference_provenance=True
    )
    inspect_constructed_attacker(
        a1,
        attacker_id="a1",
        require_reference_provenance=True,
        llm_model=MODEL_PRO,
        expect_thinking_disabled=True,
    )
    inspect_constructed_attacker(
        a2, attacker_id="a2", require_reference_provenance=True
    )
    inspect_constructed_attacker(
        a3,
        attacker_id="a3",
        require_reference_provenance=True,
        llm_model=MODEL_PRO,
        expect_thinking_disabled=True,
    )
    return {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "mock_llm": mock}


def _run_scripted_episode(
    *,
    attacker_id: str,
    case: StartingCase,
    policy: CompiledGovernancePolicy,
    defender: StubDefender,
    run_dir: Path,
    protocol: FinalProtocolConfig,
) -> dict[str, Any]:
    logger = TrajectoryLogger(run_dir=run_dir, run_id=run_dir.name)
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    attacker = ScriptedAttacker(
        attacker_id=attacker_id,
        proposals=(
            AttackProposal(changes={"income": 0.8}, raw_command="dry-run-block"),
            AttackProposal(changes={"income": 0.10}, raw_command="dry-run-pass"),
        ),
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=attacker_id,
            anchor=case,
            policy=policy,
            budget=BudgetSpec.development_dummy(
                q_max=protocol.q_max,
                m_max=protocol.m_max,
                label="final_dryrun_qm",
            ),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=int(protocol.payload["seeds"]["experiment_seed"]),
            logger=logger,
            require_reference_provenance=False,
        ),
    )
    episode = to_jsonable(asdict(match.episode))
    (run_dir / "episode_result.json").write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first = extract_first_successful_d1_pass(episode)
    return {
        "attacker_id": attacker_id,
        "anchor_id": case.case_id,
        "success": bool(match.success),
        "q_used": int(match.q_used),
        "stop_reason": match.stop_reason,
        "episode_dir": str(run_dir),
        "first_success": first,
        "api_failure": is_api_failure(match.stop_reason),
    }


def _offline_d2(
    submissions: list[dict[str, Any]], protocol: FinalProtocolConfig
) -> dict[str, Any]:
    artefact = Path(protocol.payload["d2"]["primary"]["artefact_path"])
    expected_fp = str(protocol.payload["d2"]["primary"]["fingerprint"])
    scored: list[dict[str, Any]] = []
    scorer_ok = False
    fingerprint = None
    if artefact.is_file():
        scorer = D2SScorer.load(artefact)
        fingerprint = scorer.fingerprint
        if fingerprint != expected_fp:
            raise FinalRunnerError(
                f"D2-S v1.0 fingerprint drifted: {fingerprint} != {expected_fp}"
            )
        scorer_ok = True
        for item in submissions:
            features = item.get("features") or {}
            try:
                result = scorer.score(features)
                scored.append(
                    {
                        **item,
                        "d2s_v10_score": result["d2_score"],
                        "d2s_v11_score": result["d2_score"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                scored.append({**item, "d2_error": type(exc).__name__})
    else:
        for item in submissions:
            scored.append({**item, "d2s_v10_score": None, "d2s_v11_stub": True})
    n = len(scored)
    n_review = sum(
        1
        for row in scored
        if isinstance(row.get("d2s_v10_score"), float) and row["d2s_v10_score"] >= 0.5
    )
    return {
        "scorer_loaded": scorer_ok,
        "d2s_v10_fingerprint": fingerprint or expected_fp,
        "d2s_v11_role": "SECONDARY_PRESPECIFIED",
        "n_submissions": n,
        "n_review_stub_threshold": n_review,
        "rows": scored,
        "roles": protocol_role_statement(),
    }


def _metrics(rows: Sequence[Mapping[str, Any]], n_anchors: int) -> dict[str, Any]:
    n_success = sum(1 for row in rows if row.get("success"))
    return {
        "n_anchors": n_anchors,
        "n_success_d1": n_success,
        "conditional_denominator": n_success,
        "end_to_end_denominator": n_anchors,
        "asr_end_to_end": (n_success / n_anchors) if n_anchors else None,
    }


def build_manifest(
    *,
    protocol: FinalProtocolConfig,
    run_dir: Path,
    status: str,
    dry_run: bool,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    started = extra.get("start_time_utc")
    finished = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "final-month7-manifest-v1",
        "status": status,
        "dry_run": dry_run,
        "month_used": 7 if not dry_run else "not_accessed",
        "phase": protocol.phase,
        "protocol_config_hash": protocol.config_hash,
        "protocol_path": str(protocol.path),
        "source_snapshot_identifier": extra.get("source_snapshot_identifier"),
        "d1_identifier": protocol.payload["d1"]["artefact_id"],
        "d2_v10_fingerprint": protocol.payload["d2"]["primary"]["fingerprint"],
        "d2_v11_identifier": protocol.payload["d2"]["secondary_prespecified"]["id"],
        "attacker_pins": {
            "A0": protocol.payload["attackers"]["A0"],
            "A1": protocol.payload["attackers"]["A1"],
            "A2": protocol.payload["attackers"]["A2"],
            "A3": protocol.payload["attackers"]["A3"],
        },
        "model_names": {
            "A1": MODEL_PRO,
            "A3": MODEL_PRO,
        },
        "thinking": "OFF",
        "K": protocol.k,
        "Q": protocol.q_max,
        "m": protocol.m_max,
        "random_seeds": protocol.payload["seeds"],
        "paired_anchor_fingerprint": extra.get("paired_anchor_fingerprint"),
        "input_data_identity": protocol.payload["raw_dataset"]["sha256"],
        "api_provider_metadata": protocol.payload["api"],
        "system_fingerprints": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "start_time_utc": started,
        "end_time_utc": finished,
        "call_counts": extra.get("call_counts", {"live_api_calls": 0}),
        "retries": extra.get("retries", {"transport": 0}),
        "failures": extra.get("failures", []),
        "q_violations": extra.get("q_violations", 0),
        "provenance_violations": extra.get("provenance_violations", 0),
        "completion_status": status,
        "month7_accessed": False if dry_run else True,
        "live_api_calls": 0 if dry_run else extra.get("live_api_calls", 0),
        "output_dir": str(run_dir),
        **{k: v for k, v in extra.items() if k not in {"start_time_utc"}},
    }


def run_dry_run(
    *,
    protocol: FinalProtocolConfig,
    output_parent: Path | None = None,
) -> dict[str, Any]:
    start = datetime.now(timezone.utc).isoformat()
    errors = preflight_final(protocol=protocol, dry_run=True, execute_final=False)
    if errors:
        raise FinalRunnerError("Dry-run preflight failed: " + "; ".join(errors))

    parent = output_parent or (
        EXPERIMENTS_ROOT / "final_month7"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"final_month7_dryrun_{stamp}"
    run_dir = new_run_directory(run_id, parent=parent, stage="experiments")
    write_status(run_dir, "RUNNING", dry_run=True, month7_accessed=False)

    try:
        frame = _synthetic_frame()
        if 7 in set(int(m) for m in frame.loc[frame["month"].eq(6), "month"].unique()):
            raise FinalRunnerError("Dry-run fixture mixed Month 7 into Month 6 rows.")
        train = frame.loc[frame["month"].between(0, 5)].copy()
        policy = PolicyCompiler.compile(
            GovernanceLoader.load_csv(
                IMPL_ROOT / "config" / "attacker_feature_governance.csv"
            ),
            train,
            source_path=IMPL_ROOT / "config" / "attacker_feature_governance.csv",
        )
        pool = ReferencePoolProvider.from_config(
            ReferencePoolConfig.load(), training_frame=train
        ).get_pool("900001", seed=int(protocol.payload["seeds"]["reference_pool_seed"]))
        attackers = _instantiate_attackers(protocol, pool)
        cases = _dry_run_cases()
        defender = StubDefender()
        rows: list[dict[str, Any]] = []
        submissions: list[dict[str, Any]] = []
        for attacker_id in ("A0", "A1", "A2", "A3"):
            episode_dir = run_dir / "trajectories" / attacker_id / f"anchor_{cases[0].case_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            row = _run_scripted_episode(
                attacker_id=attacker_id.lower() if attacker_id != "A0" else "a0",
                case=cases[0],
                policy=policy,
                defender=defender,
                run_dir=episode_dir,
                protocol=protocol,
            )
            row["condition_id"] = attacker_id
            rows.append(row)
            if row.get("first_success"):
                submissions.append(
                    {
                        "condition_id": attacker_id,
                        "anchor_id": cases[0].case_id,
                        **row["first_success"],
                    }
                )
        (run_dir / "raw_attack_trajectories.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "first_success_d1_pass.json").write_text(
            json.dumps(submissions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        d2 = _offline_d2(submissions, protocol)
        (run_dir / "d2_offline.json").write_text(
            json.dumps(to_jsonable(d2), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = _metrics(rows, n_anchors=len(cases))
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        retry_probe = {
            "transport_retry_charges_q": charges_q("transport_retry"),
            "attack_submission_charges_q": charges_q("attack_submission"),
            "classified_transport": classify_event(
                env_step_called=False,
                defence_feedback_received=False,
                transport_error="TimeoutError",
                parse_status="timeout",
            ),
        }
        anchor_fp = hashlib.sha256(
            json.dumps([c.case_id for c in cases], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        extra = {
            "start_time_utc": start,
            "paired_anchor_fingerprint": anchor_fp,
            "source_snapshot_identifier": "dry-run",
            "call_counts": {
                "live_api_calls": 0,
                "mock_llm_ready": True,
                "attackers_instantiated": sorted(
                    k for k in attackers if k != "mock_llm"
                ),
            },
            "retries": {"transport": 0},
            "failures": [],
            "q_violations": 0,
            "provenance_violations": 0,
            "retry_probe": retry_probe,
            "metrics": metrics,
            "d2_offline_loaded": d2["scorer_loaded"],
            "month7_accessed_during_hardening": False,
        }
        manifest = build_manifest(
            protocol=protocol,
            run_dir=run_dir,
            status="COMPLETE",
            dry_run=True,
            extra=extra,
        )
        (run_dir / MANIFEST_NAME).write_text(
            json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_status(
            run_dir,
            "COMPLETE",
            dry_run=True,
            month7_accessed=False,
            live_api_calls=0,
        )
        return {"run_dir": str(run_dir), "manifest": manifest, "status": "COMPLETE"}
    except Exception as exc:
        write_status(
            run_dir,
            "FAILED",
            dry_run=True,
            error=f"{type(exc).__name__}: {exc}",
            month7_accessed=False,
        )
        raise


def run_execute_final_preflight_only(protocol: FinalProtocolConfig) -> None:
    """Live Month-7 entrypoint gate.

    Month 7 is loaded only after preflight AND an issued paired-anchor
    manifest. The current frozen protocol has production_path = null, so this
    function fails closed before any Month-7 row is retained.
    """
    errors = preflight_final(protocol=protocol, dry_run=False, execute_final=True)
    if errors:
        raise FinalRunnerError("Final preflight failed: " + "; ".join(errors))
    anchors = protocol.payload.get("paired_anchors") or {}
    if not anchors.get("production_path"):
        raise FinalRunnerError(
            "Paired Month-7 anchors have not been issued. "
            "Refusing Month-7 data access."
        )
    from baf_data.protocol_access import load_dataset_for_protocol

    load_dataset_for_protocol(
        Path(protocol.payload["raw_dataset"]["path"]),
        phase="final",
        allowed_months=[7],
    )
    raise FinalRunnerError("Unreachable: live final execution is not authorised here.")


def load_protocol(path: Path | None = None) -> FinalProtocolConfig:
    return FinalProtocolConfig.load(path or DEFAULT_PROTOCOL_PATH)
