"""Runner-only regression tests for the Stage-B thin live-grid wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
ARCHIVE_SCRIPTS = SCRIPTS / "archive" / "2026-08-development"
for _path in (SCRIPTS, ARCHIVE_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_a3_prompt_ablation as ablation  # noqa: E402
import run_a3_stage_b_live_grid as grid  # noqa: E402
from attack_lab.attackers.a1_planner import LLMCompletion  # noqa: E402
from attack_lab.attackers.a3_agent import (  # noqa: E402
    PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    PROMPT_VERSION_P1_COMPACT,
    A3ModelConfig,
)
from attack_lab.outbound_payload import (  # noqa: E402
    OutboundPayloadError,
    audit_outbound_payload,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    training = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=training
    ).get_pool(starting_case.case_id)


def _anchors_file(tmp_path: Path) -> Path:
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps({"anchor_ids": [str(800000 + index) for index in range(25)]}),
        encoding="utf-8",
    )
    return path


def _runner(
    tmp_path: Path,
    *,
    output_name: str = "grid",
    single_cell_runner=grid._mock_run_variant,
    resume: bool = False,
) -> grid.StageBGridRunner:
    return grid.StageBGridRunner(
        output_root=tmp_path / output_name,
        anchors_file=_anchors_file(tmp_path),
        raw_path=tmp_path / "unused-month6.csv",
        artefact_dir=tmp_path / "unused-d1",
        resume=resume,
        mock=True,
        single_cell_runner=single_cell_runner,
        preflight_runner=grid._mock_preflight_outbound_payloads,
        client_factory=lambda _model: (_ for _ in ()).throw(
            AssertionError("mock must not construct DeepSeek client")
        ),
    )


def _patch_preflight_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    starting_case,
    reference_pool,
    governance_policy,
) -> None:
    monkeypatch.setattr(
        ablation.CompiledGovernancePolicy,
        "load",
        classmethod(lambda _cls, _path: governance_policy),
    )
    monkeypatch.setattr(
        ablation.FrozenXGBoostDefender,
        "from_artefact_dir",
        classmethod(lambda _cls, _path: object()),
    )
    monkeypatch.setattr(
        ablation,
        "load_starting_case",
        lambda *_args, **_kwargs: starting_case,
    )
    provider = SimpleNamespace(
        get_pool=lambda _anchor_id, seed: reference_pool,
    )
    monkeypatch.setattr(
        ablation.ReferencePoolProvider,
        "from_config",
        staticmethod(lambda _config, raw_path: provider),
    )


def test_b1_b2_existing_preflight_accepts_frozen_neutral_view(
    monkeypatch,
    tmp_path,
    starting_case,
    reference_pool,
    governance_policy,
):
    _patch_preflight_dependencies(
        monkeypatch,
        starting_case=starting_case,
        reference_pool=reference_pool,
        governance_policy=governance_policy,
    )
    for prompt_version in (
        PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
        PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    ):
        result = ablation.preflight_outbound_payloads(
            anchor_ids=[starting_case.case_id],
            budget=grid.AttackBudget(q_max=5, m_max=2),
            experiment_seed=20260804,
            raw_path=tmp_path / "unused.csv",
            artefact_dir=tmp_path / "unused-d1",
            temperatures=(0.0,),
            prompt_version=prompt_version,
            max_local_generation_attempts_per_query=3,
        )
        assert result["status"] == "PASS"
        assert result["payloads"][0]["preflight"] == "PASS"


@pytest.mark.parametrize(
    "neutral_view",
    [
        {"fraud_bool": 1},
        {"fields": {"not_allowlisted": 1}},
        {"note": "/Users/example/private"},
        {"note": "api_key=do-not-send"},
    ],
)
def test_neutral_view_still_fails_closed(neutral_view):
    with pytest.raises(OutboundPayloadError):
        audit_outbound_payload(
            {"neutral_affordance_view": neutral_view},
            allowed_top_level_keys=("neutral_affordance_view",),
            allowed_feature_fields=("customer_age",),
        )


def test_b0_default_and_explicit_preflight_are_identical(
    monkeypatch,
    tmp_path,
    starting_case,
    reference_pool,
    governance_policy,
):
    _patch_preflight_dependencies(
        monkeypatch,
        starting_case=starting_case,
        reference_pool=reference_pool,
        governance_policy=governance_policy,
    )
    captured: list[dict[str, Any]] = []
    existing_audit = ablation.audit_outbound_payload

    def recording_audit(*args, **kwargs):
        result = existing_audit(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(ablation, "audit_outbound_payload", recording_audit)
    common = {
        "anchor_ids": [starting_case.case_id],
        "budget": grid.AttackBudget(q_max=5, m_max=2),
        "experiment_seed": 20260804,
        "raw_path": tmp_path / "unused.csv",
        "artefact_dir": tmp_path / "unused-d1",
        "temperatures": (0.0,),
    }
    old_default = ablation.preflight_outbound_payloads(**common)
    explicit = ablation.preflight_outbound_payloads(
        **common,
        prompt_version=PROMPT_VERSION_P1_COMPACT,
    )
    assert old_default == explicit
    assert captured[0]["payload_sha256"] == captured[1]["payload_sha256"]
    assert captured[0]["external_feature_fields"] == captured[1][
        "external_feature_fields"
    ]
    assert captured[0]["top_level_keys"] == captured[1]["top_level_keys"]
    assert "neutral_affordance_view" not in captured[0]["top_level_keys"]


def test_run_variant_old_p1_semantics_and_client_injection(
    monkeypatch,
    tmp_path,
    starting_case,
    reference_pool,
    governance_policy,
):
    monkeypatch.setattr(
        ablation.CompiledGovernancePolicy,
        "load",
        classmethod(lambda _cls, _path: governance_policy),
    )
    defender = SimpleNamespace(name="frozen-test", artefact_id="frozen-id")
    monkeypatch.setattr(
        ablation.FrozenXGBoostDefender,
        "from_artefact_dir",
        classmethod(lambda _cls, _path: defender),
    )
    monkeypatch.setattr(
        ablation,
        "load_starting_case",
        lambda *_args, **_kwargs: starting_case,
    )
    provider = SimpleNamespace(get_pool=lambda _anchor_id, seed: reference_pool)
    monkeypatch.setattr(
        ablation.ReferencePoolProvider,
        "from_config",
        staticmethod(lambda _config, raw_path: provider),
    )

    captures: list[dict[str, Any]] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            captures.append(kwargs)
            self.query_records = ()
            self.memory_steps = ()
            self.total_llm_calls = 0
            self.total_retries = 0
            self.total_env_steps = 0
            self.total_local_rejections = 0
            self.total_local_regenerations = 0
            self.total_regeneration_exhaustions = 0
            self.config_hash = A3ModelConfig(
                model=kwargs["model"],
                thinking_disabled=kwargs["thinking_disabled"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_tokens=kwargs["max_tokens"],
                max_parse_retries=kwargs["max_parse_retries"],
                timeout_seconds=kwargs["timeout_seconds"],
                prompt_version=kwargs["prompt_version"],
                max_local_generation_attempts_per_query=kwargs[
                    "max_local_generation_attempts_per_query"
                ],
                portfolio_cap=kwargs["portfolio_cap"],
            ).config_hash()

        def aggregate_counters(self):
            return {
                "llm_calls": 0,
                "parse_retries": 0,
                "local_generation_attempts": 0,
                "local_rejections": 0,
                "local_regenerations": 0,
                "regeneration_exhaustions": 0,
                "env_step_calls": 0,
                "parse_failures": 0,
                "governance_failures": 0,
            }

    class FakeMatch:
        success = False
        stop_reason = "mock_no_step"
        q_used = 0
        attempts_to_success = None
        invalid_submissions = 0
        trajectory = ()

        def to_dict(self):
            return {"success": False, "stop_reason": self.stop_reason}

    class FakeOrchestrator:
        def run_episode(self, _attacker, _config):
            return FakeMatch()

    monkeypatch.setattr(ablation, "EpisodicLLMAgent", FakeAgent)
    monkeypatch.setattr(ablation, "MatchOrchestrator", FakeOrchestrator)
    common = {
        "variant": "P1",
        "anchor_ids": [starting_case.case_id],
        "budget": grid.AttackBudget(q_max=5, m_max=2),
        "experiment_seed": 20260804,
        "raw_path": tmp_path / "unused.csv",
        "artefact_dir": tmp_path / "unused-d1",
        "temperature": 0.0,
        "max_local_generation_attempts_per_query": 3,
    }
    default_summary = ablation.run_variant(
        **common,
        run_dir=tmp_path / "default",
    )
    explicit_summary = ablation.run_variant(
        **common,
        run_dir=tmp_path / "explicit",
        prompt_version=PROMPT_VERSION_P1_COMPACT,
    )
    assert default_summary == explicit_summary
    assert captures[0]["prompt_version"] == captures[1]["prompt_version"]
    assert captures[0]["llm_client"] is None
    assert captures[1]["llm_client"] is None

    sentinel_client = object()
    ablation.run_variant(
        **common,
        run_dir=tmp_path / "b1",
        prompt_version=PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
        llm_client=sentinel_client,
    )
    assert captures[-1]["prompt_version"] == PROMPT_VERSION_B1_NEUTRAL_GROUNDED
    assert captures[-1]["llm_client"] is sentinel_client


def test_default_is_preflight_only_and_live_requires_double_unlock(tmp_path):
    dry = grid.StageBGridRunner(
        output_root=tmp_path / "dry",
        anchors_file=_anchors_file(tmp_path),
        raw_path=tmp_path / "unused.csv",
        artefact_dir=tmp_path / "unused-d1",
        preflight_runner=grid._mock_preflight_outbound_payloads,
        client_factory=lambda _model: (_ for _ in ()).throw(
            AssertionError("dry-run must not create client")
        ),
    )
    result = dry.run()
    assert result["status"] == "preflight_complete_no_api"
    assert result["external_api_calls"] == 0

    locked = grid.StageBGridRunner(
        output_root=tmp_path / "locked",
        anchors_file=_anchors_file(tmp_path),
        launch=True,
        authorization_confirmation="wrong",
        preflight_runner=grid._mock_preflight_outbound_payloads,
    )
    with pytest.raises(grid.GridRunnerError, match="second unlock"):
        locked.run()
    assert not (tmp_path / "locked").exists()


def test_mock_grid_runs_fixed_order_and_225_virtual_episodes(tmp_path):
    calls: list[str] = []

    def recording_runner(**kwargs):
        calls.append(str(kwargs["prompt_version"]))
        return grid._mock_run_variant(**kwargs)

    runner = _runner(tmp_path, single_cell_runner=recording_runner)
    result = runner.run()
    assert result["status"] == "completed"
    assert result["external_api_calls"] == 0
    assert result["mock_virtual_episode_count"] == 225
    assert calls == [spec.prompt_version for spec in grid.CELL_PROTOCOL]
    summary = json.loads(
        (tmp_path / "grid" / "stage_b_grid_summary.json").read_text()
    )
    assert summary["mock_only_not_scientific"] is True
    assert summary["mock_virtual_episode_count"] == 225
    assert summary["same_25_anchors_not_n75"] is True
    assert summary["no_best_run_selection"] is True
    attempt_dirs = list((tmp_path / "grid" / "cells").glob("*/attempt_01"))
    assert len(attempt_dirs) == 9
    for spec in grid.CELL_PROTOCOL:
        cell_summary = json.loads(
            (
                tmp_path
                / "grid"
                / "cells"
                / spec.cell_id
                / "attempt_01"
                / "summary.json"
            ).read_text()
        )
        assert cell_summary["variant"] == spec.condition
        assert cell_summary["condition_id"] == f"{spec.condition}_t0"
        assert cell_summary["variant_label"] == grid.PROMPT_VARIANT_LABELS[
            spec.prompt_version
        ]
        assert cell_summary["prompt_version"] == spec.prompt_version
        assert cell_summary["config_hash"] == grid.model_config_for(spec).config_hash()


def test_infrastructure_resume_creates_new_attempt_and_skips_completed_cell(tmp_path):
    class FlakyRunner:
        def __init__(self):
            self.calls: list[str] = []
            self.failed = False

        def __call__(self, **kwargs):
            prompt = str(kwargs["prompt_version"])
            self.calls.append(prompt)
            if prompt == PROMPT_VERSION_B1_NEUTRAL_GROUNDED and not self.failed:
                self.failed = True
                raise OSError("mock infrastructure")
            return grid._mock_run_variant(**kwargs)

    flaky = FlakyRunner()
    first = _runner(tmp_path, output_name="resume", single_cell_runner=flaky)
    result = first.run()
    assert result["status"] == "infrastructure_failed"
    assert flaky.calls[:2] == [
        PROMPT_VERSION_P1_COMPACT,
        PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    ]

    resumed = _runner(
        tmp_path,
        output_name="resume",
        single_cell_runner=flaky,
        resume=True,
    )
    result = resumed.run()
    assert result["status"] == "completed"
    assert flaky.calls.count(PROMPT_VERSION_P1_COMPACT) == 3
    b0 = json.loads(
        (tmp_path / "resume" / "cells" / "r1_01_b0" / "cell_manifest.json").read_text()
    )
    b1 = json.loads(
        (tmp_path / "resume" / "cells" / "r1_02_b1" / "cell_manifest.json").read_text()
    )
    assert len(b0["attempts"]) == 1
    assert len(b1["attempts"]) == 2
    assert b1["attempts"][0]["status"] == "infrastructure_failed"
    assert b1["attempts"][1]["status"] == "completed"

    call_count = len(flaky.calls)
    already_complete = _runner(
        tmp_path,
        output_name="resume",
        single_cell_runner=flaky,
        resume=True,
    ).run()
    assert already_complete["already_completed"] is True
    assert len(flaky.calls) == call_count


class HighCostDelegate:
    def __init__(self):
        self.calls = 0

    def complete(
        self,
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        self.calls += 1
        return LLMCompletion(
            text="{}",
            model=model,
            prompt_tokens=0,
            completion_tokens=100_000_000,
            total_tokens=100_000_000,
            cached_tokens=0,
            latency_ms=1.0,
            thinking_disabled=thinking_disabled,
        )


def test_global_cost_guard_does_not_reimplement_transport_retry(tmp_path):
    class FailingDelegate:
        calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise TimeoutError("mock transport")

    delegate = FailingDelegate()
    guarded = grid.GlobalCostGuardClient(
        delegate=delegate,
        ledger_path=tmp_path / "guard.jsonl",
        cell_id="r1_01_b0",
        attempt_id="attempt_01",
    )
    with pytest.raises(TimeoutError):
        guarded.complete(
            [],
            model="deepseek-v4-flash",
            temperature=0.0,
            top_p=1.0,
            max_tokens=800,
            timeout_seconds=90.0,
            thinking_disabled=True,
        )
    assert delegate.calls == 1
    assert not (tmp_path / "guard.jsonl").exists()


def test_global_cost_guard_allows_crossing_call_then_blocks_next(tmp_path):
    delegate = HighCostDelegate()
    guarded = grid.GlobalCostGuardClient(
        delegate=delegate,
        ledger_path=tmp_path / "guard.jsonl",
        cell_id="r1_01_b0",
        attempt_id="attempt_01",
    )
    kwargs = {
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 800,
        "timeout_seconds": 90.0,
        "thinking_disabled": True,
    }
    guarded.complete([], **kwargs)
    assert grid.guard_total_cost(tmp_path / "guard.jsonl") >= 25.0
    with pytest.raises(grid.GlobalCostCapReached):
        guarded.complete([], **kwargs)
    assert delegate.calls == 1
    entries = grid._read_guard_entries(tmp_path / "guard.jsonl")
    assert len(entries) == 1
    assert "messages" not in entries[0]


def test_grid_cost_cap_stops_incomplete_attempt_without_summary(tmp_path):
    delegate = HighCostDelegate()

    def call_guard_until_blocked(**kwargs):
        client = kwargs["llm_client"]
        completion_kwargs = {
            "model": "deepseek-v4-flash",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 800,
            "timeout_seconds": 90.0,
            "thinking_disabled": True,
        }
        client.complete([], **completion_kwargs)
        client.complete([], **completion_kwargs)
        raise AssertionError("second guarded call must not return")

    runner = grid.StageBGridRunner(
        output_root=tmp_path / "cap",
        anchors_file=_anchors_file(tmp_path),
        raw_path=tmp_path / "unused.csv",
        artefact_dir=tmp_path / "unused-d1",
        launch=True,
        authorization_confirmation=grid.LIVE_CONFIRMATION,
        single_cell_runner=call_guard_until_blocked,
        preflight_runner=grid._mock_preflight_outbound_payloads,
        client_factory=lambda _model: delegate,
    )
    result = runner.run()
    assert result["status"] == "cost_cap_reached"
    assert delegate.calls == 1
    assert not (
        tmp_path / "cap" / "cells" / "r1_01_b0" / "attempt_01" / "summary.json"
    ).exists()
    cell = json.loads(
        (tmp_path / "cap" / "cells" / "r1_01_b0" / "cell_manifest.json").read_text()
    )
    assert cell["status"] == "cost_cap_stopped"
    assert cell["attempts"][0]["status"] == "cost_cap_stopped"
