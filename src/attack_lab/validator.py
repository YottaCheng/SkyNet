"""Policy-driven validation for attack proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from attack_lab.governance import (
    CompiledFieldPolicy,
    CompiledGovernancePolicy,
    is_sentinel,
)
from attack_lab.types import AttackProposal, ValidityResult
from baf_data.config import FROZEN_CONFIG, DataLayerConfig


class AttackLabValidationError(RuntimeError):
    """Raised for programmer or policy-construction errors."""


@dataclass(frozen=True)
class EpisodeLockPreparation:
    """Values fixed before any first-attempt D1 feedback is produced."""

    locked_values: Mapping[str, Any]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintValidator:
    """Validate all attacker actions against one compiled governance policy."""

    policy: CompiledGovernancePolicy
    feature_columns: tuple[str, ...]
    enabled_action_keys: tuple[str, ...]

    @classmethod
    def from_policy(
        cls,
        policy: CompiledGovernancePolicy,
        *,
        enabled_action_keys: tuple[str, ...] | list[str] | None = None,
        data_config: DataLayerConfig = FROZEN_CONFIG,
    ) -> "ConstraintValidator":
        feature_columns = data_config.feature_columns
        model_actions = tuple(
            action
            for action in policy.available_action_keys
            if (
                (rule := policy.field_for_action(action)) is not None
                and rule.feature in feature_columns
            )
        )
        enabled = (
            model_actions
            if enabled_action_keys is None
            else tuple(enabled_action_keys)
        )
        invalid = [action for action in enabled if action not in model_actions]
        if invalid:
            raise AttackLabValidationError(
                "Requested actions are not permitted by compiled governance: "
                f"{invalid}."
            )
        if len(enabled) != len(set(enabled)):
            raise AttackLabValidationError("Enabled action keys must be unique.")
        return cls(
            policy=policy,
            feature_columns=feature_columns,
            enabled_action_keys=enabled,
        )

    @property
    def mutable_fields(self) -> tuple[str, ...]:
        """Raw-value actions only; proxy actions have separate abstract keys."""
        return tuple(
            action
            for action in self.enabled_action_keys
            if (
                (rule := self.policy.field_for_action(action)) is not None
                and rule.agent_action_mode == "raw_value"
            )
        )

    @property
    def proxy_actions(self) -> dict[str, tuple[str, ...]]:
        catalogue = self.policy.proxy_action_catalogue()
        return {
            action: catalogue[action]
            for action in self.enabled_action_keys
            if action in catalogue
        }

    def visible_fields(self, features: Mapping[str, Any]) -> dict[str, Any]:
        visible = self.policy.visible_fields(features)
        return {
            name: value
            for name, value in visible.items()
            if name in self.feature_columns
        }

    def prepare_episode_locks(
        self,
        baseline_features: Mapping[str, Any],
        first_proposal: AttackProposal,
    ) -> EpisodeLockPreparation:
        """Freeze every episode-static field before the first D1 call.

        Omitted fields freeze to the anchor.  Invalid first-submission values
        also fail closed to the anchor and make that first proposal INVALID;
        later attempts still cannot change the frozen field.
        """
        self._assert_baseline(baseline_features)
        changes = dict(first_proposal.changes)
        locked: dict[str, Any] = {}
        errors: list[str] = []
        for field_name in self.policy.locked_fields:
            if field_name not in self.feature_columns:
                continue
            rule = self.policy.fields[field_name]
            action_key = (
                rule.proxy_action_key
                if rule.agent_action_mode == "proxy_action"
                else field_name
            )
            anchor = baseline_features[field_name]
            if action_key is None or action_key not in changes:
                locked[field_name] = anchor
                continue
            parsed, error = self._resolve_action_value(
                action_key, changes[action_key], rule
            )
            if error is None:
                error = self._validate_value(rule, parsed, anchor)
            if error is not None:
                errors.append(error)
                locked[field_name] = anchor
            else:
                locked[field_name] = parsed
        return EpisodeLockPreparation(locked_values=locked, errors=tuple(errors))

    def mutable_feature_names(self) -> tuple[str, ...]:
        """Model features that are attacker-mutable under the compiled policy."""
        names: list[str] = []
        for action in self.enabled_action_keys:
            rule = self.policy.field_for_action(action)
            if (
                rule is not None
                and rule.is_mutable
                and rule.feature in self.feature_columns
            ):
                names.append(rule.feature)
        # Episode-locked mutable fields remain billable even after locking.
        for field_name in self.policy.locked_fields:
            if (
                field_name in self.feature_columns
                and field_name not in names
                and self.policy.fields[field_name].is_mutable
            ):
                names.append(field_name)
        return tuple(dict.fromkeys(names))

    def project_for_billing(
        self,
        baseline_features: Mapping[str, Any],
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Best-effort candidate projection used only for edit-cost accounting.

        This does not authorise the proposal and must never replace validation.
        Unresolvable actions still mark their target feature as changed so that
        invalid submissions can charge the proposed edit cost under the default
        budget rules.
        """
        self._assert_baseline(baseline_features)
        candidate = {
            name: baseline_features[name] for name in self.feature_columns
        }
        locks = dict(locked_values or {})
        for name, value in locks.items():
            if name in candidate:
                candidate[name] = value

        for action_key, raw_value in dict(proposal.changes).items():
            rule = self.policy.field_for_action(action_key)
            if rule is None or rule.feature not in self.feature_columns:
                continue
            if action_key not in self.enabled_action_keys and rule.feature not in locks:
                # Unknown/disabled actions are ignored for billing projection;
                # governance validation will still reject them.
                continue
            parsed, error = self._resolve_action_value(action_key, raw_value, rule)
            if error is None:
                candidate[rule.feature] = parsed
                continue
            # Unresolved but intentionally proposed change: bill as edited unless
            # the raw token clearly matches the anchor string form.
            anchor = baseline_features[rule.feature]
            if str(raw_value) != str(anchor):
                # Use a distinct marker object so equality with the anchor fails.
                candidate[rule.feature] = f"__unresolved_edit__:{raw_value!r}"
        return candidate

    def validate(
        self,
        baseline_features: Mapping[str, Any],
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None = None,
        pre_feedback_errors: tuple[str, ...] = (),
    ) -> ValidityResult:
        self._assert_baseline(baseline_features)
        errors = list(pre_feedback_errors)
        changes = dict(proposal.changes)
        locks = dict(locked_values or {})
        candidate = {
            name: baseline_features[name] for name in self.feature_columns
        }
        for name, value in locks.items():
            if name in candidate:
                candidate[name] = value

        changed_fields: set[str] = set()
        for action_key, raw_value in changes.items():
            rule = self.policy.field_for_action(action_key)
            if rule is None or action_key not in self.enabled_action_keys:
                errors.append("Action is not permitted by governance policy.")
                continue
            if rule.feature not in self.feature_columns:
                errors.append("Action is not permitted by governance policy.")
                continue
            parsed, error = self._resolve_action_value(action_key, raw_value, rule)
            if error is None:
                error = self._validate_value(
                    rule, parsed, baseline_features[rule.feature]
                )
            if error is not None:
                errors.append(error)
                continue
            if rule.is_episode_locked:
                if rule.feature not in locks:
                    errors.append("Episode lock state is missing; submission refused.")
                    continue
                if not _values_equal(parsed, locks[rule.feature]):
                    errors.append(
                        "Episode-locked action cannot change after first submission."
                    )
                    continue
            candidate[rule.feature] = parsed
            if not _values_equal(parsed, baseline_features[rule.feature]):
                changed_fields.add(rule.feature)

        for field_name in changed_fields:
            rule = self.policy.fields[field_name]
            errors.extend(self._relationship_errors(rule, candidate))

        if errors:
            return ValidityResult(False, tuple(dict.fromkeys(errors)), None)
        return ValidityResult(True, (), candidate)

    def _assert_baseline(self, baseline_features: Mapping[str, Any]) -> None:
        if set(baseline_features) != set(self.feature_columns):
            raise AttackLabValidationError(
                "Baseline application does not match the frozen model feature schema."
            )

    def _resolve_action_value(
        self,
        action_key: str,
        raw_value: Any,
        rule: CompiledFieldPolicy,
    ) -> tuple[Any, str | None]:
        if rule.agent_action_mode == "proxy_action":
            if action_key != rule.proxy_action_key:
                return None, "Proxy fields accept abstract actions only."
            action_name = str(raw_value)
            if action_name not in rule.resolved_proxy_actions:
                return None, "Unknown proxy action."
            return rule.resolved_proxy_actions[action_name], None
        return self._parse_value(rule, raw_value)

    def _parse_value(
        self, rule: CompiledFieldPolicy, raw_value: Any
    ) -> tuple[Any, str | None]:
        if rule.data_type == "categorical":
            return str(raw_value), None
        if raw_value is None or (
            isinstance(raw_value, float) and np.isnan(raw_value)
        ):
            return None, "Missing values cannot be proposed directly."
        if isinstance(raw_value, str) and raw_value.strip().lower() in {
            "",
            "nan",
            "none",
            "null",
        }:
            return None, "Missing values cannot be proposed directly."
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            return None, "Numeric action value could not parse."
        if rule.data_type in {"binary", "integer"}:
            if not float(number).is_integer():
                return None, "Integer action received a non-integer value."
            return int(number), None
        return float(number), None

    def _validate_value(
        self,
        rule: CompiledFieldPolicy,
        value: Any,
        anchor_value: Any,
    ) -> str | None:
        if (
            rule.sentinel_policy == "retain_anchor_only"
            and is_sentinel(value, rule.sentinel_spec)
            and not _values_equal(value, anchor_value)
        ):
            return "Sentinel values may only be retained from the anchor."
        if rule.data_type == "binary" and value not in (0, 1):
            return "Binary action must resolve to 0 or 1."
        if rule.allowed_values and not any(
            _values_equal(value, allowed) for allowed in rule.allowed_values
        ):
            return "Action value is outside the compiled train-supported domain."
        if rule.lower_bound is not None and float(value) < rule.lower_bound:
            return "Action value is below the compiled lower bound."
        if rule.upper_bound is not None and float(value) > rule.upper_bound:
            return "Action value is above the compiled upper bound."
        return None

    @staticmethod
    def _relationship_errors(
        rule: CompiledFieldPolicy,
        candidate: Mapping[str, Any],
    ) -> tuple[str, ...]:
        errors: list[str] = []
        for constraint in rule.hard_constraints:
            if constraint.get("type") != "conditional_train_range":
                continue
            conditions = tuple(constraint.get("condition_fields", ()))
            binning = dict(constraint.get("condition_binning", {}))
            wanted: dict[str, Any] = {}
            for condition in conditions:
                value = candidate[condition]
                width = float(binning.get(condition, 0))
                wanted[condition] = (
                    math_floor(float(value) / width) * width
                    if width > 0
                    else value
                )
            matches = [
                item
                for item in constraint.get("compiled_ranges", ())
                if all(
                    _values_equal(item["conditions"].get(name), value)
                    for name, value in wanted.items()
                )
            ]
            if not matches:
                errors.append(
                    "Action violates a compiled train-supported relationship."
                )
                continue
            value = float(candidate[rule.feature])
            if not any(
                float(item["min"]) <= value <= float(item["max"])
                for item in matches
            ):
                errors.append(
                    "Action violates a compiled train-supported relationship."
                )
        return tuple(errors)


def math_floor(value: float) -> int:
    """Small named helper keeps conditional binning deterministic."""
    return int(np.floor(value))


def _values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


__all__ = [
    "AttackLabValidationError",
    "ConstraintValidator",
    "EpisodeLockPreparation",
]
