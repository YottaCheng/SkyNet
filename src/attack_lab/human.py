"""Human attacker CLI interaction helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, TextIO

from attack_lab.environment import AttackEnvironment
from attack_lab.types import AttackProposal, Observation


class HumanAttackerError(ValueError):
    """Raised for unparseable human commands."""


@dataclass
class HumanAttacker:
    """Interactive human attacker with a pending proposal draft."""

    env: AttackEnvironment
    stdin: TextIO
    stdout: TextIO
    pending_changes: dict[str, Any] = field(default_factory=dict)

    def render_observation(self, obs: Observation) -> None:
        """Print attacker-view state only (no score/threshold)."""
        lines = [
            "",
            "=== Attack Laboratory (attacker view) ===",
            f"case_id: {obs.case_id}",
            f"attempt: {obs.attempt} / {obs.max_attempts}",
            f"remaining_attempts: {obs.remaining_attempts}",
            f"q_remaining: {obs.q_remaining}",
            f"m_max: {obs.m_max}",
            f"feedback_mode: {obs.feedback_mode}",
            f"mutable_fields: {', '.join(obs.mutable_fields)}",
            "proxy_actions: "
            + (
                "; ".join(
                    f"{key}=[{', '.join(actions)}]"
                    for key, actions in sorted(obs.proxy_actions.items())
                )
                or "(none)"
            ),
            "",
            "Visible fields (current application):",
        ]
        for name in sorted(obs.visible_fields):
            marker = "*" if name in obs.mutable_fields else " "
            lines.append(f"  {marker} {name} = {obs.visible_fields[name]!r}")
        if self.pending_changes:
            lines.append("")
            lines.append("Pending proposal changes:")
            for name, value in self.pending_changes.items():
                lines.append(f"  {name} = {value!r}")
        if obs.last_feedback is not None:
            lines.append("")
            lines.append(
                f"Last public feedback: {obs.last_feedback.label} — "
                f"{obs.last_feedback.message}"
            )
        lines.append("")
        lines.append(obs.instructions)
        lines.append(
            "Enter field=value to stage a change, or a command "
            "(submit / reset-current-proposal / show / quit)."
        )
        self.stdout.write("\n".join(lines) + "\n")
        self.stdout.flush()

    def parse_line(self, line: str) -> str | AttackProposal | None:
        """Parse one input line.

        Returns:
            'quit' | 'show' | 'reset' | AttackProposal | None (staged change only)
        """
        text = line.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"quit", "exit", "q"}:
            return "quit"
        if lowered in {"show", "status"}:
            return "show"
        if lowered in {"reset-current-proposal", "reset"}:
            return "reset"
        if lowered == "submit":
            return AttackProposal(
                changes=dict(self.pending_changes),
                raw_command=text,
            )

        # field=value or several whitespace-separated assignments
        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            raise HumanAttackerError(f"Could not parse input: {exc}") from exc

        staged_any = False
        for token in tokens:
            if "=" not in token:
                raise HumanAttackerError(
                    f"Unrecognised token {token!r}. Use field=value or a command."
                )
            field_name, raw_value = token.split("=", 1)
            field_name = field_name.strip()
            if not field_name:
                raise HumanAttackerError("Empty field name in assignment.")
            self.pending_changes[field_name] = _coerce_literal(raw_value)
            staged_any = True
        return None if staged_any else None

    def run(self) -> None:
        """Drive an interactive episode until success, budget end, or quit."""
        while not self.env.done:
            obs = self.env.observation()
            self.render_observation(obs)
            self.stdout.write("> ")
            self.stdout.flush()
            line = self.stdin.readline()
            if line == "":
                self.stdout.write("\nEOF — ending episode without success.\n")
                break
            try:
                parsed = self.parse_line(line)
            except HumanAttackerError as exc:
                self.stdout.write(f"ERROR: {exc}\n")
                continue

            if parsed == "quit":
                self.stdout.write("Quit requested.\n")
                break
            if parsed == "show":
                continue
            if parsed == "reset":
                self.pending_changes.clear()
                self.stdout.write("Pending proposal cleared.\n")
                continue
            if parsed is None:
                self.stdout.write(
                    f"Staged changes: {self.pending_changes}\n"
                )
                continue
            if isinstance(parsed, AttackProposal):
                if not parsed.changes:
                    self.stdout.write(
                        "ERROR: no pending changes to submit. "
                        "Stage field=value first.\n"
                    )
                    continue
                record = self.env.step(parsed)
                self.pending_changes.clear()
                self.stdout.write(
                    f"Public feedback: {record.public_feedback.label} — "
                    f"{record.public_feedback.message}\n"
                    f"Remaining attempts: {record.public_feedback.remaining_attempts}\n"
                )
                if self.env.done:
                    break

        if self.env.done:
            result = self.env.result()
            self.stdout.write(
                f"\nEpisode finished: success={result.success}, "
                f"reason={result.stop_reason}, attempts={result.attempts_used}\n"
            )
        else:
            self.env.abort(reason="attacker_quit")
            result = self.env.result()
            self.stdout.write(
                f"\nEpisode finished: success={result.success}, "
                f"reason={result.stop_reason}, attempts={result.attempts_used}\n"
            )


def _coerce_literal(raw_value: str) -> Any:
    """Best-effort literal coercion for CLI values; validator does final checks."""
    text = raw_value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "nan"}:
        return float("nan")
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text
