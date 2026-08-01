"""Episode trajectory and public-transcript logging."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from attack_lab.paths import new_run_directory
from attack_lab.types import EpisodeResult, StepRecord, to_jsonable


@dataclass
class TrajectoryLogger:
    """Write researcher-internal and attacker-public artefacts for one episode."""

    run_dir: Path
    run_id: str

    @classmethod
    def create(
        cls,
        run_id: str | None = None,
        *,
        parent: Path | None = None,
    ) -> "TrajectoryLogger":
        run_dir = new_run_directory(run_id, parent=parent)
        return cls(run_dir=run_dir, run_id=run_dir.name)

    @property
    def trajectory_path(self) -> Path:
        return self.run_dir / "trajectory.jsonl"

    @property
    def public_transcript_path(self) -> Path:
        return self.run_dir / "public_transcript.txt"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def governance_manifest_path(self) -> Path:
        return self.run_dir / "governance_manifest.json"

    def append_step(self, step: StepRecord) -> None:
        payload = to_jsonable(asdict(step))
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        public_lines = [
            f"attempt={step.attempt}",
            f"proposed_changes={json.dumps(to_jsonable(step.proposed_changes), sort_keys=True)}",
            f"valid={step.validity.is_valid}",
        ]
        public_lines.append(f"public_feedback={step.public_feedback.label}")
        public_lines.append(f"public_message={step.public_feedback.message}")
        public_lines.append(f"success={step.success}")
        public_lines.append(f"elapsed_ms={step.elapsed_ms:.3f}")
        public_lines.append("---")
        with self.public_transcript_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(public_lines) + "\n")

    def write_governance_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Record compiled policy provenance outside the attacker transcript."""
        self.governance_manifest_path.write_text(
            json.dumps(to_jsonable(dict(manifest)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            **dict(manifest),
        }
        # Enrich with package versions when readily available.
        for package in ("numpy", "pandas", "sklearn", "xgboost", "joblib"):
            try:
                module = __import__(package if package != "sklearn" else "sklearn")
                payload["software"][package] = getattr(module, "__version__", "unknown")
            except Exception:  # noqa: BLE001
                payload["software"][package] = "unavailable"
        self.manifest_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_episode_summary(self, episode: EpisodeResult) -> None:
        path = self.run_dir / "episode_result.json"
        path.write_text(
            json.dumps(to_jsonable(asdict(episode)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
