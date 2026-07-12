from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from .config import ReproducibilityConfig


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_tensors_match(left_path: Path, right_path: Path) -> bool:
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    return all(
        torch.equal(left["model_state_dict"][key], right["model_state_dict"][key])
        for key in left["model_state_dict"]
    )


class ReproducibilityRunner:
    def __init__(self, config: ReproducibilityConfig) -> None:
        self.config = config

    def run_once(self, work_dir: Path, render_graphs: bool) -> dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = work_dir / self.config.output_artifact_name
        summary = self.config.training_runner(self.config.training_args_factory(artifact_path))
        canonical_summary = load_json(self.config.canonical_model_summary)
        predictions = self.config.prediction_verifier(artifact_path)
        external = self.config.external_verifier(artifact_path)
        graphs = self.config.graph_renderer(work_dir / "graphs") if render_graphs else None
        metrics_match = summary["metrics"] == canonical_summary["metrics"]
        tensors_match = checkpoint_tensors_match(self.config.canonical_model_artifact, artifact_path)
        return {
            "work_dir": str(work_dir),
            "training_matches_canonical": metrics_match,
            "training_config": summary["training_config"],
            "tensor_exact_match": tensors_match,
            "predictions": predictions,
            "external": external,
            "graphs": graphs,
            "passed": (
                metrics_match
                and tensors_match
                and predictions["passed"]
                and external["passed"]
                and (graphs is None or graphs["passed"])
            ),
        }

    def run_cli(
        self,
        *,
        work_dir: Path | None,
        keep_work_dir: bool,
        render_graphs: bool,
        temp_prefix: str,
    ) -> dict[str, Any]:
        if work_dir is not None:
            return self.run_once(work_dir, render_graphs)
        if keep_work_dir:
            retained = Path(tempfile.mkdtemp(prefix=f"{temp_prefix}_keep_"))
            payload = self.run_once(retained, render_graphs)
            payload["work_dir_removed_on_exit"] = False
            return payload
        with tempfile.TemporaryDirectory(prefix=f"{temp_prefix}_") as tmp_name:
            payload = self.run_once(Path(tmp_name), render_graphs)
            payload["work_dir_removed_on_exit"] = True
            return payload
