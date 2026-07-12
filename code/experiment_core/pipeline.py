from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_core import ModelArtifact, PredictionService
from runtime.runtime import get_runtime

from .config import ArtifactAppConfig


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe_frame = frame.replace({np.nan: None})
    return [
        {key: _json_ready(value) for key, value in row.items()}
        for row in safe_frame.to_dict(orient="records")
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class ArtifactPredictionPipeline:
    def __init__(self, config: ArtifactAppConfig) -> None:
        self.config = config

    def _resolve_project_path(self, path: str | Path) -> Path:
        resolved = (
            (self.config.project_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
        if self.config.project_root not in resolved.parents and resolved != self.config.project_root:
            raise ValueError(f"Path is outside the project directory: {path}")
        return resolved

    @lru_cache(maxsize=1)
    def final_prediction_manifest(self) -> dict[str, Any]:
        path = self.config.final_prediction_manifest
        if not path.exists():
            raise FileNotFoundError(f"Final prediction manifest not found: {path}")
        return load_json(path)

    @lru_cache(maxsize=1)
    def final_prediction_frame(self) -> pd.DataFrame:
        path = self.config.final_prediction_csv
        if not path.exists():
            raise FileNotFoundError(f"Final ranked predictions not found: {path}")
        return pd.read_csv(path)

    def final_model_info(self) -> dict[str, Any]:
        manifest = self.final_prediction_manifest()
        artifact_path = self._resolve_project_path(manifest["artifact_path"])
        summary_path = artifact_path.with_suffix(".json")
        artifact_summary = load_json(summary_path) if summary_path.exists() else None
        comparison = (
            self.config.final_comparison_markdown.read_text(encoding="utf-8")
            if self.config.final_comparison_markdown
            and self.config.final_comparison_markdown.exists()
            else None
        )
        return {
            "selected_artifact": str(artifact_path.relative_to(self.config.project_root)),
            "prediction_manifest": manifest,
            "artifact_summary": artifact_summary,
            "final_comparison_markdown": comparison,
        }

    def list_predictions(self, limit: int = 25, offset: int = 0) -> dict[str, Any]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if offset < 0:
            raise ValueError("offset must be zero or greater")
        frame = self.final_prediction_frame().sort_values("predicted_rank")
        return {
            "total": len(frame),
            "limit": limit,
            "offset": offset,
            "items": frame_to_records(frame.iloc[offset : offset + limit]),
        }

    def get_prediction(self, record_id: str) -> dict[str, Any] | None:
        frame = self.final_prediction_frame()
        matches = frame.loc[frame[self.config.schema.id_column].astype(str) == str(record_id)]
        return None if matches.empty else frame_to_records(matches.head(1))[0]

    def list_runs(self) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for summary_path in sorted(self.config.artifacts_dir.rglob("*.json")):
            if summary_path.name in self.config.excluded_run_summaries:
                continue
            try:
                summary = load_json(summary_path)
            except json.JSONDecodeError:
                continue
            predictions_path = summary.get("predictions_path")
            metrics = summary.get("metrics", {})
            test_raw = metrics.get("test", {}).get("raw", {}) if isinstance(metrics, dict) else {}
            runs.append(
                {
                    "name": summary_path.stem,
                    "summary_path": str(summary_path.relative_to(self.config.project_root)),
                    "artifact_path": summary.get("artifact_path"),
                    "predictions_path": predictions_path,
                    "target_column": summary.get("target_column"),
                    "target_transform": summary.get("target_transform"),
                    "feature_count": summary.get("feature_count"),
                    "row_count": summary.get("row_count"),
                    "test_r2": test_raw.get("r2"),
                    "test_rmse": test_raw.get("rmse"),
                    "test_mae": test_raw.get("mae"),
                    "has_predictions": bool(
                        predictions_path and self._resolve_project_path(predictions_path).exists()
                    ),
                }
            )
        return {"total": len(runs), "runs": runs}

    def get_run_summary(self, run_name: str) -> dict[str, Any] | None:
        matches = sorted(self.config.artifacts_dir.rglob(f"{run_name}.json"))
        return None if not matches else load_json(matches[0])

    def get_run_predictions(self, run_name: str, limit: int = 25, offset: int = 0) -> dict[str, Any]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if offset < 0:
            raise ValueError("offset must be zero or greater")
        summary = self.get_run_summary(run_name)
        if summary is None:
            raise FileNotFoundError(f"No run summary found for {run_name!r}")
        predictions_path = summary.get("predictions_path")
        if not predictions_path:
            raise FileNotFoundError(f"Run {run_name!r} does not declare a predictions_path")
        path = self._resolve_project_path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"Prediction CSV not found for run {run_name!r}: {path}")
        frame = pd.read_csv(path)
        return {
            "run": run_name,
            "predictions_path": str(path.relative_to(self.config.project_root)),
            "total": len(frame),
            "limit": limit,
            "offset": offset,
            "items": frame_to_records(frame.iloc[offset : offset + limit]),
        }

    def recompute_predictions(
        self,
        table_path: str | Path | None = None,
        embedding_path: str | Path | None = None,
        artifact_path: str | Path | None = None,
        batch_size: int = 256,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        manifest = self.final_prediction_manifest()
        table = self._resolve_project_path(table_path or manifest["table_path"])
        embeddings = self._resolve_project_path(embedding_path or manifest["embedding_path"])
        artifact = self._resolve_project_path(artifact_path or manifest["artifact_path"])
        prediction = PredictionService(
            ModelArtifact.load(artifact),
            schema=self.config.schema,
        ).predict_table(table_path=table, embedding_path=embeddings, batch_size=batch_size)
        output = prediction.frame.copy()
        prediction_column = f"predicted_{prediction.target_column}"
        if self.config.prediction_postprocessor is not None:
            output = self.config.prediction_postprocessor(output, prediction_column)
        payload = {
            "table_path": str(table.relative_to(self.config.project_root)),
            "embedding_path": str(embeddings.relative_to(self.config.project_root)),
            "artifact_path": str(artifact.relative_to(self.config.project_root)),
            "row_count": len(output),
            f"{prediction_column}_min": _json_ready(output[prediction_column].min()),
            f"{prediction_column}_max": _json_ready(output[prediction_column].max()),
            "top_10": frame_to_records(output.head(10)),
        }
        get_runtime().state["last_recomputed_predictions"] = payload
        return payload
