from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from api_core import ApiEndpointConfig
from ml_core import DatasetSchema


TrainingArgsFactory = Callable[[Path], Any]
GraphRenderer = Callable[[Path], dict[str, Any]]
ExternalVerifier = Callable[[Path], dict[str, Any]]
PredictionVerifier = Callable[[Path], dict[str, Any]]
PredictionFramePostprocessor = Callable[[Any, str], Any]
TrainingRunner = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class ArtifactAppConfig:
    project_root: Path
    artifacts_dir: Path
    final_prediction_manifest: Path
    final_prediction_csv: Path
    final_comparison_markdown: Path | None
    schema: DatasetSchema
    api: ApiEndpointConfig
    excluded_run_summaries: frozenset[str] = frozenset()
    prediction_postprocessor: PredictionFramePostprocessor | None = None


@dataclass(frozen=True)
class ReproducibilityConfig:
    canonical_model_artifact: Path
    canonical_model_summary: Path
    prediction_table: Path
    prediction_embeddings: Path
    canonical_ranked_predictions: Path
    external_summary: Path
    canonical_graphs: Path
    output_artifact_name: str
    training_args_factory: TrainingArgsFactory
    training_runner: TrainingRunner
    prediction_verifier: PredictionVerifier
    external_verifier: ExternalVerifier
    graph_renderer: GraphRenderer


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    app: ArtifactAppConfig
    reproducibility: ReproducibilityConfig
