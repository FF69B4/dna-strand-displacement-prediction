from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api_core import ApiEndpointConfig
from experiment_core import ArtifactAppConfig, ExperimentConfig, ReproducibilityConfig
from ml_core import (
    DatasetSchema,
    DnaFeatureSetBuilder,
    DnaSequenceValidator,
    ModelArtifact,
    PredictionService,
)
from viz_core import FigureOutput, monochrome_serif_theme

from dissertation import compare_to_akay
from dissertation import graphs as generate_paper_graphs
from dissertation import regression_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "training" / "artifacts"
FINAL_MODEL_ARTIFACT = ARTIFACTS / "frozen_regression" / "k1_regression_head_with_availability_tuned1.pt"
FINAL_MODEL_SUMMARY = FINAL_MODEL_ARTIFACT.with_suffix(".json")
FINAL_TABLE3_RANKED = ARTIFACTS / "final_predictions" / "table_3_final_predictions_ranked.csv"
FINAL_AKAY_SUMMARY = ARTIFACTS / "akay_comparisons" / "akay_comparison_with_availability_tuned1.json"
CANONICAL_GRAPHS = ARTIFACTS / "paper_graphs"


def dissertation_schema() -> DatasetSchema:
    return DatasetSchema(
        id_column="No.",
        sequence_column="Sequence",
        availability_column="Nucleotide Availability",
        validator=DnaSequenceValidator(),
        feature_set_builder=DnaFeatureSetBuilder(),
    )


def selected_model_args(artifact_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        table_path=Path("data/extracted/clean/table_1.csv"),
        embedding_path=Path("bert/table_1_embeddings.npz"),
        artifact_path=artifact_path,
        target_column="k1",
        id_column="No.",
        sequence_column="Sequence",
        availability_column="Nucleotide Availability",
        feature_set="embeddings+nn+availability",
        nn_feature_mode="both",
        target_transform="auto",
        hidden_dims=[768, 256],
        dropout=0.1,
        activation="gelu",
        normalization="layer",
        batch_size=128,
        epochs=250,
        learning_rate=0.0005,
        weight_decay=0.0003,
        patience=30,
        min_delta=0.0001,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=7,
        device="cpu",
        print_every=10,
        domain_adapt_path=None,
        domain_adapt_embedding_path=Path("bert/table_3_embeddings.npz"),
        domain_adapt_id_column="No.",
        domain_adapt_sequence_column="Sequence",
        domain_adapt_availability_column="Nucleotide Availability",
        domain_weight_clip_min=0.25,
        domain_weight_clip_max=4.0,
        domain_classifier_c=1.0,
    )


def verify_ranked_predictions(artifact_path: Path) -> dict[str, Any]:
    result = PredictionService(
        ModelArtifact.load(artifact_path),
        schema=dissertation_schema(),
    ).predict_table(
        table_path=PROJECT_ROOT / "data/extracted/clean/table_3.csv",
        embedding_path=PROJECT_ROOT / "bert/table_3_embeddings.npz",
        batch_size=256,
    )
    generated = result.frame.copy().sort_values("predicted_rank").reset_index(drop=True)
    generated["No."] = generated["No."].astype(str)
    generated["predicted_k1_int"] = generated["predicted_k1"].astype(int)
    canonical = pd.read_csv(FINAL_TABLE3_RANKED).sort_values("predicted_rank").reset_index(drop=True)
    canonical["No."] = canonical["No."].astype(str)
    prediction_diff = generated["predicted_k1_int"].to_numpy() - canonical["predicted_k1"].to_numpy()
    rank_diff = generated["predicted_rank"].to_numpy() - canonical["predicted_rank"].to_numpy()
    passed = generated["No."].tolist() == canonical["No."].tolist() and bool(np.all(rank_diff == 0))
    return {
        "rows": len(generated),
        "ordered_ids_match": generated["No."].tolist() == canonical["No."].tolist(),
        "rank_exact_match": bool(np.all(rank_diff == 0)),
        "max_abs_rank_diff": int(np.max(np.abs(rank_diff))),
        "prediction_int_max_abs_diff": int(np.max(np.abs(prediction_diff))),
        "prediction_int_nonzero_diffs": int(np.count_nonzero(prediction_diff)),
        "top_id": str(generated.iloc[0]["No."]),
        "passed": passed,
    }


def verify_external_metrics(artifact_path: Path) -> dict[str, Any]:
    akay_frame = compare_to_akay.load_akay_frame(PROJECT_ROOT / "data/extracted/clean/table_4.csv")
    availability_frame = compare_to_akay.load_availability_frame(
        PROJECT_ROOT / "data/extracted/clean/table_3.csv"
    )
    embedding_frame, embeddings = compare_to_akay.load_embedding_frame(
        PROJECT_ROOT / "bert/table_3_embeddings.npz"
    )
    merged = akay_frame.merge(embedding_frame, on="record_id", how="left", validate="one_to_one")
    merged = merged.merge(availability_frame, on="record_id", how="left", validate="one_to_one")
    predictions = compare_to_akay.build_frozen_predictions(merged, embeddings, artifact_path, batch_size=32)
    metrics = compare_to_akay.compute_metrics(
        merged["Predicted k1"].to_numpy(dtype=np.float32),
        predictions,
        merged["record_id"].tolist(),
    )
    canonical = json.loads(FINAL_AKAY_SUMMARY.read_text(encoding="utf-8"))["metrics"][
        "k1_regression_head_with_availability_tuned1"
    ]
    diffs = {key: metrics[key] - canonical[key] for key in canonical}
    passed = all(value == 0 for value in diffs.values())
    return {
        "exact_match": passed,
        "max_abs_diff": max(abs(value) for value in diffs.values()),
        "selected_metrics": {
            "pearson_log10": metrics["pearson_log10"],
            "spearman_log10": metrics["spearman_log10"],
            "within_2x_vs_akay": metrics["within_2x_vs_akay"],
            "top_10_overlap": metrics["top_10_overlap"],
            "top_50_overlap": metrics["top_50_overlap"],
            "top_100_overlap": metrics["top_100_overlap"],
        },
        "passed": passed,
    }


def render_graphs(output_dir: Path) -> dict[str, Any]:
    generate_paper_graphs.OUTPUT = output_dir
    generate_paper_graphs.THEME = monochrome_serif_theme()
    generate_paper_graphs.THEME.apply()
    generate_paper_graphs.FIGURES = FigureOutput(output_dir, generate_paper_graphs.THEME)
    generate_paper_graphs.main()
    files = sorted(path.name for path in output_dir.iterdir())
    canonical_files = sorted(path.name for path in CANONICAL_GRAPHS.iterdir() if path.is_file())
    passed = files == canonical_files
    return {
        "output_dir": str(output_dir),
        "file_count": len(files),
        "png_count": len([name for name in files if name.endswith(".png")]),
        "matches_canonical_file_set": passed,
        "files": files,
        "passed": passed,
    }


def integerize_k1_predictions(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    if prediction_column == "predicted_k1":
        frame[prediction_column] = frame[prediction_column].astype(int)
    return frame


CONFIG = ExperimentConfig(
    name="Dissertation ML API",
    app=ArtifactAppConfig(
        project_root=PROJECT_ROOT,
        artifacts_dir=ARTIFACTS,
        final_prediction_manifest=ARTIFACTS / "final_predictions" / "table_3_final_predictions.json",
        final_prediction_csv=FINAL_TABLE3_RANKED,
        final_comparison_markdown=ARTIFACTS / "model_comparison" / "final_model_comparison.md",
        schema=dissertation_schema(),
        api=ApiEndpointConfig(
            name="Dissertation ML API",
            predictions_path="/predictions/table3",
            prediction_path="/predictions/table3/<record_id>",
            recompute_path="/predictions/table3/recompute",
            recompute_message="Use POST to recompute table 3 predictions.",
            recompute_body_example={
                "batch_size": 256,
                "table_path": "optional path, defaults to final manifest table_path",
                "embedding_path": "optional path, defaults to final manifest embedding_path",
                "artifact_path": "optional path, defaults to final manifest artifact_path",
            },
        ),
        excluded_run_summaries=frozenset(
            {
                "table_3_final_predictions.json",
                "final_model_comparison.json",
                "domain_adaptation_percentage_summary.json",
            }
        ),
        prediction_postprocessor=integerize_k1_predictions,
    ),
    reproducibility=ReproducibilityConfig(
        canonical_model_artifact=FINAL_MODEL_ARTIFACT,
        canonical_model_summary=FINAL_MODEL_SUMMARY,
        prediction_table=PROJECT_ROOT / "data/extracted/clean/table_3.csv",
        prediction_embeddings=PROJECT_ROOT / "bert/table_3_embeddings.npz",
        canonical_ranked_predictions=FINAL_TABLE3_RANKED,
        external_summary=FINAL_AKAY_SUMMARY,
        canonical_graphs=CANONICAL_GRAPHS,
        output_artifact_name="k1_regression_head_with_availability_tuned1.pt",
        training_args_factory=selected_model_args,
        training_runner=regression_training.train_regression_head,
        prediction_verifier=verify_ranked_predictions,
        external_verifier=verify_external_metrics,
        graph_renderer=render_graphs,
    ),
)
