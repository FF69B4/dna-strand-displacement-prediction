from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dissertation import finetune as finetune_module
from dissertation import regression_training as rg_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Akay result set in table_4.csv against local model predictions "
            "from the frozen regression head and fine-tuned DNA-BERT checkpoints."
        )
    )
    parser.add_argument(
        "--akay-path",
        type=Path,
        default=Path("data/extracted/clean/table_4.csv"),
        help="CSV containing the Akay result set.",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("bert/table_3_embeddings.npz"),
        help="Embedding NPZ aligned to the Akay sequences.",
    )
    parser.add_argument(
        "--availability-path",
        type=Path,
        default=Path("data/extracted/clean/table_3.csv"),
        help="CSV containing availability vectors aligned to the Akay sequences.",
    )
    parser.add_argument(
        "--frozen-artifact",
        type=Path,
        default=Path("training/artifacts/frozen_regression/k1_regression_head.pt"),
        help="Frozen regression-head checkpoint to evaluate.",
    )
    parser.add_argument(
        "--finetune-artifacts",
        type=Path,
        nargs="*",
        default=[
            Path("training/artifacts/finetuned_dnabert/k1_finetuned_dnabert.pt"),
            Path("training/artifacts/finetuned_dnabert/k1_finetuned_dnabert_tuned3.pt"),
        ],
        help="Fine-tuned checkpoint files to compare against Akay.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("training/artifacts/akay_comparisons/akay_comparison"),
        help="Prefix used for the generated CSV and JSON outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for model inference.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="Torch device used for fine-tuned model inference.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model weights from the local Hugging Face cache only.",
    )
    return parser.parse_args()


def rank_descending(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", ascending=False).to_numpy(dtype=int)


def top_overlap(ids: list[str], reference_scores: np.ndarray, candidate_scores: np.ndarray, k: int) -> float:
    frame = pd.DataFrame(
        {
            "id": ids,
            "reference": reference_scores,
            "candidate": candidate_scores,
        }
    )
    reference_top = set(frame.nlargest(k, "reference")["id"])
    candidate_top = set(frame.nlargest(k, "candidate")["id"])
    return float(len(reference_top & candidate_top) / k)


def compute_metrics(reference: np.ndarray, candidate: np.ndarray, ids: list[str]) -> dict[str, float]:
    reference_series = pd.Series(reference)
    candidate_series = pd.Series(candidate)
    log_reference = np.log10(reference)
    log_candidate = np.log10(candidate)
    log_ratio = log_candidate - log_reference
    safe_reference = np.maximum(np.abs(reference), np.finfo(np.float32).eps)
    safe_smape_denominator = np.maximum(
        (np.abs(reference) + np.abs(candidate)) / 2.0,
        np.finfo(np.float32).eps,
    )
    percent_error = ((candidate - reference) / safe_reference) * 100.0
    absolute_percent_error = np.abs(percent_error)
    symmetric_absolute_percent_error = (
        np.abs(candidate - reference) / safe_smape_denominator
    ) * 100.0
    absolute_log_ratio = np.abs(log_ratio)
    fold_error = 10.0 ** absolute_log_ratio

    return {
        "pearson_raw": float(reference_series.corr(candidate_series, method="pearson")),
        "spearman_raw": float(reference_series.corr(candidate_series, method="spearman")),
        "pearson_log10": float(pd.Series(log_reference).corr(pd.Series(log_candidate), method="pearson")),
        "spearman_log10": float(pd.Series(log_reference).corr(pd.Series(log_candidate), method="spearman")),
        "rmse_log10_vs_akay": float(np.sqrt(np.mean(np.square(log_candidate - log_reference)))),
        "mae_log10_vs_akay": float(np.mean(absolute_log_ratio)),
        "rmse_vs_akay": float(np.sqrt(np.mean(np.square(candidate - reference)))),
        "mae_vs_akay": float(np.mean(np.abs(candidate - reference))),
        "mean_signed_percent_error_vs_akay": float(np.mean(percent_error)),
        "mape_vs_akay_percent": float(np.mean(absolute_percent_error)),
        "median_ape_vs_akay_percent": float(np.median(absolute_percent_error)),
        "smape_vs_akay_percent": float(np.mean(symmetric_absolute_percent_error)),
        "mean_fold_error_vs_akay": float(np.mean(fold_error)),
        "median_fold_error_vs_akay": float(np.median(fold_error)),
        "within_2x_vs_akay": float(np.mean(fold_error <= 2.0)),
        "within_3x_vs_akay": float(np.mean(fold_error <= 3.0)),
        "within_10x_vs_akay": float(np.mean(fold_error <= 10.0)),
        "top_10_overlap": top_overlap(ids, reference, candidate, 10),
        "top_50_overlap": top_overlap(ids, reference, candidate, 50),
        "top_100_overlap": top_overlap(ids, reference, candidate, 100),
    }


def load_akay_frame(akay_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(akay_path)
    required_columns = {"No.", "Sequence", "Predicted k1"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {akay_path}: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    frame = frame.copy()
    frame["record_id"] = frame["No."].astype(str)
    frame["normalized_sequence"] = frame["Sequence"].map(finetune_module.normalize_sequence)
    frame["Predicted k1"] = frame["Predicted k1"].astype(np.float32)
    return frame


def load_availability_frame(availability_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(availability_path)
    required_columns = {"No.", "Sequence", rg_module.AVAILABILITY_COLUMN}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {availability_path}: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    frame = frame.copy()
    frame["record_id"] = frame["No."].astype(str)
    frame["availability_sequence"] = frame["Sequence"].map(finetune_module.normalize_sequence)
    return frame[["record_id", "availability_sequence", rg_module.AVAILABILITY_COLUMN]]


def load_embedding_frame(embedding_path: Path) -> pd.DataFrame:
    npz = np.load(embedding_path, allow_pickle=True)
    return pd.DataFrame(
        {
            "record_id": [str(item) for item in npz["ids"].tolist()],
            "embedding_sequence": [
                finetune_module.normalize_sequence(item)
                for item in npz["sequences"].tolist()
            ],
            "embedding_index": np.arange(len(npz["ids"])),
        }
    ), npz["embeddings"].astype(np.float32)


def build_frozen_predictions(
    merged_frame: pd.DataFrame,
    embeddings: np.ndarray,
    frozen_artifact: Path,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(frozen_artifact, map_location="cpu", weights_only=False)
    feature_config = checkpoint["feature_config"]
    model_config = checkpoint["model_config"]
    target_transform = checkpoint["target_config"]["target_transform"]

    feature_blocks: list[np.ndarray] = []
    if rg_module.uses_embeddings(feature_config["feature_set"]):
        embedding_indices = merged_frame["embedding_index"].astype(int).to_numpy()
        feature_blocks.append(embeddings[embedding_indices])
    if rg_module.uses_nn_features(feature_config["feature_set"]):
        nn_features, _ = rg_module.encode_nn_features(
            merged_frame["normalized_sequence"].tolist(),
            feature_config["nn_feature_mode"],
        )
        feature_blocks.append(nn_features.astype(np.float32))
    if rg_module.uses_availability_features(feature_config["feature_set"]):
        availability_features, _ = rg_module.encode_availability_features(
            merged_frame[rg_module.AVAILABILITY_COLUMN].tolist(),
            merged_frame["normalized_sequence"].tolist(),
        )
        feature_blocks.append(availability_features.astype(np.float32))

    feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    feature_mean = np.asarray(feature_config["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(feature_config["feature_std"], dtype=np.float32)
    scaled_features = ((feature_matrix - feature_mean) / feature_std).astype(np.float32)

    model = rg_module.RegressionHead(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        activation=model_config.get("activation", "gelu"),
        normalization=model_config.get("normalization", "layer"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(scaled_features), batch_size):
            stop = start + batch_size
            batch = torch.from_numpy(scaled_features[start:stop])
            predictions = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(predictions)

    predicted_model_scale = np.concatenate(outputs, axis=0)
    return rg_module.invert_target_transform(predicted_model_scale, target_transform)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but no MPS device is available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_finetuned_model(
    merged_frame: pd.DataFrame,
    artifact_path: Path,
    batch_size: int,
    device: torch.device,
    local_files_only: bool,
) -> np.ndarray:
    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    encoder_config = checkpoint["encoder_config"]
    nn_feature_config = checkpoint["nn_feature_config"]
    availability_feature_config = checkpoint.get("availability_feature_config", {})
    target_transform = checkpoint["target_config"]["target_transform"]

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_config["model_name"],
        trust_remote_code=encoder_config.get("trust_remote_code", False),
        local_files_only=local_files_only or encoder_config.get("local_files_only", False),
    )
    encoder = AutoModel.from_pretrained(
        encoder_config["model_name"],
        trust_remote_code=encoder_config.get("trust_remote_code", False),
        local_files_only=local_files_only or encoder_config.get("local_files_only", False),
    )
    finetune_module.configure_tokenizer(tokenizer, encoder)

    model = finetune_module.DNABertFineTuner(
        encoder=encoder,
        pooling=model_config["pooling"],
        head_hidden_dims=model_config["head_hidden_dims"],
        dropout=model_config["dropout"],
        nn_feature_dim=model_config.get("nn_feature_dim", 0),
        nn_projection_dim=model_config["nn_projection_dim"],
        availability_feature_dim=model_config.get("availability_feature_dim", 0),
        availability_projection_dim=model_config.get("availability_projection_dim", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    texts = [
        finetune_module.to_kmers(sequence, model_config["kmer_size"])
        for sequence in merged_frame["normalized_sequence"].tolist()
    ]

    nn_features = None
    if finetune_module.uses_nn_features(model_config["feature_set"]):
        nn_features, _ = finetune_module.encode_nn_features(
            merged_frame["normalized_sequence"].tolist(),
            model_config["nn_feature_mode"],
        )
        feature_mean = nn_feature_config["feature_mean"]
        feature_std = nn_feature_config["feature_std"]
        if feature_mean is not None and feature_std is not None:
            feature_mean = np.asarray(feature_mean, dtype=np.float32)
            feature_std = np.asarray(feature_std, dtype=np.float32)
            nn_features = ((nn_features - feature_mean) / feature_std).astype(np.float32)

    availability_features = None
    if finetune_module.uses_availability_features(model_config["feature_set"]):
        availability_features, _ = rg_module.encode_availability_features(
            merged_frame[rg_module.AVAILABILITY_COLUMN].tolist(),
            merged_frame["normalized_sequence"].tolist(),
        )
        feature_mean = availability_feature_config.get("feature_mean")
        feature_std = availability_feature_config.get("feature_std")
        if feature_mean is not None and feature_std is not None:
            feature_mean = np.asarray(feature_mean, dtype=np.float32)
            feature_std = np.asarray(feature_std, dtype=np.float32)
            availability_features = (
                (availability_features - feature_mean) / feature_std
            ).astype(np.float32)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            encoded = tokenizer(
                texts[start:stop],
                padding=True,
                truncation=True,
                max_length=model_config["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_nn = None
            if nn_features is not None:
                batch_nn = torch.from_numpy(nn_features[start:stop]).to(device)
            batch_availability = None
            if availability_features is not None:
                batch_availability = torch.from_numpy(availability_features[start:stop]).to(device)
            predictions = model(
                nn_features=batch_nn,
                availability_features=batch_availability,
                **encoded,
            )
            outputs.append(predictions.detach().cpu().numpy().astype(np.float32))

    predicted_model_scale = np.concatenate(outputs, axis=0)
    return finetune_module.invert_target_transform(predicted_model_scale, target_transform)


def main() -> None:
    args = parse_args()

    akay_frame = load_akay_frame(args.akay_path)
    availability_frame = load_availability_frame(args.availability_path)
    embedding_frame, embeddings = load_embedding_frame(args.embedding_path)
    merged = akay_frame.merge(embedding_frame, on="record_id", how="left", validate="one_to_one")
    merged = merged.merge(availability_frame, on="record_id", how="left", validate="one_to_one")
    if merged["embedding_index"].isna().any():
        missing_ids = merged.loc[merged["embedding_index"].isna(), "record_id"].tolist()
        raise ValueError(f"Missing embeddings for Akay ids: {missing_ids[:10]}")
    mismatched_sequences = merged["normalized_sequence"] != merged["embedding_sequence"]
    if mismatched_sequences.any():
        first = merged.loc[mismatched_sequences].iloc[0]
        raise ValueError(
            "Akay sequences do not match the embedding file for record "
            f"{first['record_id']}: {first['normalized_sequence']} vs {first['embedding_sequence']}"
        )
    if merged[rg_module.AVAILABILITY_COLUMN].isna().any():
        missing_ids = merged.loc[merged[rg_module.AVAILABILITY_COLUMN].isna(), "record_id"].tolist()
        raise ValueError(f"Missing availability vectors for Akay ids: {missing_ids[:10]}")
    availability_mismatch = merged["normalized_sequence"] != merged["availability_sequence"]
    if availability_mismatch.any():
        first = merged.loc[availability_mismatch].iloc[0]
        raise ValueError(
            "Akay sequences do not match the availability file for record "
            f"{first['record_id']}: {first['normalized_sequence']} vs {first['availability_sequence']}"
        )

    output_frame = merged[["No.", "record_id", "Sequence", "Predicted k1"]].copy()
    output_frame["akay_rank"] = rank_descending(output_frame["Predicted k1"].to_numpy())

    summaries: dict[str, dict[str, Any]] = {}
    ids = output_frame["record_id"].tolist()
    akay_scores = output_frame["Predicted k1"].to_numpy(dtype=np.float32)

    frozen_predictions = build_frozen_predictions(
        merged_frame=merged,
        embeddings=embeddings,
        frozen_artifact=args.frozen_artifact,
        batch_size=args.batch_size,
    )
    frozen_name = args.frozen_artifact.stem
    output_frame[f"{frozen_name}_predicted_k1"] = frozen_predictions
    output_frame[f"{frozen_name}_rank"] = rank_descending(frozen_predictions)
    summaries[frozen_name] = compute_metrics(akay_scores, frozen_predictions, ids)

    device = resolve_device(args.device)
    for artifact_path in args.finetune_artifacts:
        if not artifact_path.exists():
            continue
        predictions = predict_finetuned_model(
            merged_frame=merged,
            artifact_path=artifact_path,
            batch_size=args.batch_size,
            device=device,
            local_files_only=args.local_files_only,
        )
        model_name = artifact_path.stem
        output_frame[f"{model_name}_predicted_k1"] = predictions
        output_frame[f"{model_name}_rank"] = rank_descending(predictions)
        summaries[model_name] = compute_metrics(akay_scores, predictions, ids)

    output_frame = output_frame.sort_values("Predicted k1", ascending=False).reset_index(drop=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(csv_path, index=False)

    payload = {
        "akay_path": str(args.akay_path),
        "embedding_path": str(args.embedding_path),
        "availability_path": str(args.availability_path),
        "frozen_artifact": str(args.frozen_artifact),
        "finetune_artifacts": [str(path) for path in args.finetune_artifacts if path.exists()],
        "row_count": len(output_frame),
        "metrics": summaries,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
        "notes": (
            "Percentage-error fields are computed locally as signed percent error, "
            "MAPE, median absolute percentage error, and sMAPE. Log-scale agreement "
            "is summarized separately with log10 error, fold error, and within-kx "
            "rates relative to Akay's Predicted k1 values because no explicit "
            "Akay-specific percent-error formula was found in the local project files."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
