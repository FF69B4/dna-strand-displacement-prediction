from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.runtime import get_runtime


BASES = "ACGT"
DINUCLEOTIDES = [left + right for left in BASES for right in BASES]
DINUCLEOTIDE_TO_INDEX = {pair: index for index, pair in enumerate(DINUCLEOTIDES)}
VALID_BASES = set(BASES)
AVAILABILITY_COLUMN = "Nucleotide Availability"


@dataclass
class PreparedDataset:
    ids: list[str]
    sequences: list[str]
    features: np.ndarray
    targets_raw: np.ndarray
    targets_model: np.ndarray
    feature_names: list[str]
    embedding_metadata: dict[str, Any]
    target_column: str
    target_transform: str


@dataclass
class PreparedFeatureMatrix:
    ids: list[str]
    sequences: list[str]
    features: np.ndarray
    feature_names: list[str]
    embedding_metadata: dict[str, Any]


def build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def build_normalization(name: str, hidden_dim: int) -> nn.Module | None:
    if name == "layer":
        return nn.LayerNorm(hidden_dim)
    if name == "batch":
        return nn.BatchNorm1d(hidden_dim)
    if name == "none":
        return None
    raise ValueError(f"Unsupported normalization: {name}")


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str = "gelu",
        normalization: str = "layer",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            normalization_layer = build_normalization(normalization, hidden_dim)
            if normalization_layer is not None:
                layers.append(normalization_layer)
            layers.append(build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression head on precomputed DNA-BERT embeddings plus "
            "nearest-neighbor strand features."
        )
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("data/extracted/clean/table_1.csv"),
        help="CSV containing sequences and regression targets.",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("bert/table_1_embeddings.npz"),
        help="NPZ file produced by the dissertation DNA-BERT embedding step.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("training/artifacts/frozen_regression/k1_regression_head.pt"),
        help="Where to save the trained torch artifact.",
    )
    parser.add_argument(
        "--target-column",
        default="k1",
        help="Target column to predict. Use k1 or k2.",
    )
    parser.add_argument(
        "--id-column",
        default="No.",
        help="Identifier column used to align rows with the embedding file.",
    )
    parser.add_argument(
        "--sequence-column",
        default="Sequence",
        help="Sequence column used for alignment and NN feature generation.",
    )
    parser.add_argument(
        "--availability-column",
        default=AVAILABILITY_COLUMN,
        help="Column containing per-position nucleotide availability scores.",
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            "embeddings+nn+availability",
            "embeddings+availability",
            "nn+availability",
            "embeddings+nn",
            "embeddings",
            "nn",
            "availability",
        ),
        default="embeddings+nn",
        help="Which feature families to feed into the regression head.",
    )
    parser.add_argument(
        "--nn-feature-mode",
        choices=("positional", "frequency", "both"),
        default="both",
        help="How to encode nearest-neighbor dinucleotide features from each strand.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("auto", "none", "log10", "ln"),
        default="auto",
        help="Target transform applied before fitting.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 128],
        help="Hidden layer sizes for the regression head.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
        help="Dropout used between hidden layers.",
    )
    parser.add_argument(
        "--activation",
        choices=("gelu", "relu", "silu", "tanh", "elu", "identity"),
        default="gelu",
        help="Activation function used after each hidden layer.",
    )
    parser.add_argument(
        "--normalization",
        choices=("layer", "batch", "none"),
        default="layer",
        help="Normalization layer used after each hidden linear layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience measured in epochs.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation improvement required to reset patience.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of samples reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for data splitting and optimization.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="How often to print epoch progress.",
    )
    parser.add_argument(
        "--domain-adapt-path",
        type=Path,
        default=None,
        help="Optional unlabeled target-domain CSV used to compute domain-importance weights.",
    )
    parser.add_argument(
        "--domain-adapt-embedding-path",
        type=Path,
        default=None,
        help="Embedding NPZ aligned to --domain-adapt-path.",
    )
    parser.add_argument(
        "--domain-adapt-id-column",
        default="No.",
        help="Identifier column for the unlabeled target-domain table.",
    )
    parser.add_argument(
        "--domain-adapt-sequence-column",
        default="Sequence",
        help="Sequence column for the unlabeled target-domain table.",
    )
    parser.add_argument(
        "--domain-adapt-availability-column",
        default=AVAILABILITY_COLUMN,
        help="Availability column for the unlabeled target-domain table.",
    )
    parser.add_argument(
        "--domain-weight-clip-min",
        type=float,
        default=0.25,
        help="Minimum clipped importance weight before renormalization.",
    )
    parser.add_argument(
        "--domain-weight-clip-max",
        type=float,
        default=4.0,
        help="Maximum clipped importance weight before renormalization.",
    )
    parser.add_argument(
        "--domain-classifier-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the domain logistic-regression model.",
    )
    return parser.parse_args()


def normalize_sequence(sequence: str) -> str:
    cleaned = "".join(str(sequence).upper().split())
    invalid = sorted(set(cleaned) - VALID_BASES)
    if invalid:
        raise ValueError(
            f"Sequence contains unsupported bases {invalid}. Allowed bases are {sorted(VALID_BASES)}."
        )
    return cleaned


def feature_set_parts(feature_set: str) -> set[str]:
    return set(feature_set.split("+"))


def uses_embeddings(feature_set: str) -> bool:
    return "embeddings" in feature_set_parts(feature_set)


def uses_nn_features(feature_set: str) -> bool:
    return "nn" in feature_set_parts(feature_set)


def uses_availability_features(feature_set: str) -> bool:
    return "availability" in feature_set_parts(feature_set)


def scalar_from_npz(npz_file: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in npz_file:
        return None
    value = npz_file[key]
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0].item() if hasattr(value[0], "item") else value[0]
    return value.tolist() if isinstance(value, np.ndarray) else value


def choose_target_transform(transform_name: str, values: np.ndarray) -> str:
    if transform_name != "auto":
        return transform_name
    if np.all(values > 0):
        return "log10"
    return "none"


def apply_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "none":
        return values.astype(np.float32)
    if transform_name == "log10":
        return np.log10(values).astype(np.float32)
    if transform_name == "ln":
        return np.log(values).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {transform_name}")


def invert_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "none":
        return values.astype(np.float32)
    if transform_name == "log10":
        return np.power(10.0, values).astype(np.float32)
    if transform_name == "ln":
        return np.exp(values).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {transform_name}")


def encode_nn_features(sequences: list[str], mode: str) -> tuple[np.ndarray, list[str]]:
    if not sequences:
        raise ValueError("No sequences available for nearest-neighbor feature generation.")

    pair_count = len(sequences[0]) - 1
    if pair_count <= 0:
        raise ValueError("Sequences must be at least length 2 for nearest-neighbor features.")

    positional_features: np.ndarray | None = None
    positional_names: list[str] = []
    frequency_features: np.ndarray | None = None
    frequency_names: list[str] = []

    if mode in {"positional", "both"}:
        positional_features = np.zeros(
            (len(sequences), pair_count * len(DINUCLEOTIDES)), dtype=np.float32
        )
        for position in range(pair_count):
            for pair in DINUCLEOTIDES:
                positional_names.append(f"nn_pos_{position + 1:02d}_{pair}")

    if mode in {"frequency", "both"}:
        frequency_features = np.zeros((len(sequences), len(DINUCLEOTIDES)), dtype=np.float32)
        frequency_names = [f"nn_freq_{pair}" for pair in DINUCLEOTIDES]

    for row_index, sequence in enumerate(sequences):
        if len(sequence) != pair_count + 1:
            raise ValueError(
                "All sequences must share the same length when using positional NN features."
            )

        counts = np.zeros(len(DINUCLEOTIDES), dtype=np.float32)
        for position in range(pair_count):
            pair = sequence[position : position + 2]
            pair_index = DINUCLEOTIDE_TO_INDEX[pair]
            counts[pair_index] += 1.0
            if positional_features is not None:
                offset = position * len(DINUCLEOTIDES)
                positional_features[row_index, offset + pair_index] = 1.0

        if frequency_features is not None:
            frequency_features[row_index] = counts / float(pair_count)

    matrices: list[np.ndarray] = []
    feature_names: list[str] = []
    if positional_features is not None:
        matrices.append(positional_features)
        feature_names.extend(positional_names)
    if frequency_features is not None:
        matrices.append(frequency_features)
        feature_names.extend(frequency_names)

    return np.concatenate(matrices, axis=1), feature_names


def parse_numeric_vector(raw_value: Any) -> np.ndarray:
    if isinstance(raw_value, np.ndarray):
        vector = raw_value.astype(np.float32)
    elif isinstance(raw_value, (list, tuple)):
        vector = np.asarray(raw_value, dtype=np.float32)
    elif isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse vector literal: {raw_value!r}") from exc
        vector = np.asarray(parsed, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported vector value type: {type(raw_value)!r}")

    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D numeric vector, got shape {vector.shape}")
    return vector


def encode_availability_features(
    raw_values: list[Any],
    sequences: list[str],
) -> tuple[np.ndarray, list[str]]:
    if not raw_values:
        raise ValueError("No availability values available for feature generation.")

    matrix: list[np.ndarray] = []
    expected_length = len(sequences[0])
    for index, (raw_value, sequence) in enumerate(zip(raw_values, sequences, strict=True)):
        vector = parse_numeric_vector(raw_value)
        if len(sequence) != expected_length:
            raise ValueError("All sequences must share the same length for availability features.")
        if vector.shape[0] != len(sequence):
            raise ValueError(
                "Availability vector length does not match sequence length for row "
                f"{index}: {vector.shape[0]} vs {len(sequence)}"
            )
        matrix.append(vector)

    feature_names = [f"availability_{position + 1:02d}" for position in range(expected_length)]
    return np.vstack(matrix).astype(np.float32), feature_names


def load_aligned_table(
    table_path: Path,
    embedding_path: Path,
    id_column: str,
    sequence_column: str,
    availability_column: str,
    target_column: str,
    feature_set: str,
    nn_feature_mode: str,
    target_transform: str,
) -> PreparedDataset:
    table = pd.read_csv(table_path)
    required_columns = {id_column, sequence_column, target_column}
    if uses_availability_features(feature_set):
        required_columns.add(availability_column)
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {table_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(table.columns)}"
        )

    selected_columns = [id_column, sequence_column, target_column]
    if uses_availability_features(feature_set):
        selected_columns.append(availability_column)
    rows = table.loc[table[target_column].notna(), selected_columns].copy()
    rows["record_id"] = rows[id_column].astype(str)
    rows["normalized_sequence"] = rows[sequence_column].map(normalize_sequence)
    rows["target_raw"] = rows[target_column].astype(np.float32)

    if rows["record_id"].duplicated().any():
        duplicate_ids = rows.loc[rows["record_id"].duplicated(), "record_id"].unique().tolist()
        raise ValueError(f"Duplicate ids found in {table_path}: {duplicate_ids[:10]}")

    npz_file = np.load(embedding_path, allow_pickle=True)
    embedding_ids = [str(item) for item in npz_file["ids"].tolist()]
    embedding_sequences = [normalize_sequence(item) for item in npz_file["sequences"].tolist()]
    embedding_vectors = npz_file["embeddings"].astype(np.float32)

    if len(embedding_ids) != len(set(embedding_ids)):
        raise ValueError(f"Duplicate ids found in embedding file {embedding_path}")

    embedding_frame = pd.DataFrame(
        {
            "record_id": embedding_ids,
            "embedding_index": np.arange(len(embedding_ids)),
            "embedding_sequence": embedding_sequences,
        }
    )

    merged = rows.merge(embedding_frame, on="record_id", how="left", validate="one_to_one")
    if merged["embedding_index"].isna().any():
        missing_ids = merged.loc[merged["embedding_index"].isna(), "record_id"].tolist()
        raise ValueError(
            f"Some table rows are missing embeddings in {embedding_path}: {missing_ids[:10]}"
        )

    sequence_mismatch = merged["normalized_sequence"] != merged["embedding_sequence"]
    if sequence_mismatch.any():
        mismatch_rows = merged.loc[sequence_mismatch, ["record_id", "normalized_sequence"]]
        first_mismatch = mismatch_rows.iloc[0].to_dict()
        raise ValueError(
            "Sequence mismatch between table and embeddings for record "
            f"{first_mismatch['record_id']}: {first_mismatch['normalized_sequence']}"
        )

    embedding_indices = merged["embedding_index"].astype(int).to_numpy()
    aligned_embeddings = embedding_vectors[embedding_indices]

    features: list[np.ndarray] = []
    feature_names: list[str] = []
    if uses_embeddings(feature_set):
        embedding_dimension = aligned_embeddings.shape[1]
        features.append(aligned_embeddings)
        feature_names.extend([f"embedding_{index:04d}" for index in range(embedding_dimension)])

    if uses_nn_features(feature_set):
        nn_features, nn_names = encode_nn_features(
            merged["normalized_sequence"].tolist(),
            mode=nn_feature_mode,
        )
        features.append(nn_features)
        feature_names.extend(nn_names)

    if uses_availability_features(feature_set):
        availability_features, availability_names = encode_availability_features(
            merged[availability_column].tolist(),
            merged["normalized_sequence"].tolist(),
        )
        features.append(availability_features)
        feature_names.extend(availability_names)

    selected_transform = choose_target_transform(
        target_transform,
        merged["target_raw"].to_numpy(dtype=np.float32),
    )

    feature_matrix = np.concatenate(features, axis=1).astype(np.float32)
    raw_targets = merged["target_raw"].to_numpy(dtype=np.float32)
    transformed_targets = apply_target_transform(raw_targets, selected_transform)

    embedding_metadata = {
        "model_name": scalar_from_npz(npz_file, "model_name"),
        "pooling": scalar_from_npz(npz_file, "pooling"),
        "kmer_size": scalar_from_npz(npz_file, "kmer_size"),
        "model_kmer_size": scalar_from_npz(npz_file, "model_kmer_size"),
        "embedding_path": str(embedding_path),
    }

    return PreparedDataset(
        ids=merged["record_id"].tolist(),
        sequences=merged["normalized_sequence"].tolist(),
        features=feature_matrix,
        targets_raw=raw_targets,
        targets_model=transformed_targets,
        feature_names=feature_names,
        embedding_metadata=embedding_metadata,
        target_column=target_column,
        target_transform=selected_transform,
    )


def load_feature_matrix(
    table_path: Path,
    embedding_path: Path,
    id_column: str,
    sequence_column: str,
    availability_column: str,
    feature_set: str,
    nn_feature_mode: str,
) -> PreparedFeatureMatrix:
    table = pd.read_csv(table_path)
    required_columns = {id_column, sequence_column}
    if uses_availability_features(feature_set):
        required_columns.add(availability_column)
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {table_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(table.columns)}"
        )

    selected_columns = [id_column, sequence_column]
    if uses_availability_features(feature_set):
        selected_columns.append(availability_column)
    rows = table.loc[:, selected_columns].copy()
    rows["record_id"] = rows[id_column].astype(str)
    rows["normalized_sequence"] = rows[sequence_column].map(normalize_sequence)

    if rows["record_id"].duplicated().any():
        duplicate_ids = rows.loc[rows["record_id"].duplicated(), "record_id"].unique().tolist()
        raise ValueError(f"Duplicate ids found in {table_path}: {duplicate_ids[:10]}")

    npz_file = np.load(embedding_path, allow_pickle=True)
    embedding_ids = [str(item) for item in npz_file["ids"].tolist()]
    embedding_sequences = [normalize_sequence(item) for item in npz_file["sequences"].tolist()]
    embedding_vectors = npz_file["embeddings"].astype(np.float32)

    if len(embedding_ids) != len(set(embedding_ids)):
        raise ValueError(f"Duplicate ids found in embedding file {embedding_path}")

    embedding_frame = pd.DataFrame(
        {
            "record_id": embedding_ids,
            "embedding_index": np.arange(len(embedding_ids)),
            "embedding_sequence": embedding_sequences,
        }
    )

    merged = rows.merge(embedding_frame, on="record_id", how="left", validate="one_to_one")
    if merged["embedding_index"].isna().any():
        missing_ids = merged.loc[merged["embedding_index"].isna(), "record_id"].tolist()
        raise ValueError(
            f"Some table rows are missing embeddings in {embedding_path}: {missing_ids[:10]}"
        )

    sequence_mismatch = merged["normalized_sequence"] != merged["embedding_sequence"]
    if sequence_mismatch.any():
        mismatch_rows = merged.loc[sequence_mismatch, ["record_id", "normalized_sequence"]]
        first_mismatch = mismatch_rows.iloc[0].to_dict()
        raise ValueError(
            "Sequence mismatch between table and embeddings for record "
            f"{first_mismatch['record_id']}: {first_mismatch['normalized_sequence']}"
        )

    embedding_indices = merged["embedding_index"].astype(int).to_numpy()
    aligned_embeddings = embedding_vectors[embedding_indices]

    features: list[np.ndarray] = []
    feature_names: list[str] = []
    if uses_embeddings(feature_set):
        embedding_dimension = aligned_embeddings.shape[1]
        features.append(aligned_embeddings)
        feature_names.extend([f"embedding_{index:04d}" for index in range(embedding_dimension)])

    if uses_nn_features(feature_set):
        nn_features, nn_names = encode_nn_features(
            merged["normalized_sequence"].tolist(),
            mode=nn_feature_mode,
        )
        features.append(nn_features)
        feature_names.extend(nn_names)

    if uses_availability_features(feature_set):
        availability_features, availability_names = encode_availability_features(
            merged[availability_column].tolist(),
            merged["normalized_sequence"].tolist(),
        )
        features.append(availability_features)
        feature_names.extend(availability_names)

    embedding_metadata = {
        "model_name": scalar_from_npz(npz_file, "model_name"),
        "pooling": scalar_from_npz(npz_file, "pooling"),
        "kmer_size": scalar_from_npz(npz_file, "kmer_size"),
        "model_kmer_size": scalar_from_npz(npz_file, "model_kmer_size"),
        "embedding_path": str(embedding_path),
    }

    return PreparedFeatureMatrix(
        ids=merged["record_id"].tolist(),
        sequences=merged["normalized_sequence"].tolist(),
        features=np.concatenate(features, axis=1).astype(np.float32),
        feature_names=feature_names,
        embedding_metadata=embedding_metadata,
    )


def split_indices(
    sample_count: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)

    test_count = int(round(sample_count * test_fraction))
    val_count = int(round(sample_count * val_fraction))
    train_count = sample_count - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split configuration leaves no training examples.")

    train_indices = order[:train_count]
    val_indices = order[train_count : train_count + val_count]
    test_indices = order[train_count + val_count :]

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def standardize_features(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_indices].mean(axis=0)
    std = features[train_indices].std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (features - mean) / std
    return scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not cuda_is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device("mps")

    if cuda_is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_is_available() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CUDA initialization: The NVIDIA driver on your system is too old.*",
            category=UserWarning,
        )
        return torch.cuda.is_available()


def set_seed(seed: int, include_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    if sample_weights is None:
        sample_weights = np.ones(features.shape[0], dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(targets[indices]),
        torch.from_numpy(sample_weights[indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_weight = 0.0

    for batch_features, batch_targets, batch_weights in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        batch_weights = batch_weights.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(batch_features)
        per_item_loss = criterion(predictions, batch_targets)
        normalized_weight = batch_weights / batch_weights.sum().clamp_min(1e-8)
        loss = torch.sum(per_item_loss * normalized_weight)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_weight_sum = float(batch_weights.detach().cpu().sum())
        total_loss += float((per_item_loss * batch_weights).detach().cpu().sum())
        total_weight += batch_weight_sum

    return total_loss / max(total_weight, 1.0)


def estimate_domain_importance_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    clip_min: float,
    clip_max: float,
    classifier_c: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if clip_min <= 0.0:
        raise ValueError("--domain-weight-clip-min must be positive.")
    if clip_max < clip_min:
        raise ValueError("--domain-weight-clip-max must be >= --domain-weight-clip-min.")

    stacked_features = np.concatenate([source_features, target_features], axis=0).astype(np.float32)
    feature_mean = stacked_features.mean(axis=0)
    feature_std = stacked_features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    scaled_source = ((source_features - feature_mean) / feature_std).astype(np.float32)
    scaled_target = ((target_features - feature_mean) / feature_std).astype(np.float32)
    design_matrix = np.concatenate([scaled_source, scaled_target], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(scaled_source), dtype=np.int64),
            np.ones(len(scaled_target), dtype=np.int64),
        ]
    )
    domain_sample_weights = np.concatenate(
        [
            np.full(len(scaled_source), 0.5 / max(len(scaled_source), 1), dtype=np.float64),
            np.full(len(scaled_target), 0.5 / max(len(scaled_target), 1), dtype=np.float64),
        ]
    )

    classifier = LogisticRegression(
        C=classifier_c,
        max_iter=2000,
        solver="lbfgs",
        random_state=7,
    )
    classifier.fit(design_matrix, labels, sample_weight=domain_sample_weights)

    domain_probabilities = classifier.predict_proba(design_matrix)[:, 1]
    source_target_probabilities = classifier.predict_proba(scaled_source)[:, 1]
    source_odds = source_target_probabilities / np.clip(1.0 - source_target_probabilities, 1e-6, None)
    clipped_weights = np.clip(source_odds, clip_min, clip_max)
    normalized_weights = clipped_weights / np.mean(clipped_weights)

    summary = {
        "domain_auc": float(roc_auc_score(labels, domain_probabilities, sample_weight=domain_sample_weights)),
        "mean_raw_weight": float(np.mean(source_odds)),
        "median_raw_weight": float(np.median(source_odds)),
        "mean_weight": float(np.mean(normalized_weights)),
        "median_weight": float(np.median(normalized_weights)),
        "min_weight": float(np.min(normalized_weights)),
        "max_weight": float(np.max(normalized_weights)),
        "weight_p10": float(np.percentile(normalized_weights, 10)),
        "weight_p90": float(np.percentile(normalized_weights, 90)),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "classifier_c": float(classifier_c),
    }
    return normalized_weights.astype(np.float32), summary


def predict_array(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            batch = torch.from_numpy(features[start:stop]).to(device)
            predictions = model(batch).detach().cpu().numpy()
            outputs.append(predictions.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else math.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def collect_split_metrics(
    indices_by_split: dict[str, np.ndarray],
    raw_targets: np.ndarray,
    raw_predictions: np.ndarray,
    model_targets: np.ndarray,
    model_predictions: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, split_indices in indices_by_split.items():
        results[split_name] = {
            "raw": regression_metrics(raw_targets[split_indices], raw_predictions[split_indices]),
            "model_scale": regression_metrics(
                model_targets[split_indices],
                model_predictions[split_indices],
            ),
        }
    return results


def save_predictions_csv(
    path: Path,
    dataset: PreparedDataset,
    indices_by_split: dict[str, np.ndarray],
    raw_predictions: np.ndarray,
    model_predictions: np.ndarray,
) -> None:
    split_labels = np.empty(len(dataset.ids), dtype=object)
    for split_name, split_indices in indices_by_split.items():
        split_labels[split_indices] = split_name

    frame = pd.DataFrame(
        {
            "id": dataset.ids,
            "sequence": dataset.sequences,
            "split": split_labels,
            f"{dataset.target_column}_actual": dataset.targets_raw,
            f"{dataset.target_column}_predicted": raw_predictions,
            f"{dataset.target_column}_error": raw_predictions - dataset.targets_raw,
            f"{dataset.target_column}_model_actual": dataset.targets_model,
            f"{dataset.target_column}_model_predicted": model_predictions,
        }
    )
    frame.to_csv(path, index=False)


def train_regression_head(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed, include_cuda=args.device != "cpu")

    prepared = load_aligned_table(
        table_path=args.table_path,
        embedding_path=args.embedding_path,
        id_column=args.id_column,
        sequence_column=args.sequence_column,
        availability_column=args.availability_column,
        target_column=args.target_column,
        feature_set=args.feature_set,
        nn_feature_mode=args.nn_feature_mode,
        target_transform=args.target_transform,
    )

    indices_by_split = split_indices(
        sample_count=len(prepared.ids),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    scaled_features, feature_mean, feature_std = standardize_features(
        prepared.features,
        indices_by_split["train"],
    )
    train_sample_weights = np.ones(len(prepared.ids), dtype=np.float32)
    domain_adaptation_summary: dict[str, Any] | None = None
    if args.domain_adapt_path is not None:
        if args.domain_adapt_embedding_path is None:
            raise ValueError("--domain-adapt-embedding-path is required when --domain-adapt-path is set.")
        target_domain = load_feature_matrix(
            table_path=args.domain_adapt_path,
            embedding_path=args.domain_adapt_embedding_path,
            id_column=args.domain_adapt_id_column,
            sequence_column=args.domain_adapt_sequence_column,
            availability_column=args.domain_adapt_availability_column,
            feature_set=args.feature_set,
            nn_feature_mode=args.nn_feature_mode,
        )
        if target_domain.feature_names != prepared.feature_names:
            raise ValueError("Target-domain feature names do not match the source training features.")
        scaled_target_features = ((target_domain.features - feature_mean) / feature_std).astype(np.float32)
        train_weights, domain_summary = estimate_domain_importance_weights(
            source_features=scaled_features[indices_by_split["train"]],
            target_features=scaled_target_features,
            clip_min=args.domain_weight_clip_min,
            clip_max=args.domain_weight_clip_max,
            classifier_c=args.domain_classifier_c,
        )
        train_sample_weights[indices_by_split["train"]] = train_weights
        domain_adaptation_summary = {
            "enabled": True,
            "target_table_path": str(args.domain_adapt_path),
            "target_embedding_path": str(args.domain_adapt_embedding_path),
            "target_row_count": len(target_domain.ids),
            "source_train_row_count": int(len(indices_by_split["train"])),
            "target_embedding_metadata": target_domain.embedding_metadata,
            "weight_summary": domain_summary,
        }
    else:
        domain_adaptation_summary = {"enabled": False}

    device = resolve_device(args.device)
    model = RegressionHead(
        input_dim=scaled_features.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=getattr(args, "activation", "gelu"),
        normalization=getattr(args, "normalization", "layer"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss(reduction="none")

    train_loader = build_loader(
        scaled_features,
        prepared.targets_model,
        indices_by_split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        sample_weights=train_sample_weights,
    )
    val_loader = build_loader(
        scaled_features,
        prepared.targets_model,
        indices_by_split["val"],
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        sample_weights=None,
    )

    best_state: dict[str, Any] | None = None
    best_val_loss = math.inf
    best_epoch = 0
    patience_left = args.patience
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < (best_val_loss - args.min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} with best validation loss {best_val_loss:.6f}")
                break

    if best_state is None:
        raise RuntimeError("Training ended without capturing a valid model state.")

    model.load_state_dict(best_state)

    model_predictions = predict_array(
        model,
        scaled_features,
        batch_size=args.batch_size,
        device=device,
    )
    raw_predictions = invert_target_transform(model_predictions, prepared.target_transform)

    metrics = collect_split_metrics(
        indices_by_split=indices_by_split,
        raw_targets=prepared.targets_raw,
        raw_predictions=raw_predictions,
        model_targets=prepared.targets_model,
        model_predictions=model_predictions,
    )

    artifact_path = args.artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = artifact_path.with_name(f"{artifact_path.stem}_predictions.csv")
    summary_path = artifact_path.with_suffix(".json")

    artifact = {
        "model_state_dict": best_state,
        "model_config": {
            "input_dim": scaled_features.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "activation": getattr(args, "activation", "gelu"),
            "normalization": getattr(args, "normalization", "layer"),
        },
        "feature_config": {
            "feature_set": args.feature_set,
            "nn_feature_mode": args.nn_feature_mode,
            "feature_names": prepared.feature_names,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        "target_config": {
            "target_column": prepared.target_column,
            "target_transform": prepared.target_transform,
        },
        "embedding_metadata": prepared.embedding_metadata,
        "data_config": {
            "table_path": str(args.table_path),
            "embedding_path": str(args.embedding_path),
            "id_column": args.id_column,
            "sequence_column": args.sequence_column,
            "availability_column": args.availability_column,
            "row_count": len(prepared.ids),
        },
        "training_config": {
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "activation": getattr(args, "activation", "gelu"),
            "normalization": getattr(args, "normalization", "layer"),
            "seed": args.seed,
            "device": str(device),
        },
        "domain_adaptation": domain_adaptation_summary,
        "splits": {name: values.tolist() for name, values in indices_by_split.items()},
        "metrics": metrics,
        "history": history,
        "predictions_path": str(predictions_path),
    }
    torch.save(artifact, artifact_path)

    summary = {
        "artifact_path": str(artifact_path),
        "predictions_path": str(predictions_path),
        "target_column": prepared.target_column,
        "target_transform": prepared.target_transform,
        "feature_count": len(prepared.feature_names),
        "row_count": len(prepared.ids),
        "embedding_metadata": prepared.embedding_metadata,
        "training_config": artifact["training_config"],
        "domain_adaptation": domain_adaptation_summary,
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_predictions_csv(
        predictions_path,
        dataset=prepared,
        indices_by_split=indices_by_split,
        raw_predictions=raw_predictions,
        model_predictions=model_predictions,
    )

    runtime = get_runtime()
    runtime.state["regression_head"] = summary

    print(
        f"Saved artifact to {artifact_path} with test RMSE "
        f"{metrics['test']['raw']['rmse']:.3f} and test R^2 {metrics['test']['raw']['r2']:.3f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    train_regression_head(args)


if __name__ == "__main__":
    main()
