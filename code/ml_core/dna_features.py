from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .regression import PreparedFeatureMatrix


BASES = "ACGT"
DINUCLEOTIDES = [left + right for left in BASES for right in BASES]
DINUCLEOTIDE_TO_INDEX = {pair: index for index, pair in enumerate(DINUCLEOTIDES)}
VALID_BASES = set(BASES)


def normalize_dna_sequence(sequence: str) -> str:
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


def _scalar_from_npz(npz_file: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in npz_file:
        return None
    value = npz_file[key]
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0].item() if hasattr(value[0], "item") else value[0]
    return value.tolist() if isinstance(value, np.ndarray) else value


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
            positional_names.extend(f"nn_pos_{position + 1:02d}_{pair}" for pair in DINUCLEOTIDES)
    if mode in {"frequency", "both"}:
        frequency_features = np.zeros((len(sequences), len(DINUCLEOTIDES)), dtype=np.float32)
        frequency_names = [f"nn_freq_{pair}" for pair in DINUCLEOTIDES]
    for row_index, sequence in enumerate(sequences):
        if len(sequence) != pair_count + 1:
            raise ValueError("All sequences must share the same length when using positional NN features.")
        counts = np.zeros(len(DINUCLEOTIDES), dtype=np.float32)
        for position in range(pair_count):
            pair_index = DINUCLEOTIDE_TO_INDEX[sequence[position : position + 2]]
            counts[pair_index] += 1.0
            if positional_features is not None:
                positional_features[row_index, position * len(DINUCLEOTIDES) + pair_index] = 1.0
        if frequency_features is not None:
            frequency_features[row_index] = counts / float(pair_count)
    matrices = [matrix for matrix in (positional_features, frequency_features) if matrix is not None]
    return np.concatenate(matrices, axis=1), positional_names + frequency_names


def _parse_numeric_vector(raw_value: Any) -> np.ndarray:
    if isinstance(raw_value, np.ndarray):
        vector = raw_value.astype(np.float32)
    elif isinstance(raw_value, (list, tuple)):
        vector = np.asarray(raw_value, dtype=np.float32)
    elif isinstance(raw_value, str):
        try:
            vector = np.asarray(ast.literal_eval(raw_value), dtype=np.float32)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse vector literal: {raw_value!r}") from exc
    else:
        raise ValueError(f"Unsupported vector value type: {type(raw_value)!r}")
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D numeric vector, got shape {vector.shape}")
    return vector


def encode_availability_features(raw_values: list[Any], sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    if not raw_values:
        raise ValueError("No availability values available for feature generation.")
    matrix: list[np.ndarray] = []
    expected_length = len(sequences[0])
    for index, (raw_value, sequence) in enumerate(zip(raw_values, sequences, strict=True)):
        vector = _parse_numeric_vector(raw_value)
        if len(sequence) != expected_length:
            raise ValueError("All sequences must share the same length for availability features.")
        if vector.shape[0] != len(sequence):
            raise ValueError(
                "Availability vector length does not match sequence length for row "
                f"{index}: {vector.shape[0]} vs {len(sequence)}"
            )
        matrix.append(vector)
    names = [f"availability_{position + 1:02d}" for position in range(expected_length)]
    return np.vstack(matrix).astype(np.float32), names


def load_dna_feature_matrix(
    *,
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
    rows["normalized_sequence"] = rows[sequence_column].map(normalize_dna_sequence)
    if rows["record_id"].duplicated().any():
        duplicate_ids = rows.loc[rows["record_id"].duplicated(), "record_id"].unique().tolist()
        raise ValueError(f"Duplicate ids found in {table_path}: {duplicate_ids[:10]}")

    npz_file = np.load(embedding_path, allow_pickle=True)
    embedding_ids = [str(item) for item in npz_file["ids"].tolist()]
    embedding_sequences = [normalize_dna_sequence(item) for item in npz_file["sequences"].tolist()]
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
        raise ValueError(f"Some table rows are missing embeddings in {embedding_path}: {missing_ids[:10]}")
    sequence_mismatch = merged["normalized_sequence"] != merged["embedding_sequence"]
    if sequence_mismatch.any():
        first_mismatch = merged.loc[sequence_mismatch, ["record_id", "normalized_sequence"]].iloc[0]
        raise ValueError(
            "Sequence mismatch between table and embeddings for record "
            f"{first_mismatch['record_id']}: {first_mismatch['normalized_sequence']}"
        )

    embedding_indices = merged["embedding_index"].astype(int).to_numpy()
    aligned_embeddings = embedding_vectors[embedding_indices]
    features: list[np.ndarray] = []
    feature_names: list[str] = []
    if uses_embeddings(feature_set):
        features.append(aligned_embeddings)
        feature_names.extend(f"embedding_{index:04d}" for index in range(aligned_embeddings.shape[1]))
    if uses_nn_features(feature_set):
        matrix, names = encode_nn_features(merged["normalized_sequence"].tolist(), nn_feature_mode)
        features.append(matrix)
        feature_names.extend(names)
    if uses_availability_features(feature_set):
        matrix, names = encode_availability_features(
            merged[availability_column].tolist(),
            merged["normalized_sequence"].tolist(),
        )
        features.append(matrix)
        feature_names.extend(names)
    metadata = {
        "model_name": _scalar_from_npz(npz_file, "model_name"),
        "pooling": _scalar_from_npz(npz_file, "pooling"),
        "kmer_size": _scalar_from_npz(npz_file, "kmer_size"),
        "model_kmer_size": _scalar_from_npz(npz_file, "model_kmer_size"),
        "embedding_path": str(embedding_path),
    }
    return PreparedFeatureMatrix(
        ids=merged["record_id"].tolist(),
        sequences=merged["normalized_sequence"].tolist(),
        features=np.concatenate(features, axis=1).astype(np.float32),
        feature_names=feature_names,
        embedding_metadata=metadata,
    )
