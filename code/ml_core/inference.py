from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from .dna_features import load_dna_feature_matrix, normalize_dna_sequence
from .regression import (
    PreparedFeatureMatrix,
    RegressionHead,
    invert_target_transform,
    predict_array,
)


class RepresentationValidator(Protocol):
    name: str

    def validate(self, value: Any) -> str:
        """Return the normalized record representation or raise ValueError."""


class DnaSequenceValidator:
    name = "dna"

    def validate(self, value: Any) -> str:
        return normalize_dna_sequence(str(value))


class NoOpRepresentationValidator:
    name = "none"

    def validate(self, value: Any) -> str:
        return str(value)


class DatasetSchema:
    id_column: str
    sequence_column: str
    availability_column: str
    validator: RepresentationValidator | None
    feature_set_builder: FeatureSetBuilder | None

    def __init__(
        self,
        id_column: str = "id",
        sequence_column: str = "representation",
        availability_column: str = "availability",
        validator: RepresentationValidator | None = None,
        feature_set_builder: "FeatureSetBuilder | None" = None,
    ) -> None:
        self.id_column = id_column
        self.sequence_column = sequence_column
        self.availability_column = availability_column
        self.validator = validator
        self.feature_set_builder = feature_set_builder

    @classmethod
    def from_artifact_config(
        cls,
        config: dict[str, Any],
        *,
        id_column: str | None = None,
        sequence_column: str | None = None,
        availability_column: str | None = None,
        validator: RepresentationValidator | None = None,
        feature_set_builder: "FeatureSetBuilder | None" = None,
    ) -> "DatasetSchema":
        data_config = config.get("data_config", {})
        return cls(
            id_column=id_column or data_config.get("id_column", "id"),
            sequence_column=sequence_column or data_config.get("sequence_column", "representation"),
            availability_column=availability_column
            or data_config.get("availability_column", "availability"),
            validator=validator,
            feature_set_builder=feature_set_builder,
        )

class FeatureSetBuilder(Protocol):
    def prepare(
        self,
        *,
        recipe: "FeatureRecipe",
        table_path: Path,
        embedding_path: Path,
        schema: DatasetSchema,
    ) -> "PreparedFeatureMatrix":
        """Prepare an inference feature matrix for this feature-set family."""


class DnaFeatureSetBuilder:
    def prepare(
        self,
        *,
        recipe: "FeatureRecipe",
        table_path: Path,
        embedding_path: Path,
        schema: DatasetSchema,
    ) -> "PreparedFeatureMatrix":
        return load_dna_feature_matrix(
            table_path=table_path,
            embedding_path=embedding_path,
            id_column=schema.id_column,
            sequence_column=schema.sequence_column,
            availability_column=schema.availability_column,
            feature_set=recipe.feature_set,
            nn_feature_mode=recipe.nn_feature_mode,
        )


class GenericEmbeddingFeatureSetBuilder:
    def prepare(
        self,
        *,
        recipe: "FeatureRecipe",
        table_path: Path,
        embedding_path: Path,
        schema: DatasetSchema,
    ) -> "PreparedFeatureMatrix":
        return load_generic_embedding_matrix(
            table_path=table_path,
            embedding_path=embedding_path,
            id_column=schema.id_column,
            sequence_column=schema.sequence_column,
            feature_set=recipe.feature_set,
            validator=schema.validator,
        )


def resolve_feature_set_builder(recipe: "FeatureRecipe", schema: DatasetSchema) -> FeatureSetBuilder:
    return schema.feature_set_builder or GenericEmbeddingFeatureSetBuilder()


@dataclass
class FeatureRecipe:
    feature_set: str
    nn_feature_mode: str
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @classmethod
    def from_artifact_config(cls, config: dict[str, Any]) -> "FeatureRecipe":
        feature_config = config["feature_config"]
        return cls(
            feature_set=feature_config["feature_set"],
            nn_feature_mode=feature_config["nn_feature_mode"],
            feature_names=list(feature_config["feature_names"]),
            feature_mean=np.asarray(feature_config["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(feature_config["feature_std"], dtype=np.float32),
        )

    def prepare_features(
        self,
        *,
        table_path: Path,
        embedding_path: Path,
        schema: DatasetSchema,
    ) -> "PreparedFeatureMatrix":
        prepared = resolve_feature_set_builder(self, schema).prepare(
            recipe=self,
            table_path=table_path,
            embedding_path=embedding_path,
            schema=schema,
        )

        if prepared.feature_names != self.feature_names:
            raise ValueError("Prepared feature names do not match the saved model artifact.")
        return prepared

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.feature_mean) / self.feature_std).astype(np.float32)


@dataclass
class ModelArtifact:
    path: Path
    checkpoint: dict[str, Any]
    feature_recipe: FeatureRecipe
    target_column: str
    target_transform: str

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        artifact_path = Path(path)
        checkpoint = load_checkpoint(str(artifact_path))
        target_config = checkpoint["target_config"]
        return cls(
            path=artifact_path,
            checkpoint=checkpoint,
            feature_recipe=FeatureRecipe.from_artifact_config(checkpoint),
            target_column=target_config.get("target_column", "prediction"),
            target_transform=target_config["target_transform"],
        )

    def build_model(self) -> RegressionHead:
        model_config = self.checkpoint["model_config"]
        model = RegressionHead(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            dropout=model_config["dropout"],
            activation=model_config.get("activation", "gelu"),
            normalization=model_config.get("normalization", "layer"),
        )
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        return model

    def invert_target(self, predictions: np.ndarray) -> np.ndarray:
        return invert_target_transform(predictions, self.target_transform)


@dataclass
class ArtifactPredictionResult:
    table_path: Path
    embedding_path: Path
    artifact_path: Path
    schema: DatasetSchema
    feature_recipe: FeatureRecipe
    target_column: str
    frame: pd.DataFrame


@lru_cache(maxsize=8)
def load_checkpoint(artifact_path: str) -> dict[str, Any]:
    return torch.load(artifact_path, map_location="cpu", weights_only=False)


def _scalar_from_npz(npz_file: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in npz_file:
        return None
    value = npz_file[key]
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0].item() if hasattr(value[0], "item") else value[0]
    return value.tolist() if isinstance(value, np.ndarray) else value


def load_generic_embedding_matrix(
    *,
    table_path: Path,
    embedding_path: Path,
    id_column: str,
    sequence_column: str,
    feature_set: str,
    validator: RepresentationValidator | None = None,
) -> PreparedFeatureMatrix:
    if feature_set != "embeddings":
        raise ValueError(
            "Generic non-DNA schemas currently support feature_set='embeddings' only. "
            "Nearest-neighbor and availability features require DNA sequence semantics."
        )

    table = pd.read_csv(table_path)
    required_columns = {id_column, sequence_column}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {table_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(table.columns)}"
        )

    representation_validator = validator or NoOpRepresentationValidator()
    rows = table.loc[:, [id_column, sequence_column]].copy()
    rows["record_id"] = rows[id_column].astype(str)
    rows["record_sequence"] = rows[sequence_column].map(representation_validator.validate)

    if rows["record_id"].duplicated().any():
        duplicate_ids = rows.loc[rows["record_id"].duplicated(), "record_id"].unique().tolist()
        raise ValueError(f"Duplicate ids found in {table_path}: {duplicate_ids[:10]}")

    npz_file = np.load(embedding_path, allow_pickle=True)
    embedding_ids = [str(item) for item in npz_file["ids"].tolist()]
    embedding_sequences = [
        representation_validator.validate(item)
        for item in npz_file["sequences"].tolist()
    ]
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

    sequence_mismatch = merged["record_sequence"] != merged["embedding_sequence"]
    if sequence_mismatch.any():
        first_mismatch = merged.loc[sequence_mismatch, ["record_id", "record_sequence"]].iloc[0]
        raise ValueError(
            "Record representation mismatch between table and embeddings for record "
            f"{first_mismatch['record_id']}: {first_mismatch['record_sequence']}"
        )

    embedding_indices = merged["embedding_index"].astype(int).to_numpy()
    aligned_embeddings = embedding_vectors[embedding_indices]
    feature_names = [f"embedding_{index:04d}" for index in range(aligned_embeddings.shape[1])]
    embedding_metadata = {
        "model_name": _scalar_from_npz(npz_file, "model_name"),
        "pooling": _scalar_from_npz(npz_file, "pooling"),
        "kmer_size": _scalar_from_npz(npz_file, "kmer_size"),
        "model_kmer_size": _scalar_from_npz(npz_file, "model_kmer_size"),
        "embedding_path": str(embedding_path),
    }

    return PreparedFeatureMatrix(
        ids=merged["record_id"].tolist(),
        sequences=merged["record_sequence"].tolist(),
        features=aligned_embeddings,
        feature_names=feature_names,
        embedding_metadata=embedding_metadata,
    )


class PredictionService:
    def __init__(self, artifact: ModelArtifact, schema: DatasetSchema | None = None) -> None:
        self.artifact = artifact
        self.schema = schema or DatasetSchema.from_artifact_config(artifact.checkpoint)

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        schema: DatasetSchema | None = None,
    ) -> "PredictionService":
        return cls(ModelArtifact.load(artifact_path), schema=schema)

    def predict_table(
        self,
        *,
        table_path: str | Path,
        embedding_path: str | Path,
        batch_size: int = 256,
    ) -> ArtifactPredictionResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        table = Path(table_path)
        embeddings = Path(embedding_path)
        prepared = self.artifact.feature_recipe.prepare_features(
            table_path=table,
            embedding_path=embeddings,
            schema=self.schema,
        )
        scaled_features = self.artifact.feature_recipe.transform(prepared.features)
        model = self.artifact.build_model()

        model_predictions = predict_array(
            model=model,
            features=scaled_features,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )
        raw_predictions = self.artifact.invert_target(model_predictions)
        ranks = pd.Series(raw_predictions).rank(method="min", ascending=False).astype(int).to_numpy()

        frame = pd.DataFrame(
            {
                self.schema.id_column: prepared.ids,
                self.schema.sequence_column: prepared.sequences,
                f"predicted_{self.artifact.target_column}": raw_predictions,
                "predicted_rank": ranks,
            }
        ).sort_values("predicted_rank")

        return ArtifactPredictionResult(
            table_path=table,
            embedding_path=embeddings,
            artifact_path=self.artifact.path,
            schema=self.schema,
            feature_recipe=self.artifact.feature_recipe,
            target_column=self.artifact.target_column,
            frame=frame,
        )


def predict_from_artifact(
    *,
    table_path: str | Path,
    embedding_path: str | Path,
    artifact_path: str | Path,
    id_column: str | None = None,
    sequence_column: str | None = None,
    availability_column: str | None = None,
    validator: RepresentationValidator | None = None,
    feature_set_builder: FeatureSetBuilder | None = None,
    batch_size: int = 256,
) -> ArtifactPredictionResult:
    artifact = ModelArtifact.load(artifact_path)
    schema = DatasetSchema.from_artifact_config(
        artifact.checkpoint,
        id_column=id_column,
        sequence_column=sequence_column,
        availability_column=availability_column,
        validator=validator,
        feature_set_builder=feature_set_builder,
    )
    return PredictionService(artifact, schema=schema).predict_table(
        table_path=table_path,
        embedding_path=embedding_path,
        batch_size=batch_size,
    )
