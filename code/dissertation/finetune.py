from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dissertation.generate_dna_bert_embeddings import (  # noqa: E402
    pool_hidden_states,
    resolve_model_kmer_size,
    to_kmers,
    validate_model_compatibility,
)
from runtime.runtime import get_runtime  # noqa: E402
from dissertation import regression_training as rg_module  # noqa: E402


BASES = "ACGT"
DINUCLEOTIDES = [left + right for left in BASES for right in BASES]
DINUCLEOTIDE_TO_INDEX = {pair: index for index, pair in enumerate(DINUCLEOTIDES)}
VALID_BASES = set(BASES)
AVAILABILITY_COLUMN = "Nucleotide Availability"


@dataclass
class PreparedFineTuneData:
    ids: list[str]
    sequences: list[str]
    texts: list[str]
    nn_features: np.ndarray | None
    nn_feature_names: list[str]
    availability_features: np.ndarray | None
    availability_feature_names: list[str]
    targets_raw: np.ndarray
    targets_model: np.ndarray
    target_column: str
    target_transform: str


class SequenceRegressionDataset(Dataset):
    def __init__(self, prepared: PreparedFineTuneData) -> None:
        self.prepared = prepared

    def __len__(self) -> int:
        return len(self.prepared.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": self.prepared.ids[index],
            "sequence": self.prepared.sequences[index],
            "text": self.prepared.texts[index],
            "target": float(self.prepared.targets_model[index]),
        }
        if self.prepared.nn_features is not None:
            item["nn_features"] = self.prepared.nn_features[index]
        if self.prepared.availability_features is not None:
            item["availability_features"] = self.prepared.availability_features[index]
        return item


class TokenizingCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            **encoded,
            "targets": torch.tensor(
                [item["target"] for item in items],
                dtype=torch.float32,
            ),
            "ids": [item["id"] for item in items],
            "sequences": [item["sequence"] for item in items],
        }
        if "nn_features" in items[0]:
            batch["nn_features"] = torch.tensor(
                np.stack([item["nn_features"] for item in items]),
                dtype=torch.float32,
            )
        if "availability_features" in items[0]:
            batch["availability_features"] = torch.tensor(
                np.stack([item["availability_features"] for item in items]),
                dtype=torch.float32,
            )
        return batch


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class DNABertFineTuner(nn.Module):
    def __init__(
        self,
        encoder,
        pooling: str,
        head_hidden_dims: list[int],
        dropout: float,
        nn_feature_dim: int,
        nn_projection_dim: int,
        availability_feature_dim: int,
        availability_projection_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.hidden_size = int(encoder.config.hidden_size)

        combined_dim = self.hidden_size
        self.nn_adapter: nn.Module | None = None
        if nn_feature_dim > 0:
            self.nn_adapter = nn.Sequential(
                nn.LayerNorm(nn_feature_dim),
                nn.Linear(nn_feature_dim, nn_projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            combined_dim += nn_projection_dim

        self.availability_adapter: nn.Module | None = None
        if availability_feature_dim > 0:
            self.availability_adapter = nn.Sequential(
                nn.LayerNorm(availability_feature_dim),
                nn.Linear(availability_feature_dim, availability_projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            combined_dim += availability_projection_dim

        self.regression_head = RegressionHead(
            input_dim=combined_dim,
            hidden_dims=head_hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        nn_features: torch.Tensor | None = None,
        availability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        pooled = pool_hidden_states(
            outputs.last_hidden_state,
            attention_mask,
            self.pooling,
            torch,
        )

        if self.nn_adapter is not None:
            if nn_features is None:
                raise ValueError("NN features were enabled but no nn_features tensor was provided.")
            pooled = torch.cat([pooled, self.nn_adapter(nn_features)], dim=-1)

        if self.availability_adapter is not None:
            if availability_features is None:
                raise ValueError(
                    "Availability features were enabled but no availability_features tensor was provided."
                )
            pooled = torch.cat([pooled, self.availability_adapter(availability_features)], dim=-1)

        return self.regression_head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune DNA-BERT end-to-end for sequence regression. "
            "This trains the encoder and regression head together instead of "
            "using frozen precomputed embeddings."
        )
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("data/extracted/clean/table_1.csv"),
        help="CSV containing sequences and regression targets.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("training/artifacts/finetuned_dnabert/k1_finetuned_dnabert.pt"),
        help="Where to save the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--target-column",
        default="k1",
        help="Target column to predict. Use k1 or k2.",
    )
    parser.add_argument(
        "--id-column",
        default="No.",
        help="Identifier column used to align rows.",
    )
    parser.add_argument(
        "--sequence-column",
        default="Sequence",
        help="Sequence column used for tokenization and NN feature generation.",
    )
    parser.add_argument(
        "--availability-column",
        default=AVAILABILITY_COLUMN,
        help="Column containing per-position nucleotide availability scores.",
    )
    parser.add_argument(
        "--model-name",
        default="zhihan1996/DNA_bert_6",
        help="Base DNA-BERT checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--model-kmer-size",
        type=int,
        help="Expected k-mer size for the model. Inferred automatically when possible.",
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=6,
        help="K-mer size fed into the tokenizer. Must match the model checkpoint.",
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            "bert+nn+availability",
            "bert+availability",
            "bert+nn",
            "bert",
        ),
        default="bert+nn",
        help="Which auxiliary feature families to concatenate to the pooled BERT representation.",
    )
    parser.add_argument(
        "--nn-feature-mode",
        choices=("positional", "frequency", "both"),
        default="both",
        help="How to encode nearest-neighbor dinucleotide features.",
    )
    parser.add_argument(
        "--nn-projection-dim",
        type=int,
        default=128,
        help="Projection size used before concatenating NN features to the BERT output.",
    )
    parser.add_argument(
        "--availability-projection-dim",
        type=int,
        default=64,
        help="Projection size used before concatenating availability features to the BERT output.",
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean", "max"),
        default="mean",
        help="How to pool token embeddings into one sequence representation.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("auto", "none", "log10", "ln"),
        default="auto",
        help="Target transform applied before fitting.",
    )
    parser.add_argument(
        "--head-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 128],
        help="Hidden layer sizes for the regression head after pooling.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
        help="Dropout used in the NN adapter and regression head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate used for trainable DNA-BERT parameters.",
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate used for the regression head and NN adapter.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=2,
        help=(
            "How many encoder layers to unfreeze. "
            "Use 0 for head-only training or -1 for full fine-tuning."
        ),
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Also unfreeze the embedding layer during fine-tuning.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Max gradient norm. Set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "linear", "cosine"),
        default="cosine",
        help="Learning-rate scheduler used during fine-tuning.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps reserved for LR warmup.",
    )
    parser.add_argument(
        "--loss",
        choices=("mse", "huber"),
        default="mse",
        help="Regression loss used during training.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.25,
        help="Delta parameter for Huber loss when --loss huber is selected.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience measured in epochs.",
    )
    parser.add_argument(
        "--head-only-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Optional number of initial epochs to train only the regression head "
            "before unfreezing encoder layers."
        ),
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
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length before truncation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap on the number of rows to use, useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Run training without writing the large checkpoint file.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the tokenizer and model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer and model.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="How often to print epoch progress.",
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


def uses_nn_features(feature_set: str) -> bool:
    return "nn" in feature_set_parts(feature_set)


def uses_availability_features(feature_set: str) -> bool:
    return "availability" in feature_set_parts(feature_set)


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
            (len(sequences), pair_count * len(DINUCLEOTIDES)),
            dtype=np.float32,
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

    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(
            "Current split settings produced an empty train/val/test partition. "
            "Adjust --val-fraction, --test-fraction, or --max-samples."
        )

    return {
        "train": order[:train_count],
        "val": order[train_count : train_count + val_count],
        "test": order[train_count + val_count :],
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


def cuda_is_available() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CUDA initialization: The NVIDIA driver on your system is too old.*",
            category=UserWarning,
        )
        return torch.cuda.is_available()


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


def set_seed(seed: int, include_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def load_prepared_data(args: argparse.Namespace) -> PreparedFineTuneData:
    table = pd.read_csv(args.table_path)
    required_columns = {args.id_column, args.sequence_column, args.target_column}
    if uses_availability_features(args.feature_set):
        required_columns.add(args.availability_column)
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {args.table_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(table.columns)}"
        )

    selected_columns = [args.id_column, args.sequence_column, args.target_column]
    if uses_availability_features(args.feature_set):
        selected_columns.append(args.availability_column)
    rows = table.loc[
        table[args.target_column].notna(),
        selected_columns,
    ].copy()
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be greater than zero.")
        rows = rows.head(args.max_samples).copy()

    rows["record_id"] = rows[args.id_column].astype(str)
    rows["normalized_sequence"] = rows[args.sequence_column].map(normalize_sequence)
    rows["target_raw"] = rows[args.target_column].astype(np.float32)

    if rows["record_id"].duplicated().any():
        duplicate_ids = rows.loc[rows["record_id"].duplicated(), "record_id"].unique().tolist()
        raise ValueError(f"Duplicate ids found in {args.table_path}: {duplicate_ids[:10]}")

    model_kmer_size, model_kmer_source = resolve_model_kmer_size(
        args.model_name,
        args.model_kmer_size,
    )
    validate_model_compatibility(args.kmer_size, model_kmer_size, model_kmer_source)

    selected_transform = choose_target_transform(
        args.target_transform,
        rows["target_raw"].to_numpy(dtype=np.float32),
    )
    texts = [
        to_kmers(sequence, args.kmer_size)
        for sequence in rows["normalized_sequence"].tolist()
    ]

    nn_features = None
    nn_feature_names: list[str] = []
    if uses_nn_features(args.feature_set):
        nn_features, nn_feature_names = encode_nn_features(
            rows["normalized_sequence"].tolist(),
            mode=args.nn_feature_mode,
        )

    availability_features = None
    availability_feature_names: list[str] = []
    if uses_availability_features(args.feature_set):
        availability_features, availability_feature_names = rg_module.encode_availability_features(
            rows[args.availability_column].tolist(),
            rows["normalized_sequence"].tolist(),
        )

    return PreparedFineTuneData(
        ids=rows["record_id"].tolist(),
        sequences=rows["normalized_sequence"].tolist(),
        texts=texts,
        nn_features=nn_features,
        nn_feature_names=nn_feature_names,
        availability_features=availability_features,
        availability_feature_names=availability_feature_names,
        targets_raw=rows["target_raw"].to_numpy(dtype=np.float32),
        targets_model=apply_target_transform(
            rows["target_raw"].to_numpy(dtype=np.float32),
            selected_transform,
        ),
        target_column=args.target_column,
        target_transform=selected_transform,
    )


def configure_tokenizer(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        fallback_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token
        if fallback_token is None:
            raise RuntimeError("Tokenizer does not define a pad token or usable fallback token.")
        tokenizer.pad_token = fallback_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id


def configure_encoder_tuning(
    encoder,
    unfreeze_last_n_layers: int,
    train_embeddings: bool,
) -> dict[str, Any]:
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoder_stack = getattr(getattr(encoder, "encoder", None), "layer", None)
    if encoder_stack is None:
        raise RuntimeError(
            "Could not locate encoder layers on this checkpoint. "
            "This script currently expects a BERT-style model with encoder.layer."
        )

    total_layers = len(encoder_stack)
    if unfreeze_last_n_layers < -1:
        raise ValueError("--unfreeze-last-n-layers must be -1 or greater.")
    if unfreeze_last_n_layers > total_layers:
        raise ValueError(
            f"--unfreeze-last-n-layers {unfreeze_last_n_layers} exceeds model depth {total_layers}."
        )

    if unfreeze_last_n_layers == -1:
        target_layers = list(encoder_stack)
    elif unfreeze_last_n_layers == 0:
        target_layers = []
    else:
        target_layers = list(encoder_stack[-unfreeze_last_n_layers:])

    for layer in target_layers:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    if train_embeddings:
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Requested --train-embeddings, but this model has no embeddings module.")
        for parameter in embeddings.parameters():
            parameter.requires_grad = True

    return {
        "total_encoder_layers": total_layers,
        "train_embeddings": train_embeddings,
        "unfreeze_last_n_layers": unfreeze_last_n_layers,
        "trainable_encoder_parameter_count": sum(
            parameter.numel()
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ),
    }


def build_loader(
    dataset: SequenceRegressionDataset,
    indices: np.ndarray,
    collator: TokenizingCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collator,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    device_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            device_batch[key] = value.to(device)
        else:
            device_batch[key] = value
    return device_batch


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


def create_optimizer(
    model: DNABertFineTuner,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder_params = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    parameter_groups: list[dict[str, Any]] = []
    if encoder_params:
        parameter_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if head_params:
        parameter_groups.append(
            {
                "params": head_params,
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were configured.")
    return torch.optim.AdamW(parameter_groups)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    warmup_ratio: float,
    total_training_steps: int,
):
    if scheduler_name == "none":
        return None, 0
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1.")
    if total_training_steps <= 0:
        raise ValueError("Training must have at least one optimization step.")

    warmup_steps = int(round(total_training_steps * warmup_ratio))
    if scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler, warmup_steps
    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler, warmup_steps
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def create_criterion(loss_name: str, huber_delta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def run_epoch(
    model: DNABertFineTuner,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
    gradient_clip: float,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch["token_type_ids"]
        nn_features = batch.get("nn_features")
        availability_features = batch.get("availability_features")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(
            nn_features=nn_features,
            availability_features=availability_features,
            **model_inputs,
        )
        loss = criterion(predictions, batch["targets"])

        if is_training:
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = batch["targets"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def predict_dataset(
    model: DNABertFineTuner,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str], list[str]]:
    model.eval()
    predictions: list[np.ndarray] = []
    ids: list[str] = []
    sequences: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"]
            nn_features = batch.get("nn_features")
            availability_features = batch.get("availability_features")
            batch_predictions = model(
                nn_features=nn_features,
                availability_features=availability_features,
                **model_inputs,
            )
            predictions.append(batch_predictions.detach().cpu().numpy().astype(np.float32))
            ids.extend(batch["ids"])
            sequences.extend(batch["sequences"])

    return np.concatenate(predictions, axis=0), ids, sequences


def save_predictions_csv(
    path: Path,
    prepared: PreparedFineTuneData,
    indices_by_split: dict[str, np.ndarray],
    raw_predictions: np.ndarray,
    model_predictions: np.ndarray,
) -> None:
    split_labels = np.empty(len(prepared.ids), dtype=object)
    for split_name, split_indices in indices_by_split.items():
        split_labels[split_indices] = split_name

    frame = pd.DataFrame(
        {
            "id": prepared.ids,
            "sequence": prepared.sequences,
            "split": split_labels,
            f"{prepared.target_column}_actual": prepared.targets_raw,
            f"{prepared.target_column}_predicted": raw_predictions,
            f"{prepared.target_column}_error": raw_predictions - prepared.targets_raw,
            f"{prepared.target_column}_model_actual": prepared.targets_model,
            f"{prepared.target_column}_model_predicted": model_predictions,
        }
    )
    frame.to_csv(path, index=False)


def train_finetuned_regressor(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed, include_cuda=args.device != "cpu")

    prepared = load_prepared_data(args)
    indices_by_split = split_indices(
        sample_count=len(prepared.ids),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    nn_features = prepared.nn_features
    nn_feature_mean = None
    nn_feature_std = None
    if nn_features is not None:
        nn_features, nn_feature_mean, nn_feature_std = standardize_features(
            nn_features,
            indices_by_split["train"],
        )
    availability_features = prepared.availability_features
    availability_feature_mean = None
    availability_feature_std = None
    if availability_features is not None:
        availability_features, availability_feature_mean, availability_feature_std = standardize_features(
            availability_features,
            indices_by_split["train"],
        )

    prepared = PreparedFineTuneData(
        ids=prepared.ids,
        sequences=prepared.sequences,
        texts=prepared.texts,
        nn_features=nn_features,
        nn_feature_names=prepared.nn_feature_names,
        availability_features=availability_features,
        availability_feature_names=prepared.availability_feature_names,
        targets_raw=prepared.targets_raw,
        targets_model=prepared.targets_model,
        target_column=prepared.target_column,
        target_transform=prepared.target_transform,
    )

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    encoder = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    configure_tokenizer(tokenizer, encoder)

    model = DNABertFineTuner(
        encoder=encoder,
        pooling=args.pooling,
        head_hidden_dims=args.head_hidden_dims,
        dropout=args.dropout,
        nn_feature_dim=0 if prepared.nn_features is None else prepared.nn_features.shape[1],
        nn_projection_dim=args.nn_projection_dim,
        availability_feature_dim=(
            0 if prepared.availability_features is None else prepared.availability_features.shape[1]
        ),
        availability_projection_dim=args.availability_projection_dim,
    ).to(device)

    dataset = SequenceRegressionDataset(prepared)
    collator = TokenizingCollator(tokenizer, max_length=args.max_length)
    train_loader = build_loader(
        dataset=dataset,
        indices=indices_by_split["train"],
        collator=collator,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = build_loader(
        dataset=dataset,
        indices=indices_by_split["val"],
        collator=collator,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    all_loader = build_loader(
        dataset=dataset,
        indices=np.arange(len(prepared.ids)),
        collator=collator,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    criterion = create_criterion(args.loss, args.huber_delta)

    if args.head_only_warmup_epochs < 0:
        raise ValueError("--head-only-warmup-epochs must be zero or greater.")

    phase_plans: list[dict[str, Any]] = []
    if args.head_only_warmup_epochs > 0 and (
        args.unfreeze_last_n_layers != 0 or args.train_embeddings
    ):
        if args.head_only_warmup_epochs >= args.epochs:
            raise ValueError(
                "--head-only-warmup-epochs must be smaller than --epochs when "
                "encoder fine-tuning is enabled."
            )
        phase_plans.append(
            {
                "name": "head_warmup",
                "epochs": args.head_only_warmup_epochs,
                "unfreeze_last_n_layers": 0,
                "train_embeddings": False,
            }
        )
        phase_plans.append(
            {
                "name": "finetune",
                "epochs": args.epochs - args.head_only_warmup_epochs,
                "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                "train_embeddings": args.train_embeddings,
            }
        )
    else:
        phase_plans.append(
            {
                "name": "finetune",
                "epochs": args.epochs,
                "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                "train_embeddings": args.train_embeddings,
            }
        )

    best_state: dict[str, Any] | None = None
    best_val_loss = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    final_tuning_summary: dict[str, Any] | None = None
    total_warmup_steps = 0
    global_epoch = 0
    stop_training = False

    for phase in phase_plans:
        tuning_summary = configure_encoder_tuning(
            encoder=model.encoder,
            unfreeze_last_n_layers=phase["unfreeze_last_n_layers"],
            train_embeddings=phase["train_embeddings"],
        )
        final_tuning_summary = tuning_summary
        optimizer = create_optimizer(
            model=model,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
        )
        total_training_steps = len(train_loader) * phase["epochs"]
        scheduler, warmup_steps = create_scheduler(
            optimizer=optimizer,
            scheduler_name=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            total_training_steps=total_training_steps,
        )
        total_warmup_steps += warmup_steps

        patience_left = args.patience
        phase_summary: dict[str, Any] = {
            "name": phase["name"],
            "epochs_requested": phase["epochs"],
            "epochs_trained": 0,
            "warmup_steps": warmup_steps,
            "tuning_config": tuning_summary,
        }

        for phase_epoch in range(1, phase["epochs"] + 1):
            global_epoch += 1
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                gradient_clip=args.gradient_clip,
            )
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                device=device,
                gradient_clip=0.0,
            )
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": phase["name"],
                    "phase_epoch": phase_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            phase_summary["epochs_trained"] = phase_epoch

            if global_epoch == 1 or global_epoch % args.print_every == 0 or global_epoch == args.epochs:
                print(
                    f"phase={phase['name']} "
                    f"epoch={global_epoch:03d} "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f}"
                )

            if val_loss < (best_val_loss - args.min_delta):
                best_val_loss = val_loss
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                phase_summary["best_epoch"] = global_epoch
                if phase["name"] != "head_warmup":
                    patience_left = args.patience
            elif phase["name"] != "head_warmup":
                patience_left -= 1
                if patience_left <= 0:
                    print(
                        f"Early stopping at epoch {global_epoch} "
                        f"with best validation loss {best_val_loss:.6f}"
                    )
                    stop_training = True
                    break

        phase_summaries.append(phase_summary)
        if stop_training:
            break

    if best_state is None:
        raise RuntimeError("Training ended without capturing a valid model state.")

    model.load_state_dict(best_state)
    model_predictions, ordered_ids, ordered_sequences = predict_dataset(
        model=model,
        loader=all_loader,
        device=device,
    )

    if ordered_ids != prepared.ids or ordered_sequences != prepared.sequences:
        raise RuntimeError("Prediction order drifted away from the original dataset order.")

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
    summary_path = artifact_path.with_suffix(".json")
    predictions_path = artifact_path.with_name(f"{artifact_path.stem}_predictions.csv")

    checkpoint_payload = {
        "model_state_dict": best_state,
        "model_config": {
            "pooling": args.pooling,
            "head_hidden_dims": args.head_hidden_dims,
            "dropout": args.dropout,
            "nn_projection_dim": args.nn_projection_dim,
            "availability_projection_dim": args.availability_projection_dim,
            "feature_set": args.feature_set,
            "nn_feature_dim": 0 if prepared.nn_features is None else prepared.nn_features.shape[1],
            "availability_feature_dim": (
                0 if prepared.availability_features is None else prepared.availability_features.shape[1]
            ),
            "nn_feature_mode": args.nn_feature_mode,
            "kmer_size": args.kmer_size,
            "max_length": args.max_length,
        },
        "encoder_config": {
            "model_name": args.model_name,
            "model_kmer_size": args.model_kmer_size or args.kmer_size,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        },
        "tuning_config": final_tuning_summary,
        "target_config": {
            "target_column": prepared.target_column,
            "target_transform": prepared.target_transform,
        },
        "nn_feature_config": {
            "feature_names": prepared.nn_feature_names,
            "feature_mean": nn_feature_mean,
            "feature_std": nn_feature_std,
        },
        "availability_feature_config": {
            "feature_names": prepared.availability_feature_names,
            "feature_mean": availability_feature_mean,
            "feature_std": availability_feature_std,
        },
        "training_config": {
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "scheduler": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": total_warmup_steps,
            "loss": args.loss,
            "huber_delta": args.huber_delta if args.loss == "huber" else None,
            "patience": args.patience,
            "head_only_warmup_epochs": args.head_only_warmup_epochs,
            "seed": args.seed,
            "device": str(device),
        },
        "data_config": {
            "table_path": str(args.table_path),
            "id_column": args.id_column,
            "sequence_column": args.sequence_column,
            "row_count": len(prepared.ids),
        },
        "splits": {name: values.tolist() for name, values in indices_by_split.items()},
        "metrics": metrics,
        "phase_summaries": phase_summaries,
        "history": history,
        "predictions_path": str(predictions_path),
    }

    if not args.skip_save:
        torch.save(checkpoint_payload, artifact_path)

    save_predictions_csv(
        path=predictions_path,
        prepared=prepared,
        indices_by_split=indices_by_split,
        raw_predictions=raw_predictions,
        model_predictions=model_predictions,
    )

    summary = {
        "artifact_path": None if args.skip_save else str(artifact_path),
        "predictions_path": str(predictions_path),
        "target_column": prepared.target_column,
        "target_transform": prepared.target_transform,
        "row_count": len(prepared.ids),
        "metrics": metrics,
        "model_name": args.model_name,
        "feature_set": args.feature_set,
        "tuning_config": final_tuning_summary,
        "training_config": checkpoint_payload["training_config"],
        "phase_summaries": phase_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    runtime = get_runtime()
    runtime.state["finetune"] = summary

    checkpoint_note = "without writing a checkpoint" if args.skip_save else f"to {artifact_path}"
    print(
        f"Finished fine-tuning {checkpoint_note}. "
        f"Test RMSE {metrics['test']['raw']['rmse']:.3f}, "
        f"test R^2 {metrics['test']['raw']['r2']:.3f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    train_finetuned_regressor(args)


if __name__ == "__main__":
    main()
