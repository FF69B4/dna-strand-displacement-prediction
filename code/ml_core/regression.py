from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def invert_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "none":
        return values.astype(np.float32)
    if transform_name == "log10":
        return np.power(10.0, values).astype(np.float32)
    if transform_name == "ln":
        return np.exp(values).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {transform_name}")


def predict_array(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (feature_batch,) in loader:
            predicted = model(feature_batch.to(device))
            predictions.append(predicted.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)
