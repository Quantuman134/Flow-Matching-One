"""Conditional velocity MLP for positions [x, y], time, and one condition."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    num_hidden_layers: int = 4

    def __post_init__(self):
        #region Configuration validation
        for name in ("hidden_dim", "num_hidden_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        #endregion


class ConditionalMLP(nn.Module):
    """Predict [B, 2] FP32 velocities from x:[B,2], t:[B,1], c:[B,1]."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        layers = []
        input_dim = 4
        for _ in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim, dtype=torch.float32),
                nn.SiLU(),
            ])
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, 2, dtype=torch.float32))
        self.network = nn.Sequential(*layers)

    def forward(
        self, position: torch.Tensor, time: torch.Tensor, condition: torch.Tensor,
    ) -> torch.Tensor:
        #region Input validation
        if position.ndim != 2 or position.shape[1] != 2:
            raise ValueError("position must have shape [B, 2].")
        if time.shape != (len(position), 1) or condition.shape != time.shape:
            raise ValueError("time and condition must have shape [B, 1].")
        for value in (position, time, condition):
            if value.dtype != torch.float32:
                raise ValueError("Model inputs must be FP32 tensors.")
            if value.device != position.device:
                raise ValueError("All model inputs must use the same device.")
        #endregion
        return self.network(torch.cat((position, time, condition), dim=1))
