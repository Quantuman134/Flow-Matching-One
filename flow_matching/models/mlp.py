"""Conditional MLP for vector-valued flow states."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    hidden_dim: int = 128
    num_hidden_layers: int = 4

    def __post_init__(self):
        #region Configuration validation
        for name in ("hidden_dim", "num_hidden_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        #endregion


class MLP(nn.Module):
    """Predict [B,D] FP32 velocities from state, time, and [B,C] conditions.

    Times use [B].
    Default dimensions and parameter names preserve the spiral checkpoints.
    """

    def __init__(self, config: MLPConfig = MLPConfig(), *, state_dim=2, condition_dim=1):
        super().__init__()
        #region Dimension validation
        for value in (state_dim, condition_dim):
            if type(value) is not int or value < 1:
                raise ValueError("State and condition dimensions must be positive integers.")
        #endregion
        self.config = config
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        layers = []
        input_dim = state_dim + 1 + condition_dim
        for _ in range(config.num_hidden_layers):
            linear = nn.Linear(input_dim, config.hidden_dim, dtype=torch.float32)
            activation = nn.SiLU()
            layers.extend([linear, activation])
            input_dim = config.hidden_dim
        output = nn.Linear(input_dim, state_dim, dtype=torch.float32)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, position, times, condition):
        batch_size = len(position)
        #region Input validation
        if position.shape != (batch_size, self.state_dim):
            raise ValueError("position must have shape [B,state_dim].")
        if times.shape != (batch_size,) or condition.shape != (batch_size, self.condition_dim):
            raise ValueError("Use times [B] and condition [B,condition_dim].")
        for value in (position, times, condition):
            if value.dtype != torch.float32 or value.device != position.device:
                raise ValueError("Model inputs must be FP32 tensors on the same device.")
        #endregion
        column_times = times[:, None]
        inputs = torch.cat((position, column_times, condition), dim=1)
        velocity = self.network(inputs)
        return velocity
