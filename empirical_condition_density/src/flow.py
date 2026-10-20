"""Linear flow matching paths and the velocity regression objective."""

import torch
from torch import nn


def linear_path(
    x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x_t=(1-t)x0+t*x1 and the target velocity x1-x0.

    Time has shape [B,1]; training samples it continuously in [0,1).
    Endpoints t=0 and t=1 are also accepted for diagnostics.
    """
    #region Input validation
    if x0.ndim != 2 or x0.shape[1] != 2 or x1.shape != x0.shape:
        raise ValueError("x0 and x1 must have matching shapes [B, 2].")
    if time.shape != (len(x0), 1):
        raise ValueError("time must have shape [B, 1].")
    for value in (x0, x1, time):
        if value.dtype != torch.float32 or value.device != x0.device:
            raise ValueError("Path inputs must be FP32 tensors on the same device.")
    #endregion
    return (1 - time) * x0 + time * x1, x1 - x0


def velocity_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean error: sum over coordinates, mean over samples."""
    #region Input validation
    if prediction.ndim != 2 or prediction.shape[1] != 2 or target.shape != prediction.shape:
        raise ValueError("Predicted and target velocities must have shape [B, 2].")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("Velocities must be FP32 tensors.")
    if prediction.device != target.device:
        raise ValueError("Velocities must use the same device.")
    #endregion
    return (prediction - target).square().sum(dim=1).mean()


def flow_matching_loss(
    model: nn.Module, x0: torch.Tensor, x1: torch.Tensor,
    time: torch.Tensor, condition: torch.Tensor,
) -> torch.Tensor:
    """Regress the model velocity onto the sampled pair's displacement."""
    position, target = linear_path(x0, x1, time)
    return velocity_mse(model(position, time, condition), target)
