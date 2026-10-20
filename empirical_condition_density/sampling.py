"""Deterministic Euler inference with an explicit number of model evaluations."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SamplingResult:
    samples: torch.Tensor                  # [B, 2], final positions
    num_evaluations: int                   # one model call per Euler step
    trajectory: torch.Tensor | None = None # [S+1, B, 2], including x0


@torch.no_grad()
def sample_euler(
    model: nn.Module, x0: torch.Tensor, condition: torch.Tensor,
    num_steps: int = 1, *, return_trajectory: bool = False,
) -> SamplingResult:
    """Integrate from t=0 to t=1; evaluate velocities at t=k/S, k<S.

    Inputs are not modified, and the model's original train/eval mode is restored.
    The caller supplies noise so comparisons can reuse exactly the same x0.
    """
    #region Input validation
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    if x0.ndim != 2 or x0.shape[1] != 2 or len(x0) == 0:
        raise ValueError("x0 must have shape [B, 2] with a nonempty batch.")
    if condition.shape != (len(x0), 1):
        raise ValueError("condition must have shape [B, 1].")
    if x0.dtype != torch.float32 or condition.dtype != torch.float32:
        raise ValueError("Sampling inputs must be FP32 tensors.")
    if condition.device != x0.device or x0.device.type not in ("cpu", "cuda"):
        raise ValueError("Sampling inputs must share a CPU or CUDA device.")
    #endregion
    position = x0.clone()
    trajectory = [position.clone()] if return_trajectory else None
    was_training = model.training
    previous_precision = torch.get_float32_matmul_precision()
    model.eval()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type=x0.device.type, enabled=False):
            for step in range(num_steps):
                time = torch.full(
                    (len(x0), 1), step / num_steps, dtype=torch.float32, device=x0.device,
                )
                velocity = model(position, time, condition)
                #region Velocity validation
                if velocity.shape != position.shape or velocity.dtype != torch.float32:
                    raise ValueError("The model must return FP32 velocities of shape [B, 2].")
                if not torch.isfinite(velocity).all():
                    raise FloatingPointError("The model returned non-finite velocities.")
                #endregion
                position = position + velocity / num_steps
                if trajectory is not None:
                    trajectory.append(position.clone())
        #region Final position validation
        if not torch.isfinite(position).all():
            raise FloatingPointError("Euler sampling produced non-finite positions.")
        #endregion
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        model.train(was_training)
    return SamplingResult(
        samples=position, num_evaluations=num_steps,
        trajectory=torch.stack(trajectory) if trajectory is not None else None,
    )
