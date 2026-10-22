"""Fixed-step Euler inference with explicit NFE and optional trajectories."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingResult:
    samples: torch.Tensor
    num_evaluations: int
    trajectory: torch.Tensor | None = None


@torch.no_grad()
def euler_sample(
    model, source, condition, nfe=1, mixed_precision="no", *,
    return_trajectory=False, matmul_precision=None,
):
    """Integrate t=0 to t=1, calling model(state, times:[B], condition) nfe times.

    State updates use FP32; autocast applies to the velocity network. Optional
    matmul precision is scoped to this call. Caller model mode is restored.
    """
    #region Sampler validation
    if type(nfe) is not int or nfe < 1:
        raise ValueError("nfe must be a positive integer.")
    if mixed_precision not in ("no", "bf16"):
        raise ValueError("mixed_precision must be no or bf16.")
    if source.ndim < 2 or source.numel() == 0 or not source.is_floating_point():
        raise ValueError("source must be a nonempty floating-point tensor [B,...].")
    if condition.ndim < 1 or len(condition) != len(source) or condition.device != source.device:
        raise ValueError("condition must share the source batch size and device.")
    if matmul_precision not in (None, "highest", "high", "medium"):
        raise ValueError("matmul_precision must be highest, high, medium, or None.")
    #endregion
    state = source.float().clone()
    trajectory = None
    if return_trajectory:
        trajectory = [state.clone()]
    was_training = model.training
    previous_precision = torch.get_float32_matmul_precision()
    model.eval()
    try:
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)
        for index in range(nfe):
            times = torch.full((len(state),), index / nfe, dtype=torch.float32, device=state.device)
            with torch.autocast(state.device.type, dtype=torch.bfloat16, enabled=mixed_precision == "bf16"):
                velocity = model(state, times, condition)
            #region Velocity validation
            if velocity.shape != state.shape or not velocity.is_floating_point() or velocity.device != state.device:
                raise ValueError("Model velocity must match the source shape and device.")
            if not torch.isfinite(velocity).all():
                raise FloatingPointError("The model returned non-finite velocities.")
            #endregion
            state = state + velocity.float() / nfe
            if trajectory is not None:
                trajectory.append(state.clone())
        if not torch.isfinite(state).all():
            raise FloatingPointError("Euler sampling produced non-finite states.")
    finally:
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(previous_precision)
        model.train(was_training)
    if trajectory is not None:
        trajectory = torch.stack(trajectory)
    result = SamplingResult(state, nfe, trajectory)
    return result
