"""FP32 exposure transforms applied to sRGB source images in memory."""

from dataclasses import dataclass

import torch


def _unit_values(values: torch.Tensor, name: str) -> torch.Tensor:
    """Validate image values and use FP32 for the transfer functions."""
    #region Image value validation
    if not values.is_floating_point() or values.numel() == 0:
        raise ValueError(f"{name} must be a nonempty floating-point tensor.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values.")
    if torch.any(values < 0) or torch.any(values > 1):
        raise ValueError(f"{name} must be in [0,1].")
    #endregion
    values = values.to(torch.float32)
    return values


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Convert normalized sRGB values to normalized linear-light values."""
    srgb = _unit_values(srgb, "sRGB")
    low = srgb / 12.92
    high = (srgb + 0.055) / 1.055
    high = high.pow(2.4)
    linear = torch.where(srgb <= 0.04045, low, high)
    return linear


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert clipped linear-light values to normalized sRGB values."""
    linear = _unit_values(linear, "Linear RGB")
    low = linear * 12.92
    high = linear.pow(1 / 2.4)
    high = 1.055 * high - 0.055
    srgb = torch.where(linear <= 0.0031308, low, high)
    return srgb


@dataclass(frozen=True)
class ExposureResult:
    image: torch.Tensor       # Same shape as source; FP32 sRGB in [0,1].
    saturated: torch.Tensor   # Boolean RGB-element mask before highlight clipping.


def apply_exposure(source: torch.Tensor, exposure: float | torch.Tensor) -> ExposureResult:
    """Apply exposure in stops to [3,H,W] or [B,3,H,W] images.

    Batch conditions may be scalar, [B], or [B,1]. At zero exposure the
    original FP32 values are retained exactly, avoiding a transfer round trip.
    """
    #region Shape validation
    if source.ndim not in (3, 4) or source.shape[-3] != 3:
        raise ValueError("source must have shape [3,H,W] or [B,3,H,W].")
    #endregion
    source = _unit_values(source, "source")
    stops = torch.as_tensor(exposure, dtype=torch.float32, device=source.device)
    batch_size = 1
    if source.ndim == 4:
        batch_size = source.shape[0]

    #region Exposure validation
    if stops.ndim == 0:
        stops = stops.expand(batch_size)
    elif stops.shape == (batch_size, 1):
        stops = stops[:, 0]
    elif stops.shape != (batch_size,):
        raise ValueError("exposure must be scalar, [B], or [B,1].")
    if not torch.isfinite(stops).all():
        raise ValueError("Exposure values must be finite.")
    factors = torch.exp2(stops)
    if not torch.isfinite(factors).all() or torch.any(factors <= 0):
        raise ValueError("Exposure factors must be finite and positive in FP32.")
    #endregion
    shape = (batch_size, 1, 1, 1)
    if source.ndim == 3:
        shape = (1, 1, 1)
    factors = factors.reshape(shape)
    stops = stops.reshape(shape)

    linear = srgb_to_linear(source)
    exposed_linear = linear * factors
    saturated = exposed_linear >= 1
    exposed_linear = exposed_linear.clamp(0, 1)
    adjusted = linear_to_srgb(exposed_linear)
    adjusted = adjusted.clamp(0, 1)
    adjusted = torch.where(stops == 0, source, adjusted)
    result = ExposureResult(adjusted, saturated)
    return result
