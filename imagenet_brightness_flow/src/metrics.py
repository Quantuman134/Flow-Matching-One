"""Per-image errors with FP64 reductions and explicit flat-image handling.

Keep MSE values until aggregation: pooled RMSE is sqrt(mean(per-image MSE)).
Gain-fit components use the same gain_valid mask when forming dataset means.
"""

import math

import torch

from .brightness import srgb_to_linear


def _matching_batches(predicted: torch.Tensor, reference: torch.Tensor, dimensions: int) -> None:
    #region Metric input validation
    if predicted.shape != reference.shape or predicted.ndim != dimensions:
        raise ValueError(f"Metric inputs must have matching {dimensions}-dimensional shapes.")
    if predicted.numel() == 0 or predicted.device != reference.device:
        raise ValueError("Metric inputs must be nonempty and share a device.")
    if not predicted.is_floating_point() or not reference.is_floating_point():
        raise ValueError("Metric inputs must be floating-point tensors.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(reference).all():
        raise ValueError("Metric inputs must be finite.")
    #endregion


@torch.no_grad()
def luminance_decomposition(
    predicted: torch.Tensor, reference: torch.Tensor, flat_variance_threshold: float,
) -> dict[str, torch.Tensor]:
    """Decompose [B,H,W] luminance errors into offset, gain, and residual MSE.

    Gain, gain MSE, and residual MSE are NaN for nearly uniform references.
    The offset-plus-centered decomposition is defined for every image.
    """
    _matching_batches(predicted, reference, 3)
    #region Threshold validation
    if not math.isfinite(flat_variance_threshold) or flat_variance_threshold <= 0:
        raise ValueError("flat_variance_threshold must be finite and positive.")
    #endregion
    batch_size = predicted.shape[0]
    predicted = predicted.to(torch.float64)
    predicted = predicted.reshape(batch_size, -1)
    reference = reference.to(torch.float64)
    reference = reference.reshape(batch_size, -1)
    predicted_mean = predicted.mean(dim=1)
    reference_mean = reference.mean(dim=1)
    predicted_centered = predicted - predicted_mean[:, None]
    reference_centered = reference - reference_mean[:, None]
    reference_squared = reference_centered.square()
    reference_variance = reference_squared.mean(dim=1)
    gain_valid = reference_variance > flat_variance_threshold

    covariance_values = predicted_centered * reference_centered
    covariance = covariance_values.mean(dim=1)
    ones = torch.ones_like(reference_variance)
    denominator = torch.where(gain_valid, reference_variance, ones)
    gain = covariance / denominator
    offset = predicted_mean - gain * reference_mean
    residual = predicted_centered - gain[:, None] * reference_centered
    residual_squared = residual.square()
    residual_mse = residual_squared.mean(dim=1)
    gain_error = gain - 1
    gain_mse = gain_error.square() * reference_variance

    difference = predicted - reference
    squared = difference.square()
    luminance_mse = squared.mean(dim=1)
    signed_mean_error = predicted_mean - reference_mean
    mean_brightness_mse = signed_mean_error.square()
    centered_error = predicted_centered - reference_centered
    centered_squared = centered_error.square()
    centered_mse = centered_squared.mean(dim=1)
    unavailable = torch.full_like(gain, float("nan"))

    values = {}
    values["luminance_mse"] = luminance_mse
    values["signed_mean_luminance_error"] = signed_mean_error
    values["mean_brightness_mse"] = mean_brightness_mse
    values["centered_luminance_mse"] = centered_mse
    values["reference_variance"] = reference_variance
    values["gain_valid"] = gain_valid
    values["gain"] = torch.where(gain_valid, gain, unavailable)
    values["offset"] = torch.where(gain_valid, offset, unavailable)
    values["global_gain_mse"] = torch.where(gain_valid, gain_mse, unavailable)
    values["remaining_spatial_mse"] = torch.where(gain_valid, residual_mse, unavailable)
    return values


def _luminance(srgb: torch.Tensor) -> torch.Tensor:
    linear = srgb_to_linear(srgb)
    linear = linear.to(torch.float64)
    red = linear[:, 0] * 0.2126
    green = linear[:, 1] * 0.7152
    blue = linear[:, 2] * 0.0722
    luminance = red + green + blue
    return luminance


def _masked_error(squared_error: torch.Tensor, mask: torch.Tensor):
    selected = squared_error * mask
    error_sum = selected.sum(dim=(1, 2, 3))
    count = mask.sum(dim=(1, 2, 3))
    safe_count = count.clamp_min(1)
    mse = error_sum / safe_count
    unavailable = torch.full_like(mse, float("nan"))
    mse = torch.where(count > 0, mse, unavailable)
    return mse, error_sum, count


@torch.no_grad()
def image_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    flat_variance_threshold: float,
    saturated: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-image metrics for [B,3,H,W] outputs and FP32 references.

    Displayed RGB/luminance errors use clipped outputs. Raw-output errors and
    range violations retain the original predictions. Masked error sums and
    counts support value-weighted saturation metrics across batches.
    """
    _matching_batches(predicted, reference, 4)
    #region RGB and saturation validation
    if predicted.shape[1] != 3:
        raise ValueError("Image metrics require three RGB channels.")
    if torch.any(reference < 0) or torch.any(reference > 1):
        raise ValueError("reference RGB must be in [0,1].")
    if saturated is not None:
        if saturated.shape != reference.shape or saturated.dtype != torch.bool:
            raise ValueError("saturated must be a boolean mask matching reference RGB.")
        if saturated.device != reference.device:
            raise ValueError("The saturation mask must share the reference device.")
    #endregion
    displayed = predicted.clamp(0, 1)
    predicted_luminance = _luminance(displayed)
    reference_luminance = _luminance(reference)
    values = luminance_decomposition(
        predicted_luminance, reference_luminance, flat_variance_threshold,
    )

    displayed = displayed.to(torch.float64)
    predicted = predicted.to(torch.float64)
    reference = reference.to(torch.float64)
    difference = displayed - reference
    squared_error = difference.square()
    raw_difference = predicted - reference
    raw_squared_error = raw_difference.square()
    violations = (predicted < 0) | (predicted > 1)
    violations = violations.to(torch.float64)
    values["rgb_mse"] = squared_error.mean(dim=(1, 2, 3))
    values["raw_rgb_mse"] = raw_squared_error.mean(dim=(1, 2, 3))
    values["range_violation_fraction"] = violations.mean(dim=(1, 2, 3))

    if saturated is not None:
        saturated_mse, saturated_sum, saturated_count = _masked_error(squared_error, saturated)
        unsaturated = ~saturated
        unsaturated_mse, unsaturated_sum, unsaturated_count = _masked_error(squared_error, unsaturated)
        values["saturated_rgb_mse"] = saturated_mse
        values["saturated_squared_error_sum"] = saturated_sum
        values["saturated_value_count"] = saturated_count
        values["unsaturated_rgb_mse"] = unsaturated_mse
        values["unsaturated_squared_error_sum"] = unsaturated_sum
        values["unsaturated_value_count"] = unsaturated_count
    return values


@torch.no_grad()
def latent_mse(predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return [B] latent MSE; decoding is outside this measurement."""
    _matching_batches(predicted, reference, 4)
    predicted = predicted.to(torch.float64)
    reference = reference.to(torch.float64)
    difference = predicted - reference
    squared = difference.square()
    values = squared.mean(dim=(1, 2, 3))
    return values
