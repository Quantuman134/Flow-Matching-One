"""Flow-matching loss composition and FP32 velocity regression."""


def velocity_mse(prediction, target, *, reduction="mean"):
    """Mean of all squared errors, or mean per-sample squared Euclidean error."""
    #region Velocity validation
    if prediction.shape != target.shape or target.ndim < 2 or target.numel() == 0:
        raise ValueError("Predicted and target velocities must have matching [B,...] shapes.")
    if prediction.device != target.device:
        raise ValueError("Predicted and target velocities must share a device.")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Velocities must be floating-point tensors.")
    if reduction not in ("mean", "sum_per_sample"):
        raise ValueError("reduction must be mean or sum_per_sample.")
    #endregion
    error = prediction.float() - target.float()
    squared_error = error.square()
    if reduction == "sum_per_sample":
        features = squared_error.flatten(start_dim=1)
        squared_error = features.sum(dim=1)
    loss = squared_error.mean()
    return loss


def flow_matching_loss(prediction, target_velocity, *, reduction="mean"):
    """Combine flow-matching loss terms; currently the velocity MSE term."""
    loss = velocity_mse(prediction, target_velocity, reduction=reduction)
    return loss
