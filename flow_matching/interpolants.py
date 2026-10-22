"""Linear interpolation and target velocity for vector or image-like states."""


def linear_path(source, target, times):
    """Return source + t × displacement and target-source in FP32; times are [B]."""
    #region Path validation
    if source.ndim < 2 or source.shape != target.shape or source.numel() == 0:
        raise ValueError("Endpoints must have matching, nonempty [B,...] shapes.")
    if times.shape != (len(source),):
        raise ValueError("times must have shape [B].")
    if source.device != target.device or times.device != source.device:
        raise ValueError("Endpoints and times must share a device.")
    for values in (source, target, times):
        if not values.is_floating_point():
            raise ValueError("Endpoints and times must be floating-point tensors.")
    #endregion
    source = source.float()
    target = target.float()
    broadcast_shape = (len(source),) + (1,) * (source.ndim - 1)
    interpolation = times.float().reshape(broadcast_shape)
    displacement = target - source
    state = source + interpolation * displacement
    return state, displacement
