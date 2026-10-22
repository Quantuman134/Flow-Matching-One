"""Continuous and finite left-endpoint timestep sampling."""

import torch


def sample_timesteps(batch_size, setting, generator, device="cpu"):
    """Sample [B] FP32 times; every setting consumes one uniform draw per image."""
    #region Timestep setting validation
    if setting != "continuous" and (type(setting) is not int or setting < 1):
        raise ValueError("setting must be continuous or a positive integer.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    #endregion
    times = torch.rand(batch_size, generator=generator, device=generator.device)
    if setting != "continuous":
        times = torch.floor(times * setting) / setting
    times = times.to(device=device, dtype=torch.float32)
    return times
