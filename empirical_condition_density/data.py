"""In-memory spiral construction and fresh Gaussian/spiral pair sampling.

Targets and batches use FP32. Geometry and normalization are computed in FP64
before casting, so increasing the target count does not degrade the grid.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpiralConfig:
    num_targets: int = 50_000
    num_turns: float = 3.0
    radius_start: float = 0.2
    radius_end: float = 2.0

    def __post_init__(self):
        #region Configuration validation
        if type(self.num_targets) is not int or self.num_targets < 2:
            raise ValueError("num_targets must be an integer of at least 2.")
        values = (self.num_turns, self.radius_start, self.radius_end)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Spiral parameters must be finite.")
        if self.num_turns <= 0 or not 0 <= self.radius_start < self.radius_end:
            raise ValueError("Require num_turns > 0 and 0 <= radius_start < radius_end.")
        #endregion


@dataclass(frozen=True)
class SpiralData:
    targets: torch.Tensor       # [N, 2], FP32
    parameters: torch.Tensor    # [N], FP64; positions along the spiral
    center: torch.Tensor        # [2], FP64; mean before normalization
    scale: float                # shared normalization factor for both coordinates


def build_spiral(config: SpiralConfig = SpiralConfig()) -> SpiralData:
    """Build a fixed, centered spiral with mean squared radius equal to 2."""
    s = (torch.arange(config.num_targets, dtype=torch.float64, device="cpu") + 0.5)
    s = s / config.num_targets
    radius = config.radius_start + (config.radius_end - config.radius_start) * s
    angle = 2 * math.pi * config.num_turns * s
    raw = radius[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    center = raw.mean(dim=0)
    centered = raw - center
    scale = math.sqrt(centered.square().sum(dim=1).mean().item() / 2)
    #region Scale validation
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("Spiral geometry has zero or non-finite scale.")
    #endregion
    targets = (centered / scale).to(torch.float32)
    return SpiralData(targets=targets, parameters=s, center=center, scale=scale)


@dataclass(frozen=True)
class ConditionScheme:
    num_targets: int
    num_conditions: int

    def __post_init__(self):
        #region Configuration validation
        for name in ("num_targets", "num_conditions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_targets % self.num_conditions:
            raise ValueError("num_conditions must divide num_targets exactly.")
        #endregion

    @property
    def targets_per_condition(self) -> int:
        return self.num_targets // self.num_conditions

    def group_indices(self, target_indices: torch.Tensor) -> torch.Tensor:
        """Map a vector of target indices to integer group indices."""
        #region Input validation
        if target_indices.ndim != 1 or target_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("target_indices must be a one-dimensional integer tensor.")
        if torch.any((target_indices < 0) | (target_indices >= self.num_targets)):
            raise ValueError("Target index outside the dataset.")
        #endregion
        return target_indices // self.targets_per_condition

    def encode(self, target_indices: torch.Tensor) -> torch.Tensor:
        """Return normalized group midpoints as an FP32 tensor of shape [B, 1]."""
        groups = self.group_indices(target_indices)
        conditions = 2 * (groups.to(torch.float32) + 0.5) / self.num_conditions - 1
        return conditions.unsqueeze(1)

    def group_means(self, targets: torch.Tensor) -> torch.Tensor:
        """Exact group-mean reference for ideal one-step MSE prediction."""
        #region Input validation
        if targets.shape != (self.num_targets, 2):
            raise ValueError(f"targets must have shape [{self.num_targets}, 2].")
        #endregion
        groups = targets.reshape(self.num_conditions, self.targets_per_condition, 2)
        return groups.to(torch.float64).mean(dim=1).to(targets.dtype)


@dataclass(frozen=True)
class PairBatch:
    x0: torch.Tensor             # [B, 2], fresh independent N(0, I) noise
    x1: torch.Tensor             # [B, 2], sampled fixed targets
    condition: torch.Tensor      # [B, 1], normalized group midpoints
    target_indices: torch.Tensor # [B], indices into the fixed target set


def sample_pairs(
    data: SpiralData,
    scheme: ConditionScheme,
    batch_size: int,
    generator: torch.Generator,
) -> PairBatch:
    """Sample CPU pairs with replacement; the caller owns the RNG stream.

    Reusing the same RNG state across condition schemes yields the same noise
    and target indices. Successive calls draw fresh noise, including when a
    target is sampled again; no fixed noise-to-target pairing is stored.
    """
    #region Input validation
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if data.targets.shape != (scheme.num_targets, 2):
        raise ValueError("Dataset and condition scheme sizes do not match.")
    #endregion
    x0 = torch.randn(batch_size, 2, generator=generator, dtype=torch.float32, device="cpu")
    indices = torch.randint(
        scheme.num_targets, (batch_size,), generator=generator, device="cpu"
    )
    return PairBatch(
        x0=x0,
        x1=data.targets[indices],
        condition=scheme.encode(indices),
        target_indices=indices,
    )
