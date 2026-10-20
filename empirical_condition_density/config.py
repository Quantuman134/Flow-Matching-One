"""Strict YAML settings shared by the launcher and comparison runner."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .data import ConditionScheme, SpiralConfig
from .model import ModelConfig
from .training import REPO_ROOT, TrainConfig


@dataclass(frozen=True)
class LaunchConfig:
    """Process count, device selection, and CPU thread budget from YAML."""

    num_processes: int = 1
    device: str = "auto"
    cpu_threads: int = 4

    def __post_init__(self):
        #region Configuration validation
        for name in ("num_processes", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda.")
        #endregion


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings; dataclass defaults are fallbacks for omitted YAML keys."""

    steps: list[int] = field(default_factory=lambda: [1, 4, 16, 64])
    thresholds: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    primary_threshold: float = 0.05
    num_directions: int = 256
    direction_seed: int = 1729
    noise_seed_offset: int = 1_000_000
    batch_size: int = 4096
    spread_conditions: int = 16
    spread_samples: int = 128
    trajectory_count: int = 16
    trajectory_steps: int = 64
    scatter_steps: list[int] = field(default_factory=lambda: [1, 4, 16, 64])
    max_plot_points: int = 2000

    def __post_init__(self):
        #region Configuration validation
        for name in ("num_directions", "batch_size", "spread_conditions", "spread_samples",
                     "trajectory_count", "trajectory_steps", "max_plot_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("steps", "scatter_steps"):
            values = getattr(self, name)
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be a nonempty list of unique step counts.")
            if any(type(s) is not int or s < 1 for s in values):
                raise ValueError(f"{name} must contain positive integers.")
        if 1 not in self.steps or not set(self.scatter_steps) <= set(self.steps):
            raise ValueError("steps must include 1 and every scatter_steps entry.")
        if not isinstance(self.thresholds, list) or not self.thresholds:
            raise ValueError("thresholds must be a nonempty list.")
        if len(set(self.thresholds)) != len(self.thresholds) or any(
            not math.isfinite(e) or e <= 0 for e in self.thresholds
        ) or self.primary_threshold not in self.thresholds:
            raise ValueError("Thresholds must be positive, unique, and include primary_threshold.")
        for name in ("direction_seed", "noise_seed_offset"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32).")
        if not 2 <= self.noise_seed_offset < 2**32 - 1:
            raise ValueError("Evaluation RNG offsets must stay distinct from training and validation.")
        #endregion


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated YAML interface; each K uses the same single training seed."""

    environment: str
    launch: LaunchConfig
    mode: str
    resume: bool
    output_dir: str
    data: SpiralConfig
    model: ModelConfig
    training: dict
    condition_counts: list[int]
    seed: int
    evaluation: EvaluationConfig

    def __post_init__(self):
        #region Configuration validation
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ValueError("environment must be a nonempty Conda environment name.")
        if self.mode not in ("all", "train", "evaluate") or type(self.resume) is not bool:
            raise ValueError("mode must be all/train/evaluate; resume must be a YAML boolean.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path.")
        if not isinstance(self.condition_counts, list) or not self.condition_counts or (
            len(set(self.condition_counts)) != len(self.condition_counts)
        ):
            raise ValueError("condition_counts must be a nonempty list of unique integers.")
        reserved = {"num_conditions", "seed", "device", "output_dir"}
        if not isinstance(self.training, dict) or reserved & self.training.keys():
            raise ValueError("Put K/seed/device/output_dir in their dedicated experiment settings.")
        #endregion
        for count in self.condition_counts:
            ConditionScheme(self.data.num_targets, count)
        self.train_config(self.condition_counts[0])

    @property
    def output_root(self):
        path = Path(self.output_dir).expanduser()
        return (path if path.is_absolute() else REPO_ROOT / path).resolve()

    def train_config(self, count):
        """Combine shared settings with the current condition count for one trainer."""
        return TrainConfig(**self.training, num_conditions=count, seed=self.seed,
                           device=self.launch.device, output_dir=str(self.output_root))

    def protocol(self):
        """Fields that must match before reusing any comparison artifacts."""
        result = asdict(self)
        for key in ("environment", "launch", "mode", "resume", "output_dir"):
            result.pop(key)
        return result


def _construct(cls, values):
    #region YAML validation
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a YAML mapping.")
    unknown = values.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    #endregion
    return cls(**values)


def load_config(path):
    """Load YAML, reject unknown settings, and construct the typed interface."""
    values = yaml.safe_load(Path(path).read_text())
    #region YAML validation
    if not isinstance(values, dict):
        raise ValueError("Experiment YAML must contain a mapping.")
    #endregion
    for key, cls in (("launch", LaunchConfig), ("data", SpiralConfig),
                     ("model", ModelConfig), ("evaluation", EvaluationConfig)):
        values[key] = _construct(cls, values.get(key, {}))
    return _construct(ExperimentConfig, values)
