"""Settings and timestep coverage for the four-model comparison."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from .config import _settings, load_config
from .filesystem import resolve_project_path


@dataclass(frozen=True)
class ComparisonConfig:
    """Use one base training protocol and vary only its allowed timesteps."""

    environment: str
    base_config: str
    output_dir: str
    mode: str
    resume: bool
    timesteps: list[str | int]
    nfe: list[int]
    batch_size: int
    workers: int
    log_interval: int

    def __post_init__(self):
        #region Comparison settings validation
        for name in ("environment", "base_config", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"comparison.{name} must be a nonempty string.")
        if self.mode not in ("all", "train", "evaluate", "report"):
            raise ValueError("comparison.mode must be all/train/evaluate/report.")
        if type(self.resume) is not bool:
            raise ValueError("comparison.resume must be boolean.")
        for name in ("timesteps", "nfe"):
            values = getattr(self, name)
            if not isinstance(values, list) or not values:
                raise ValueError(f"comparison.{name} must be a nonempty list.")
            unique = set(values)
            if len(unique) != len(values):
                raise ValueError(f"comparison.{name} must contain unique settings.")
        for value in self.timesteps:
            if value != "continuous" and (type(value) is not int or value not in (16, 4, 1)):
                raise ValueError("Comparison timesteps must be continuous, 16, 4, or 1.")
        for value in self.nfe:
            if type(value) is not int or value not in (1, 4, 16):
                raise ValueError("Comparison NFE must be 1, 4, or 16.")
        for name in ("batch_size", "log_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"comparison.{name} must be a positive integer.")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("comparison.workers must be a nonnegative integer.")
        #endregion


def load_comparison(path):
    """Read comparison settings and the existing strict training YAML."""
    path = resolve_project_path(path)
    contents = path.read_text()
    values = yaml.safe_load(contents)
    settings = _settings(ComparisonConfig, values)
    # The comparison YAML and its snapshots own their base/output references.
    base_path = Path(settings.base_config)
    base_path = base_path.expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base_path = base_path.resolve()
    output = Path(settings.output_dir)
    output = output.expanduser()
    if not output.is_absolute():
        output = path.parent / output
    output = output.resolve()
    base_argument = str(base_path)
    output_argument = str(output)
    settings = replace(settings, base_config=base_argument, output_dir=output_argument)
    config = load_config(base_path)
    #region Environment validation
    if settings.environment != config.environment:
        raise ValueError("Comparison and base configuration must use the same environment.")
    #endregion
    return settings, config


def comparison_protocol(settings, config):
    """Match scientific settings while allowing relocated paths and device counts."""
    base = asdict(config)
    for name in ("environment", "validation", "wandb", "pilot"):
        base.pop(name)
    base["data"].pop("root")
    for name in ("timesteps", "output_dir", "resume_from", "device", "processes",
                 "microbatch", "accumulation", "workers", "cpu_threads"):
        base["training"].pop(name)
    result = {"base": base, "timesteps": settings.timesteps, "nfe": settings.nfe}
    return result


def timestep_coverage(timesteps, nfe):
    """Flag Euler calls outside a discrete model's supervised left-endpoint grid."""
    times = []
    includes_unsupervised = False
    for index in range(nfe):
        times.append(index / nfe)
        if timesteps != "continuous" and (index * timesteps) % nfe:
            includes_unsupervised = True
    if timesteps == "continuous":
        label = "continuous_boundary" if nfe == 1 else "continuous_field"
    else:
        label = "includes_unsupervised_timesteps" if includes_unsupervised else "supervised_grid"
    result = {
        "euler_times": times, "timestep_coverage": label,
        "includes_unsupervised_timesteps": includes_unsupervised,
    }
    return result
