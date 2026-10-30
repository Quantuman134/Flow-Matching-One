"""Validated experiment settings for online brightness-flow training."""

import math
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from .credentials import load_wandb_credentials


@dataclass(frozen=True)
class DataConfig:
    root: str
    image_size: int
    train_flip_probability: float
    exposure_min: float
    exposure_max: float
    evaluation_exposures: list[float]
    monitor_per_class: int
    preview_images: int
    expected_classes: int
    expected_train_images: int
    expected_validation_images: int

    def __post_init__(self):
        #region Data setting validation
        if not isinstance(self.root, str) or not self.root.strip():
            raise ValueError("data.root must be a nonempty path.")
        positive_fields = [
            "image_size", "monitor_per_class", "preview_images", "expected_classes",
            "expected_train_images", "expected_validation_images",
        ]
        for name in positive_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"data.{name} must be a positive integer.")
        if not 0 <= self.train_flip_probability <= 1:
            raise ValueError("data.train_flip_probability must be in [0,1].")
        if not math.isfinite(self.exposure_min) or not math.isfinite(self.exposure_max):
            raise ValueError("Exposure limits must be finite.")
        if self.exposure_min >= self.exposure_max:
            raise ValueError("exposure_min must be smaller than exposure_max.")
        if not isinstance(self.evaluation_exposures, list) or not self.evaluation_exposures:
            raise ValueError("evaluation_exposures must be a nonempty list.")
        if len(set(self.evaluation_exposures)) != len(self.evaluation_exposures):
            raise ValueError("evaluation_exposures must contain unique values.")
        for exposure in self.evaluation_exposures:
            if not math.isfinite(exposure) or not self.exposure_min <= exposure <= self.exposure_max:
                raise ValueError("Evaluation exposure must be finite and within training limits.")
        if 0 not in self.evaluation_exposures:
            raise ValueError("evaluation_exposures must include the identity condition 0.")
        if self.preview_images < len(self.evaluation_exposures):
            raise ValueError("preview_images must cover every evaluation exposure.")
        monitor_size = self.expected_classes * self.monitor_per_class
        if self.preview_images > monitor_size:
            raise ValueError("preview_images must fit inside the monitor subset.")
        #endregion


@dataclass(frozen=True)
class MetricConfig:
    flat_variance_threshold: float

    def __post_init__(self):
        #region Metric setting validation
        if not math.isfinite(self.flat_variance_threshold) or self.flat_variance_threshold <= 0:
            raise ValueError("flat_variance_threshold must be finite and positive.")
        #endregion


@dataclass(frozen=True)
class VAEConfig:
    model_id: str
    revision: str
    scale: float
    batch_size: int
    local_files_only: bool

    def __post_init__(self):
        #region VAE validation
        if self.model_id != "stabilityai/sd-vae-ft-ema":
            raise ValueError("Use the planned stabilityai/sd-vae-ft-ema VAE.")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("vae.revision must pin a 40-character commit hash.")
        if self.scale != 0.18215:
            raise ValueError("vae.scale must be 0.18215.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("vae.batch_size must be a positive integer.")
        if type(self.local_files_only) is not bool:
            raise ValueError("vae.local_files_only must be boolean.")
        #endregion


@dataclass(frozen=True)
class TrainingConfig:
    timesteps: str | int
    updates: int
    global_batch: int
    microbatch: int
    accumulation: int
    processes: int
    device: str
    mixed_precision: str
    workers: int
    cpu_threads: int
    learning_rate: float
    adam_betas: list[float]
    weight_decay: float
    ema_decay: float
    log_interval: int
    checkpoint_interval: int
    output_dir: str
    resume_from: str | None

    def __post_init__(self):
        #region Training validation
        if self.timesteps != "continuous" and (type(self.timesteps) is not int or self.timesteps not in (16, 4, 1)):
            raise ValueError("training.timesteps must be continuous, 16, 4, or 1.")
        positive = (
            "updates", "global_batch", "microbatch", "accumulation", "processes",
            "cpu_threads", "log_interval", "checkpoint_interval",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{name} must be a positive integer.")
        expected = self.microbatch * self.processes * self.accumulation
        if self.global_batch != expected:
            raise ValueError("global_batch must equal microbatch × processes × accumulation.")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("training.workers must be a nonnegative integer.")
        if self.device not in ("auto", "cpu", "cuda") or self.mixed_precision not in ("no", "bf16"):
            raise ValueError("Use device auto/cpu/cuda and mixed_precision no/bf16.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("adam_betas must contain two values in [0,1).")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1).")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path.")
        if self.resume_from is not None and (not isinstance(self.resume_from, str) or not self.resume_from.strip()):
            raise ValueError("resume_from must be null or a nonempty path.")
        #endregion


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool
    interval: int
    batch_size: int
    workers: int
    exposures: list[float]
    nfe: int

    def __post_init__(self):
        #region Validation protocol
        if type(self.enabled) is not bool:
            raise ValueError("validation.enabled must be boolean.")
        for name in ("interval", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"validation.{name} must be a positive integer.")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("validation.workers must be a nonnegative integer.")
        if not self.exposures or len(set(self.exposures)) != len(self.exposures):
            raise ValueError("validation.exposures must contain unique exposure values.")
        if self.nfe != 1 or type(self.nfe) is not int:
            raise ValueError("Training monitor validation uses NFE 1.")
        #endregion


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    group: str
    mode: str
    images_per_validation: int
    entity: str | None = None
    api_key_env: str = "WANDB_API_KEY"

    def __post_init__(self):
        #region Tracking validation
        if type(self.enabled) is not bool:
            raise ValueError("wandb.enabled must be boolean.")
        for name in ("project", "group", "api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"wandb.{name} must be a nonempty string.")
        if self.entity is not None and (not isinstance(self.entity, str) or not self.entity.strip()):
            raise ValueError("wandb.entity must be null or a nonempty string.")
        if self.mode not in ("online", "offline"):
            raise ValueError("wandb.mode must be online or offline.")
        if self.images_per_validation != 16 or type(self.images_per_validation) is not int:
            raise ValueError("wandb.images_per_validation must be 16.")
        #endregion


@dataclass(frozen=True)
class PilotConfig:
    global_batches: list[int]
    image_budget: int
    warmup_images: int
    output_dir: str

    def __post_init__(self):
        #region Pilot validation
        if not self.global_batches or len(set(self.global_batches)) != len(self.global_batches):
            raise ValueError("pilot.global_batches must contain unique batch sizes.")
        if type(self.image_budget) is not int or self.image_budget < 1:
            raise ValueError("pilot.image_budget must be a positive integer.")
        if type(self.warmup_images) is not int or not 0 <= self.warmup_images < self.image_budget:
            raise ValueError("pilot.warmup_images must be smaller than the image budget.")
        for batch in self.global_batches:
            if type(batch) is not int or batch < 1 or self.image_budget % batch or self.warmup_images % batch:
                raise ValueError("Pilot image budget and warmup must divide by each positive global batch.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("pilot.output_dir must be a nonempty path.")
        #endregion


@dataclass(frozen=True)
class InputConfig:
    environment: str
    seed: int
    data: DataConfig
    metrics: MetricConfig
    vae: VAEConfig
    training: TrainingConfig
    validation: ValidationConfig
    wandb: WandbConfig
    pilot: PilotConfig

    def __post_init__(self):
        #region Input setting validation
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ValueError("environment must be a nonempty Conda environment name.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32).")
        if self.validation.enabled:
            for exposure in self.validation.exposures:
                if exposure not in self.data.evaluation_exposures:
                    raise ValueError("Monitor exposures must belong to data.evaluation_exposures.")
            if 0 not in self.validation.exposures:
                raise ValueError("Monitor exposures must include zero for identity RMSE.")
        if self.wandb.enabled and self.data.preview_images != self.wandb.images_per_validation:
            raise ValueError("Enabled W&B logging requires 16 fixed preview cases.")
        #endregion


def _settings(config_type, values: dict):
    """Reject missing required or misspelled keys; retain dataclass defaults."""
    #region Mapping validation
    if not isinstance(values, dict):
        raise ValueError(f"{config_type.__name__} settings must be a mapping.")
    expected = set()
    required = set()
    for field in fields(config_type):
        expected.add(field.name)
        if field.default is MISSING and field.default_factory is MISSING:
            required.add(field.name)
    supplied = set(values)
    missing = sorted(required - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        raise ValueError(f"{config_type.__name__}: missing keys {missing}; unknown keys {unknown}.")
    #endregion
    result = config_type(**values)
    return result


def load_config(path: str | Path) -> InputConfig:
    """Read the implemented input settings from the experiment YAML file."""
    path = Path(path)
    with path.open() as stream:
        values = yaml.safe_load(stream)

    #region Top-level validation
    if not isinstance(values, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    expected = {"environment", "seed", "data", "metrics", "vae", "training", "validation", "wandb", "pilot"}
    if set(values) != expected:
        raise ValueError(f"Configuration keys must be {sorted(expected)}.")
    #endregion
    data = _settings(DataConfig, values["data"])
    metrics = _settings(MetricConfig, values["metrics"])
    vae = _settings(VAEConfig, values["vae"])
    training = _settings(TrainingConfig, values["training"])
    values["data"] = data
    values["metrics"] = metrics
    values["vae"] = vae
    values["training"] = training
    values["validation"] = _settings(ValidationConfig, values["validation"])
    values["wandb"] = _settings(WandbConfig, values["wandb"])
    values["pilot"] = _settings(PilotConfig, values["pilot"])
    config = InputConfig(**values)
    if config.wandb.enabled:
        load_wandb_credentials(config.wandb.api_key_env)
    return config
