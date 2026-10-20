"""FP32 Accelerate trainer with fixed global batches and resumable checkpoints."""

import json
import logging
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from .data import ConditionScheme, PairBatch, SpiralConfig, build_spiral, sample_pairs
from .flow import flow_matching_loss
from .model import ConditionalMLP, ModelConfig


MODULE_PATH = Path(__file__)
MODULE_PATH = MODULE_PATH.resolve()
REPO_ROOT = MODULE_PATH.parents[2]


@dataclass(frozen=True)
class TrainConfig:
    num_conditions: int = 1
    num_steps: int = 20_000
    global_batch_size: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"              # auto, cpu, or cuda; rank placement is automatic
    log_interval: int = 100
    validation_interval: int = 1000
    validation_batch_size: int = 2048
    checkpoint_interval: int = 50000
    output_dir: str = "empirical_condition_density/outputs"

    def __post_init__(self):
        #region Configuration validation
        for name in (
            "num_conditions", "num_steps", "global_batch_size",
            "gradient_accumulation_steps", "log_interval", "validation_interval",
            "validation_batch_size", "checkpoint_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "adam_eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if len(self.adam_betas) != 2 or not all(0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("adam_betas must contain two values in [0, 1).")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda.")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must be nonempty.")
        #endregion


@dataclass(frozen=True)
class TrainingResult:
    model: ConditionalMLP
    step: int
    run_dir: Path
    checkpoint: Path


class _TrainingState:
    """Accelerate checkpoint object for update count and the shared data RNG."""

    def __init__(self, seed: int):
        self.step = 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def state_dict(self):
        return {"step": self.step, "data_rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.generator.set_state(state["data_rng"].cpu())


def _slice_batch(batch: PairBatch, selection: slice, device: torch.device):
    return tuple(value[selection].to(device) for value in (batch.x0, batch.x1, batch.condition))


@torch.no_grad()
def _validate(model, batch, times, reference, accelerator):
    """Same held-out noise/time pairs at every validation, with separate RNG."""
    size = len(batch.x0) // accelerator.num_processes
    start = accelerator.process_index * size
    selection = slice(start, start + size)
    x0, x1, condition = _slice_batch(batch, selection, accelerator.device)
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device_type=accelerator.device.type, enabled=False):
            loss = flow_matching_loss(
                model, x0, x1, times[selection].to(accelerator.device), condition,
            )
            endpoint = x0 + model(x0, torch.zeros_like(condition), condition)
            squared_error = (endpoint - reference[selection].to(accelerator.device))
            squared_error = squared_error.square().sum(dim=1).mean()
        values = accelerator.reduce(torch.stack((loss, squared_error)), reduction="mean")
        #region Numerical validation
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite validation metrics.")
        #endregion
        return {
            "velocity_mse": values[0].item(),
            "one_step_reference_rmse": values[1].sqrt().item(),
        }
    finally:
        model.train(was_training)


def _provenance(accelerator):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "git": {"revision": revision, "dirty": dirty},
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "accelerate": version("accelerate"),
        "cuda_runtime": torch.version.cuda, "device": str(accelerator.device),
        "precision": "fp32", "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "world_size": accelerator.num_processes,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def train(
    config: TrainConfig = TrainConfig(), *,
    model_config: ModelConfig = ModelConfig(),
    spiral_config: SpiralConfig = SpiralConfig(),
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train one K/seed run; call on every rank under torchrun for multi-GPU.

    Each update draws one shared global CPU batch and splits it across ranks
    and accumulation steps. This preserves sampled pairs and global batch size
    when GPU count changes. Checkpoint resume requires the same world size and
    accumulation settings; total num_steps may increase.
    """
    scheme = ConditionScheme(spiral_config.num_targets, config.num_conditions)
    #region Device validation
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    #endregion
    accelerator = Accelerator(
        cpu=config.device == "cpu", mixed_precision="no",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    partitions = accelerator.num_processes * config.gradient_accumulation_steps
    #region Distributed batch validation
    if config.global_batch_size % partitions:
        raise ValueError("global_batch_size must divide evenly across ranks and accumulation.")
    if config.validation_batch_size % accelerator.num_processes:
        raise ValueError("validation_batch_size must divide evenly across ranks.")
    #endregion
    micro_batch_size = config.global_batch_size // partitions
    output_root = Path(config.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = output_root.resolve() / f"K-{config.num_conditions:05d}" / f"seed-{config.seed}"
    occupied = torch.tensor(int((run_dir / "run.yaml").exists()), device=accelerator.device)
    occupied = accelerator.reduce(occupied, reduction="sum").item()
    #region Output validation
    if occupied and resume_from is None:
        raise FileExistsError(f"Run already exists: {run_dir}; resume or choose another output_dir.")
    #endregion

    settings = asdict(config)
    settings.update(output_dir=str(output_root.resolve()), adam_betas=list(config.adam_betas))
    signature = {
        "model": asdict(model_config), "spiral": asdict(spiral_config),
        "training": {key: settings[key] for key in (
            "num_conditions", "global_batch_size", "gradient_accumulation_steps",
            "learning_rate", "adam_betas", "adam_eps", "weight_decay", "seed",
        )},
        "world_size": accelerator.num_processes,
    }
    if resume_from is not None:
        resume_from = Path(resume_from).expanduser().resolve()
        saved = yaml.safe_load((resume_from / "metadata.yaml").read_text())
        #region Resume validation
        if saved["signature"] != signature:
            raise ValueError("Checkpoint model, data, optimizer, batch, or world-size settings differ.")
        if saved["step"] > config.num_steps:
            raise ValueError("num_steps is earlier than the checkpoint update.")
        #endregion

    # Reset initialization and pair RNGs for every K to keep comparisons paired.
    set_seed(config.seed)
    data = build_spiral(spiral_config)
    state = _TrainingState(config.seed)
    # Validation is repeatable and independent of the advancing training stream.
    validation_rng = torch.Generator(device="cpu").manual_seed((config.seed + 1) % 2**32)
    validation_batch = sample_pairs(data, scheme, config.validation_batch_size, validation_rng)
    validation_times = torch.rand(
        config.validation_batch_size, 1, generator=validation_rng,
        dtype=torch.float32, device="cpu",
    )
    reference = scheme.group_means(data.targets)[scheme.group_indices(validation_batch.target_indices)]
    model = ConditionalMLP(model_config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas,
        eps=config.adam_eps, weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(state)
    if resume_from is not None:
        accelerator.load_state(str(resume_from))

    logger = logging.getLogger(f"empirical_condition_density.train.rank{accelerator.process_index}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = []
    metrics_file = None
    session = datetime.now(timezone.utc).isoformat()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    initial_step = state.step

    def record(event, **values):
        if metrics_file is not None:
            metrics_file.write(json.dumps({"session": session, "event": event, **values}) + "\n")
            metrics_file.flush()

    def checkpoint():
        # All ranks save state; metadata marks a fully written checkpoint.
        folder = run_dir / f"checkpoint-{state.step:07d}"
        accelerator.save_state(str(folder))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            (folder / "metadata.yaml").write_text(yaml.safe_dump({
                "step": state.step, "signature": signature,
            }, sort_keys=False))
            logger.info("Checkpoint | step=%s | path=%s", state.step, folder)
        accelerator.wait_for_everyone()
        record("checkpoint", step=state.step, path=str(folder))
        return folder

    try:
        if accelerator.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "terminal.log")]
            for handler in handlers:
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            metrics_file = (run_dir / "metrics.jsonl").open("a")
            metadata = {
                "training": settings, "model": asdict(model_config),
                "spiral": asdict(spiral_config), "provenance": _provenance(accelerator),
                "micro_batch_size": micro_batch_size,
                "resume_from": str(resume_from) if resume_from is not None else None,
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
            packages = sorted({
                f"{package.metadata['Name']}=={package.version}"
                for package in distributions() if package.metadata.get("Name")
            })
            (run_dir / "environment.txt").write_text("\n".join(packages) + "\n")
            record("start", **metadata, step=state.step)
            logger.info(
                "Training | K=%s | device=%s | processes=%s | global batch=%s "
                "| per-device micro batch=%s | accumulation=%s | start=%s",
                config.num_conditions, accelerator.device, accelerator.num_processes,
                config.global_batch_size, micro_batch_size,
                config.gradient_accumulation_steps, state.step,
            )
        accelerator.wait_for_everyone()
        model.train()
        optimizer.zero_grad()
        last_checkpoint = resume_from
        while state.step < config.num_steps:
            batch = sample_pairs(data, scheme, config.global_batch_size, state.generator)
            times = torch.rand(
                config.global_batch_size, 1, generator=state.generator,
                dtype=torch.float32, device="cpu",
            )
            step_loss = torch.zeros((), dtype=torch.float32, device=accelerator.device)
            for micro in range(config.gradient_accumulation_steps):
                # Disjoint slices cover the same global batch at any GPU count.
                offset = (micro * accelerator.num_processes + accelerator.process_index)
                start = offset * micro_batch_size
                selection = slice(start, start + micro_batch_size)
                x0, x1, condition = _slice_batch(batch, selection, accelerator.device)
                with accelerator.accumulate(model), torch.autocast(
                    device_type=accelerator.device.type, enabled=False,
                ):
                    loss = flow_matching_loss(
                        model, x0, x1, times[selection].to(accelerator.device), condition,
                    )
                    step_loss += loss.detach() / config.gradient_accumulation_steps
                    # Accelerate scales accumulated gradients and synchronizes the last microbatch.
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # Measure the synchronized norm for diagnostics; do not modify gradients.
                        grad_norm = torch.linalg.vector_norm(torch.stack([
                            torch.linalg.vector_norm(p.grad.detach())
                            for p in model.parameters() if p.grad is not None
                        ]))
                        step_loss = accelerator.reduce(step_loss, reduction="mean")
                        #region Numerical validation
                        if not torch.isfinite(step_loss) or not torch.isfinite(grad_norm):
                            raise FloatingPointError("Non-finite training loss or gradient norm.")
                        #endregion
                    optimizer.step()
                    optimizer.zero_grad()
            state.step += 1
            elapsed = time.perf_counter() - started
            updates_per_second = (state.step - initial_step) / elapsed
            values = {
                "step": state.step, "loss": step_loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad_norm.item(), "elapsed_seconds": elapsed,
                "updates_per_second": updates_per_second,
                "samples_per_second": updates_per_second * config.global_batch_size,
                "eta_seconds": (config.num_steps - state.step) / updates_per_second,
            }
            record("train", **values)
            if accelerator.is_main_process and (
                state.step % config.log_interval == 0 or state.step == config.num_steps
            ):
                logger.info(
                    "Step %s/%s | loss=%.6f | lr=%.3g | updates/s=%.2f "
                    "| samples/s=%.0f | elapsed=%.1fs | ETA=%.1fs",
                    state.step, config.num_steps, values["loss"], values["learning_rate"],
                    updates_per_second, values["samples_per_second"], elapsed, values["eta_seconds"],
                )
            if state.step % config.validation_interval == 0 or state.step == config.num_steps:
                validation = _validate(model, validation_batch, validation_times, reference, accelerator)
                record("validation", step=state.step, **validation)
                if accelerator.is_main_process:
                    logger.info("Validation | step=%s | %s", state.step, validation)
            if state.step % config.checkpoint_interval == 0 or state.step == config.num_steps:
                last_checkpoint = checkpoint()
        accelerator.wait_for_everyone()
        record("complete", step=state.step)
        return TrainingResult(
            model=accelerator.unwrap_model(model), step=state.step,
            run_dir=run_dir, checkpoint=last_checkpoint,
        )
    except Exception:
        if accelerator.is_main_process:
            logger.exception("Training failed at update %s", state.step)
        raise
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        if metrics_file is not None:
            metrics_file.close()
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
