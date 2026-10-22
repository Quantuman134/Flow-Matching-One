"""Accelerate training with online VAE pairs and resumable optimizer updates."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time

from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader
import yaml

from .brightness import apply_exposure
from .data import build_datasets
from flow_matching.losses import flow_matching_loss
from flow_matching.models import SiT
from flow_matching.interpolants import linear_path
from .filesystem import REPO_ROOT, resolve_project_path
from .state import MicrobatchSampler, TrainingState, make_ema, update_ema
from .vae import FrozenVAE


@dataclass(frozen=True)
class TrainingResult:
    model: torch.nn.Module
    averaged: torch.nn.Module
    step: int
    checkpoint: Path


def resume_path(value, run_dir):
    """Bare checkpoint names belong to their run; other relative paths use the repo."""
    path = Path(value).expanduser()
    if not path.is_absolute() and len(path.parts) == 1:
        path = run_dir / path
    else:
        path = resolve_project_path(path)
    return path


def model_signature(model):
    shapes = {}
    for name, parameter in model.state_dict().items():
        shapes[name] = list(parameter.shape)
    return shapes


def run_signature(config, datasets, model):
    """Require matching stochastic/data/optimizer settings for continuation."""
    training = asdict(config.training)
    for key in ("updates", "log_interval", "checkpoint_interval", "output_dir", "resume_from", "workers"):
        training.pop(key)
    data = asdict(config.data)
    data.pop("root")
    signature = {
        "seed": config.seed, "data": data, "vae": asdict(config.vae),
        "training": training, "model_shapes": model_signature(model),
        "fingerprints": datasets.index.fingerprints,
    }
    return signature


def provenance(accelerator):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    record = {
        "git_revision": revision.strip(), "git_dirty": bool(status.strip()),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "processes": accelerator.num_processes,
        "device": str(accelerator.device),
    }
    if accelerator.device.type == "cuda":
        record["gpu"] = torch.cuda.get_device_name(accelerator.device)
    return record


def save_checkpoint(accelerator, state, run_dir, signature):
    """Publish metadata/latest only after every rank finishes saving state."""
    folder = run_dir / f"checkpoint-{state.step:07d}"
    if (folder / "metadata.yaml").exists():
        raise FileExistsError(f"Complete checkpoint already exists: {folder}")
    accelerator.save_state(str(folder))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = {"step": state.step, "signature": signature}
        temporary = folder / "metadata.tmp"
        temporary.write_text(yaml.safe_dump(metadata, sort_keys=False))
        temporary.replace(folder / "metadata.yaml")
        latest = run_dir / "latest.tmp"
        latest.write_text(json.dumps({"checkpoint": folder.name, "step": state.step}) + "\n")
        latest.replace(run_dir / "latest.json")
    accelerator.wait_for_everyone()
    return folder


def _open_records(run_dir, config, datasets, accelerator):
    logger = logging.getLogger(f"brightness.train.rank{accelerator.process_index}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = []
    metrics = None
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "terminal.log")]
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        metadata = {
            "settings": asdict(config), "provenance": provenance(accelerator),
            "dataset_root": str(datasets.index.root), "classes": datasets.index.classes,
            "fingerprints": datasets.index.fingerprints, "previews": [],
        }
        for preview in datasets.index.previews:
            metadata["previews"].append(asdict(preview))
        if not (run_dir / "run.yaml").exists():
            (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        packages = []
        for package in distributions():
            name = package.metadata.get("Name")
            if name:
                packages.append(f"{name}=={package.version}")
        packages = sorted(set(packages))
        (run_dir / "environment.txt").write_text("\n".join(packages) + "\n")
        metrics = (run_dir / "metrics.jsonl").open("a")
        event = {"event": "start", "time": datetime.now(timezone.utc).isoformat(), **metadata}
        metrics.write(json.dumps(event) + "\n")
        metrics.flush()
    return logger, handlers, metrics


def train(config, *, datasets=None, vae=None, model=None):
    """Train one timestep setting; optional components support small fixture checks.

    Worker prefetch advances a private sample cursor. After each successful
    optimizer update, the committed cursor and all condition RNGs are saved.
    All ranks draw the same global conditions and take disjoint microbatches.
    """
    settings = config.training
    #region Runtime validation
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if config.data.image_size % 16:
        raise ValueError("SiT-S/2 images must have size divisible by 16.")
    #endregion
    torch.set_num_threads(settings.cpu_threads)
    accelerator = Accelerator(
        cpu=settings.device == "cpu", mixed_precision=settings.mixed_precision,
        gradient_accumulation_steps=settings.accumulation,
    )
    if accelerator.num_processes != settings.processes:
        raise ValueError("Launched process count differs from training.processes.")
    run_dir = resolve_project_path(settings.output_dir)
    checkpoint = None
    if settings.resume_from is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run folder is occupied: {run_dir}")
    else:
        checkpoint = resume_path(settings.resume_from, run_dir)
        if not (checkpoint / "metadata.yaml").is_file():
            raise ValueError("Resume requires a complete checkpoint with metadata.yaml.")

    set_seed(config.seed)
    if datasets is None:
        datasets = build_datasets(config)
    state = TrainingState(config.seed, len(datasets.training))
    if model is None:
        model = SiT(latent_size=config.data.image_size // 8)
    model = model.to(accelerator.device)
    averaged = make_ema(model)
    signature = run_signature(config, datasets, model)
    if checkpoint is not None:
        saved = yaml.safe_load((checkpoint / "metadata.yaml").read_text())
        if saved["signature"] != signature:
            raise ValueError("Checkpoint model, data, VAE, optimizer, or process settings differ.")
        if saved["step"] > settings.updates:
            raise ValueError("Requested updates precede the checkpoint step.")
    if vae is None:
        vae = FrozenVAE.from_config(config.vae, accelerator.device)
    vae = vae.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, betas=tuple(settings.adam_betas),
        weight_decay=settings.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(state, averaged)
    if checkpoint is not None:
        accelerator.load_state(str(checkpoint))

    sampler = MicrobatchSampler(state.order, settings, accelerator.process_index)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 5)
    loader = DataLoader(
        datasets.training, batch_sampler=sampler, num_workers=settings.workers,
        pin_memory=accelerator.device.type == "cuda", generator=loader_generator,
    )
    iterator = None
    logger, handlers, metrics = _open_records(run_dir, config, datasets, accelerator)
    started = time.perf_counter()
    initial_step = state.step
    window_started = started
    window_step = state.step
    window_loss = torch.zeros((), device=accelerator.device)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    try:
        accelerator.wait_for_everyone()
        logger.info(
            "Training | timesteps=%s | start=%s | global batch=%s = %s × %s × %s",
            settings.timesteps, state.step, settings.global_batch, settings.microbatch,
            settings.processes, settings.accumulation,
        )
        model.train()
        optimizer.zero_grad()
        iterator = iter(loader)
        while state.step < settings.updates:
            flips, exposure, times = state.draw_conditions(settings.global_batch, config.data, settings.timesteps)
            step_loss = torch.zeros((), device=accelerator.device)
            for micro in range(settings.accumulation):
                batch = next(iterator)
                sources = batch["image"].to(accelerator.device, non_blocking=True)
                start = (micro * settings.processes + accelerator.process_index) * settings.microbatch
                selection = slice(start, start + settings.microbatch)
                local_flips = flips[selection].to(accelerator.device)
                local_exposure = exposure[selection].to(accelerator.device)
                local_times = times[selection].to(accelerator.device)
                mirrored = sources.flip(-1)
                sources = torch.where(local_flips[:, None, None, None], mirrored, sources)
                reference = apply_exposure(sources, local_exposure)
                source_latent, reference_latent = vae.encode_pairs(sources, reference.image)

                with accelerator.accumulate(model):
                    state_at_time, target_velocity = linear_path(source_latent, reference_latent, local_times)
                    prediction = model(state_at_time, local_times, local_exposure)
                    loss = flow_matching_loss(prediction, target_velocity)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite velocity loss.")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        gradient_norms = []
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                norm = torch.linalg.vector_norm(parameter.grad.detach())
                                gradient_norms.append(norm)
                        gradient_norms = torch.stack(gradient_norms)
                        if not torch.isfinite(gradient_norms).all():
                            raise FloatingPointError("Non-finite model gradient.")
                    optimizer.step()
                    optimizer.zero_grad()
                step_loss += loss.detach() / settings.accumulation

            unwrapped = accelerator.unwrap_model(model)
            update_ema(averaged, unwrapped, settings.ema_decay)
            state.order.take(settings.global_batch)
            state.step += 1
            window_loss += step_loss
            finished = state.step == settings.updates
            if state.step % settings.log_interval == 0 or finished:
                mean_loss = window_loss / (state.step - window_step)
                mean_loss = accelerator.reduce(mean_loss, reduction="mean")
                now = time.perf_counter()
                elapsed = now - started
                speed = (state.step - window_step) * settings.global_batch / (now - window_started)
                remaining = settings.updates - state.step
                eta = remaining * elapsed / (state.step - initial_step)
                values = {
                    "event": "train", "step": state.step, "velocity_mse": mean_loss.item(),
                    "processed_images": state.step * settings.global_batch,
                    "learning_rate": settings.learning_rate, "images_per_second": speed,
                    "elapsed_seconds": elapsed, "eta_seconds": eta,
                }
                if accelerator.device.type == "cuda":
                    memory = torch.cuda.max_memory_allocated(accelerator.device)
                    values["peak_memory_bytes"] = memory
                if metrics is not None:
                    metrics.write(json.dumps(values) + "\n")
                    metrics.flush()
                logger.info(
                    "Update %s/%s | loss=%.6f | images/s=%.1f | elapsed=%.1fs | ETA=%.1fs",
                    state.step, settings.updates, values["velocity_mse"], speed, elapsed, eta,
                )
                window_loss.zero_()
                window_step = state.step
                window_started = now
            if state.step % settings.checkpoint_interval == 0 or finished:
                checkpoint = save_checkpoint(accelerator, state, run_dir, signature)
                logger.info("Checkpoint | update=%s | %s", state.step, checkpoint.name)

        if accelerator.is_main_process:
            weights = {"model": averaged.state_dict(), "signature": signature, "step": state.step}
            torch.save(weights, run_dir / "averaged.pt")
            completion = {"step": state.step, "checkpoint": checkpoint.name, "averaged": "averaged.pt"}
            (run_dir / "complete.json").write_text(json.dumps(completion) + "\n")
        accelerator.wait_for_everyone()
        result = TrainingResult(accelerator.unwrap_model(model), averaged, state.step, checkpoint)
        return result
    except KeyboardInterrupt:
        logger.info("Interrupted | resume from the latest complete checkpoint.")
        raise
    except Exception:
        logger.exception("Training failed at update %s", state.step)
        raise
    finally:
        if iterator is not None and settings.workers:
            # Stop prefetched workers immediately when training ends or is interrupted.
            iterator._shutdown_workers()
        if metrics is not None:
            metrics.close()
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        accelerator.end_training()
