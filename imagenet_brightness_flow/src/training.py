"""Train brightness-conditioned flow matching with online VAE pairs.

The workflow is setup/resume -> optimizer updates -> validation/checkpoints ->
final EMA weights and resource records. Helpers manage run identity and files;
``train()`` coordinates model computation, data order, and distributed workers.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import hashlib
import logging
import os
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
from .tracking import WandbTracker
from .validation import evaluate, render_previews, save_validation


@dataclass(frozen=True)
class TrainingResult:
    """Return the trained model, its EMA copy, and the final checkpoint location."""

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
    """Describe parameter/buffer shapes for checkpoint compatibility checks."""
    shapes = {}
    for name, parameter in model.state_dict().items():
        shapes[name] = list(parameter.shape)
    return shapes


def _model_digest(model):
    """Fingerprint initial weights so paired pilot runs can verify initialization."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        values = value.detach().cpu().contiguous().numpy().tobytes()
        digest.update(values)
    return digest.hexdigest()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    """Record the code revision and runtime/device identity for this run."""
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
    #region Checkpoint output validation
    if (folder / "metadata.yaml").exists():
        raise FileExistsError(f"Complete checkpoint already exists: {folder}")
    #endregion

    # Every rank saves its state before the main rank marks the checkpoint complete.
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
    # Wait for the main rank to publish metadata and the latest-checkpoint pointer.
    accelerator.wait_for_everyone()
    return folder


def _open_records(run_dir, config, datasets, accelerator):
    """Open main-rank logs and record settings, dataset identity, and packages."""
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

        # Keep the original run description; append a new start event on resume.
        metadata = {
            "settings": asdict(config), "provenance": provenance(accelerator),
            "dataset_root": str(datasets.index.root), "classes": datasets.index.classes,
            "fingerprints": datasets.index.fingerprints, "previews": [],
        }
        for preview in datasets.index.previews:
            metadata["previews"].append(asdict(preview))
        if not (run_dir / "run.yaml").exists():
            (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))

        # Store installed package versions beside the run for reproducibility.
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
    """Train one timestep setting and return model/EMA weights plus checkpoint.

    Stages: runtime/resume setup; data/model initialization; optimizer updates;
    periodic logging, EMA validation, and checkpoints; final artifacts and cleanup.
    Optional datasets/VAE/model inputs support small fixture checks.

    Worker prefetch advances a private sample cursor. After each successful
    update, the trainer commits the data cursor; checkpoints save it and the RNGs.
    All ranks draw the same global conditions and take disjoint microbatches.
    """
    settings = config.training
    #region Runtime validation
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if config.data.image_size % 16:
        raise ValueError("SiT-S/2 images must have size divisible by 16.")
    if config.wandb.enabled and config.wandb.mode == "online":
        if not os.environ.get(config.wandb.api_key_env):
            raise RuntimeError(f"Set {config.wandb.api_key_env} for online W&B logging.")
    #endregion

    # Configure this worker's device, precision, and gradient accumulation.
    torch.set_num_threads(settings.cpu_threads)
    accelerator = Accelerator(
        cpu=settings.device == "cpu", mixed_precision=settings.mixed_precision,
        gradient_accumulation_steps=settings.accumulation,
    )
    #region Process count validation
    if accelerator.num_processes != settings.processes:
        raise ValueError("Launched process count differs from training.processes.")
    #endregion

    # Select a fresh output folder or an existing, fully written resume checkpoint.
    run_dir = resolve_project_path(settings.output_dir)
    checkpoint = None
    #region Output and checkpoint validation
    if settings.resume_from is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run folder is occupied: {run_dir}")
    else:
        checkpoint = resume_path(settings.resume_from, run_dir)
        if not (checkpoint / "metadata.yaml").is_file():
            raise ValueError("Resume requires a complete checkpoint with metadata.yaml.")
    #endregion

    # Initialize paired weights and independent data/condition RNG streams.
    # Dataset construction indexes paths; workers decode raw images when requested.
    set_seed(config.seed)
    if datasets is None:
        datasets = build_datasets(config)
    state = TrainingState(config.seed, len(datasets.training))

    # Keep a trainable SiT and a separate FP32 exponential moving average (EMA).
    if model is None:
        model = SiT(latent_size=config.data.image_size // 8)
    initial_model_digest = _model_digest(model)
    model = model.to(accelerator.device)
    averaged = make_ema(model)

    # Compare run settings, dataset fingerprints, and model shapes before restoring.
    signature = run_signature(config, datasets, model)
    #region Resume compatibility validation
    if checkpoint is not None:
        saved = yaml.safe_load((checkpoint / "metadata.yaml").read_text())
        if saved["signature"] != signature:
            raise ValueError("Checkpoint model, data, VAE, optimizer, or process settings differ.")
        if saved["step"] > settings.updates:
            raise ValueError("Requested updates precede the checkpoint step.")
    #endregion

    # The frozen VAE supplies FP32 endpoints; AdamW updates the flow model parameters.
    if vae is None:
        vae = FrozenVAE.from_config(config.vae, accelerator.device)
    vae = vae.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, betas=tuple(settings.adam_betas),
        weight_decay=settings.weight_decay,
    )

    # Wrap forward/optimizer execution, then restore model, optimizer, RNGs, and EMA.
    # Custom checkpoint objects add the committed cursor/condition RNGs and EMA.
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(state, averaged)
    if checkpoint is not None:
        accelerator.load_state(str(checkpoint))

    # Prefetch uses a private data cursor; the trainer advances the committed cursor.
    sampler = MicrobatchSampler(state.order, settings, accelerator.process_index)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 5)
    loader = DataLoader(
        datasets.training, batch_sampler=sampler, num_workers=settings.workers,
        pin_memory=accelerator.device.type == "cuda", generator=loader_generator,
        multiprocessing_context="spawn" if settings.workers else None,
    )
    iterator = None

    # Open persistent records and initialize session/window resource counters.
    logger, handlers, metrics = _open_records(run_dir, config, datasets, accelerator)
    started = time.perf_counter()
    initial_step = state.step
    window_started = started
    window_step = state.step
    window_loss = torch.zeros((), device=accelerator.device)
    training_seconds = 0.0
    measured_seconds = 0.0
    measured_images = 0
    validation_seconds = 0.0
    tracker = None
    exit_code = 0
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    try:
        # Shared logs and W&B belong to the main rank; all ranks start together.
        if config.wandb.enabled and accelerator.is_main_process:
            tracker = WandbTracker.start(config, state, run_dir, resumed=checkpoint is not None)
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
            # One loop iteration completes one optimizer update (one global batch).
            _synchronize(accelerator.device)
            update_started = time.perf_counter()
            # Each rank draws the same flips, exposure [B,1], and times [B].
            flips, exposure, times = state.draw_conditions(settings.global_batch, config.data, settings.timesteps)
            step_loss = torch.zeros((), device=accelerator.device)

            for micro in range(settings.accumulation):
                # Global ordering is accumulation slot -> rank -> microbatch.
                # Match this rank's image indices with the same condition slice.
                batch = next(iterator)
                sources = batch["image"].to(accelerator.device, non_blocking=True)
                start = (micro * settings.processes + accelerator.process_index) * settings.microbatch
                selection = slice(start, start + settings.microbatch)
                local_flips = flips[selection].to(accelerator.device)
                local_exposure = exposure[selection].to(accelerator.device)
                local_times = times[selection].to(accelerator.device)

                # Flip the source first, then create its paired brightness reference.
                # RGB is [b,3,H,W]; VAE endpoints are [b,4,H/8,W/8].
                mirrored = sources.flip(-1)
                sources = torch.where(local_flips[:, None, None, None], mirrored, sources)
                reference = apply_exposure(sources, local_exposure)
                source_latent, reference_latent = vae.encode_pairs(sources, reference.image)

                with accelerator.accumulate(model):
                    # Interpolate paired latents; predict their target-source velocity.
                    # Forward uses configured autocast; path/loss calculations use FP32.
                    state_at_time, target_velocity = linear_path(source_latent, reference_latent, local_times)
                    prediction = model(state_at_time, local_times, local_exposure)
                    loss = flow_matching_loss(prediction, target_velocity)

                    #region Loss validation
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite velocity loss.")
                    #endregion

                    # Backward scales each microbatch loss for gradient accumulation.
                    accelerator.backward(loss)

                    #region Gradient validation
                    if accelerator.sync_gradients:
                        # Check the accumulated gradient before the AdamW update.
                        gradient_norms = []
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                norm = torch.linalg.vector_norm(parameter.grad.detach())
                                gradient_norms.append(norm)
                        gradient_norms = torch.stack(gradient_norms)
                        if not torch.isfinite(gradient_norms).all():
                            raise FloatingPointError("Non-finite model gradient.")
                    #endregion

                    # The prepared optimizer applies/clears gradients at the boundary.
                    optimizer.step()
                    optimizer.zero_grad()
                step_loss += loss.detach() / settings.accumulation

            # Update EMA once and commit consumed images after the full optimizer update.
            unwrapped = accelerator.unwrap_model(model)
            update_ema(averaged, unwrapped, settings.ema_decay)
            state.order.take(settings.global_batch)
            state.step += 1

            # Measure update work; throughput timing uses an image-count warmup.
            _synchronize(accelerator.device)
            update_seconds = time.perf_counter() - update_started
            training_seconds += update_seconds
            if (state.step - 1) * settings.global_batch >= config.pilot.warmup_images:
                measured_seconds += update_seconds
                measured_images += settings.global_batch
            window_loss += step_loss
            finished = state.step == settings.updates

            # Average loss over the logging window and GPUs, then record progress.
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
                if tracker is not None:
                    tracker.training(state.step, state.step * settings.global_batch, values)
                logger.info(
                    "Update %s/%s | loss=%.6f | images/s=%.1f | elapsed=%.1fs | ETA=%.1fs",
                    state.step, settings.updates, values["velocity_mse"], speed, elapsed, eta,
                )
                window_loss.zero_()
                window_step = state.step
                window_started = now

            #region Model validation and reporting
            # Periodic/final validation uses EMA; the saved update prevents duplicates.
            validation_due = finished or state.step % config.validation.interval == 0
            if config.validation.enabled and validation_due and state.validation_step != state.step:
                validation_started = time.perf_counter()
                logger.info("Validation | update=%s | monitor images=%s", state.step, len(datasets.monitor))
                # Every rank evaluates its monitor partition and combines metric sums.
                report = evaluate(
                    averaged, vae, datasets.monitor, config.validation.exposures,
                    config, accelerator, nfe=config.validation.nfe,
                )
                if accelerator.is_main_process:
                    # Save/log the shared 16-panel preview set and validation report.
                    panels = render_previews(
                        averaged, vae, datasets, config, accelerator.device,
                        step=state.step, nfe=config.validation.nfe,
                    )
                    report = save_validation(run_dir, state.step, report, panels)
                    if tracker is not None:
                        tracker.validation(state.step, state.step * settings.global_batch, report, panels)
                    event = {"event": "validation", "step": state.step, **report["overall"]}
                    metrics.write(json.dumps(event, allow_nan=False) + "\n")
                    metrics.flush()
                    logger.info("Validation | update=%s | RGB RMSE=%.6f", state.step, report["overall"]["rgb_rmse"])

                # Mark validation complete after all shared writes finish.
                accelerator.wait_for_everyone()
                state.validation_step = state.step
                validation_seconds += time.perf_counter() - validation_started
            #endregion

            # Save after validation so its update marker and W&B run ID are included.
            if state.step % settings.checkpoint_interval == 0 or finished:
                checkpoint = save_checkpoint(accelerator, state, run_dir, signature)
                logger.info("Checkpoint | update=%s | %s", state.step, checkpoint.name)

        # Summarize the slowest-rank timings and largest allocated GPU memory.
        _synchronize(accelerator.device)
        elapsed = time.perf_counter() - started
        peak_memory = 0
        if accelerator.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(accelerator.device)
        resources = torch.tensor(
            [[training_seconds, measured_seconds, elapsed, validation_seconds, peak_memory]],
            dtype=torch.float64, device=accelerator.device,
        )
        resources = accelerator.gather(resources)
        resources = resources.max(dim=0).values.cpu().tolist()
        if accelerator.is_main_process:
            # Record resource measurements and paired initialization/RNG fingerprints.
            performance = {
                "step": state.step, "processed_images": state.step * settings.global_batch,
                "session_processed_images": (state.step - initial_step) * settings.global_batch,
                "training_seconds": resources[0], "measured_training_seconds": resources[1],
                "measured_images": measured_images, "wall_seconds": resources[2],
                "validation_seconds": resources[3], "peak_memory_bytes": int(resources[4]),
                "initial_model_sha256": initial_model_digest, "stream_sha256": state.stream_digest(),
                "training_images_per_second": measured_images / resources[1] if measured_images else None,
            }
            (run_dir / "performance.json").write_text(json.dumps(performance, indent=2) + "\n")

            # Export EMA weights for sampling and publish run-relative artifact names.
            weights = {"model": averaged.state_dict(), "signature": signature, "step": state.step}
            torch.save(weights, run_dir / "averaged.pt")
            completion = {"step": state.step, "checkpoint": checkpoint.name, "averaged": "averaged.pt"}
            (run_dir / "complete.json").write_text(json.dumps(completion) + "\n")
        accelerator.wait_for_everyone()
        result = TrainingResult(accelerator.unwrap_model(model), averaged, state.step, checkpoint)
        return result
    #region Error handling and cleanup
    except KeyboardInterrupt:
        exit_code = 130
        logger.info("Interrupted | resume from the latest complete checkpoint.")
        raise
    except Exception:
        exit_code = 1
        logger.exception("Training failed at update %s", state.step)
        raise
    finally:
        # Release prefetched workers, W&B, log handles, and distributed resources.
        if iterator is not None and settings.workers:
            # Stop prefetched workers immediately when training ends or is interrupted.
            iterator._shutdown_workers()
        if tracker is not None:
            tracker.finish(exit_code)
        if metrics is not None:
            metrics.close()
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        accelerator.end_training()
    #endregion
