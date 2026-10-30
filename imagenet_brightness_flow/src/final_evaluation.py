"""Stream all validation sources with final EMA weights at each requested NFE."""

from dataclasses import asdict, replace
import logging
import sys
from types import SimpleNamespace

from accelerate import Accelerator
import torch
import yaml

from flow_matching.models import SiT
from .artifacts import file_sha256, read_json, write_json
from .comparison_config import timestep_coverage
from .data import build_datasets
from .filesystem import resolve_project_path
from .tracking import WandbTracker
from .training import provenance, run_signature
from .vae import FrozenVAE
from .validation import evaluate, render_previews, save_panels


def evaluation_signature(signature):
    """Final weights support inference with another device count or batch layout."""
    result = dict(signature)
    training = dict(result["training"])
    for name in ("device", "processes", "microbatch", "accumulation", "cpu_threads"):
        training.pop(name, None)
    result["training"] = training
    return result


def cell_complete(folder, protocol):
    """A cell is reusable only when its protocol and all fixed previews match."""
    path = folder / "metrics.json"
    if not path.is_file():
        return False
    report = read_json(path)
    #region Saved evaluation validation
    if report["protocol"] != protocol:
        raise ValueError(f"Evaluation settings or final weights changed: {folder}")
    #endregion
    panels = report.get("previews", [])
    if len(panels) != protocol["preview_count"]:
        return False
    for panel in panels:
        relative = panel["image"]
        path = folder / relative
        if not path.is_file():
            return False
    return True


def _open_logger(folder, main_process):
    """Give the main rank terminal and persistent final-evaluation records."""
    logger = logging.getLogger("brightness.final_evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = []
    if main_process:
        folder.mkdir(parents=True, exist_ok=True)
        terminal = logging.StreamHandler(sys.stdout)
        persistent = logging.FileHandler(folder / "terminal.log")
        handlers = [terminal, persistent]
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger, handlers


def evaluate_run(config, settings, *, datasets=None, model=None, vae=None, accelerator=None):
    """Load/check EMA weights, evaluate/reuse cells, then close logs and workers.

    Each GPU owns disjoint validation sources; metric sums/counts are combined
    after local batches. Only the main process writes shared reports/previews.
    Optional components support focused fixture checks.
    """
    torch.set_num_threads(config.training.cpu_threads)
    owns_accelerator = accelerator is None
    if owns_accelerator:
        accelerator = Accelerator(
            cpu=config.training.device == "cpu", mixed_precision=config.training.mixed_precision,
        )
    validation = replace(config.validation, batch_size=settings.batch_size, workers=settings.workers)
    config = replace(config, validation=validation)
    run_dir = resolve_project_path(config.training.output_dir)
    folder = run_dir / "final-evaluation"
    logger, handlers = _open_logger(folder, accelerator.is_main_process)
    tracker = None
    exit_code = 0
    try:
        #region Final model and dataset validation
        if accelerator.num_processes != config.training.processes:
            raise ValueError("Final evaluation process count differs from the configured launcher.")
        complete_path = run_dir / "complete.json"
        weights_path = run_dir / "averaged.pt"
        if not complete_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Final training artifacts are missing: {run_dir}")
        complete = read_json(complete_path)
        if complete["step"] != config.training.updates:
            raise ValueError("Final model update count differs from the configured training budget.")
        if datasets is None:
            datasets = build_datasets(config)
        if len(datasets.validation) != config.data.expected_validation_images:
            raise ValueError("Final evaluation must cover the full configured validation split.")
        #endregion

        # Load the final EMA model locally; inference needs no DDP gradient hooks.
        if model is None:
            latent_size = config.data.image_size // 8
            model = SiT(latent_size=latent_size)
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        signature = run_signature(config, datasets, model)
        signature = evaluation_signature(signature)
        saved_signature = evaluation_signature(weights["signature"])
        #region Final weight compatibility validation
        if signature != saved_signature or weights["step"] != complete["step"]:
            raise ValueError("EMA weights differ from the requested model/data/VAE/training settings.")
        #endregion
        model.load_state_dict(weights["model"])
        del weights
        model = model.to(accelerator.device)
        model.eval()
        if vae is None:
            vae = FrozenVAE.from_config(config.vae, accelerator.device)
        vae = vae.to(accelerator.device)

        # The protocol ties reused cells to exact weights and fixed source cases.
        weight_hash = file_sha256(weights_path)
        previews = []
        for case in datasets.index.previews:
            record = asdict(case)
            previews.append(record)
        metric_settings = asdict(config.metrics)
        preview_count = len(previews)
        protocol = {
            "weights_sha256": weight_hash, "signature": signature,
            "metrics": metric_settings, "exposures": config.data.evaluation_exposures,
            "previews": previews, "preview_count": preview_count,
        }
        if accelerator.is_main_process:
            resolved = asdict(config)
            resolved_text = yaml.safe_dump(resolved, sort_keys=False)
            settings_path = folder / "resolved.yaml"
            settings_path.write_text(resolved_text)
        accelerator.wait_for_everyone()

        #region Model validation and reporting
        reports = []
        for nfe in settings.nfe:
            cell = folder / f"nfe-{nfe}"
            cell_protocol = dict(protocol)
            cell_protocol["nfe"] = nfe
            reusable = cell_complete(cell, cell_protocol)
            if reusable and not settings.resume:
                raise FileExistsError(f"Final evaluation exists: {cell}; enable comparison.resume.")
            # All ranks inspect the old marker before any rank publishes a new one.
            accelerator.wait_for_everyone()
            if reusable:
                report = read_json(cell / "metrics.json")
                reports.append(report)
                if accelerator.is_main_process:
                    logger.info("Reusing final evaluation | timesteps=%s | NFE=%s", config.training.timesteps, nfe)
                continue

            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(accelerator.device)
            exposures = config.data.evaluation_exposures
            report = evaluate(
                model, vae, datasets.validation, exposures, config, accelerator,
                nfe=nfe, logger=logger, log_interval=settings.log_interval,
            )
            expected = len(datasets.validation) * len(exposures)
            if report["overall"]["comparisons"] != expected:
                raise ValueError("Final evaluation comparison count differs from the full split.")
            peak_memory = 0
            if accelerator.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(accelerator.device)
            memory = torch.tensor([peak_memory], dtype=torch.float64, device=accelerator.device)
            memory = accelerator.gather(memory)
            memory = memory.max()
            memory = memory.item()

            if accelerator.is_main_process:
                coverage = timestep_coverage(config.training.timesteps, nfe)
                report.update(coverage)
                report["training_timesteps"] = config.training.timesteps
                report["optimizer_update"] = complete["step"]
                report["protocol"] = cell_protocol
                report["provenance"] = provenance(accelerator)
                report["peak_memory_bytes"] = int(memory)
                panels = render_previews(
                    model, vae, datasets, config, accelerator.device, step=complete["step"], nfe=nfe,
                )
                preview_records = save_panels(cell, panels)
                report["previews"] = preview_records

                # Log before publishing completion so failed uploads can be retried.
                if config.wandb.enabled:
                    if tracker is None:
                        state = SimpleNamespace(wandb_run_id=None)
                        identity = read_json(run_dir / "wandb-run.json")
                        state.wandb_run_id = identity["run_id"]
                        # Preserve the training run's original config during final inference.
                        tracker = WandbTracker.start(config, state, run_dir, resumed=True, record_config=False)
                    processed_images = complete["step"] * config.training.global_batch
                    tracker.final_evaluation(complete["step"], processed_images, nfe, report, panels)
                metrics_path = cell / "metrics.json"
                write_json(metrics_path, report)
                logger.info(
                    "Final cell complete | timesteps=%s | NFE=%s | pairs=%s | RGB RMSE=%.6f",
                    config.training.timesteps, nfe, expected, report["overall"]["rgb_rmse"],
                )
            accelerator.wait_for_everyone()
            reports.append(report)
        return reports
        #endregion
    #region Error handling and cleanup
    except KeyboardInterrupt:
        exit_code = 130
        logger.info("Final evaluation interrupted | completed cells are retained.")
        raise
    except Exception:
        exit_code = 1
        logger.exception("Final evaluation failed")
        raise
    finally:
        if tracker is not None:
            tracker.finish(exit_code)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        if owns_accelerator:
            accelerator.end_training()
    #endregion
