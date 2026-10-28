"""Train/evaluate the YAML sweep; only the main rank writes shared artifacts."""

import argparse
import gc
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

# Direct file execution needs the repository on Python's import path.
if __package__ is None or __package__ == "":
    script_path = Path(__file__)
    script_path = script_path.resolve()
    repository_root = script_path.parents[2]
    repository_path = str(repository_root)
    sys.path.insert(0, repository_path)

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from safetensors.torch import load_file, save_file

from empirical_condition_density.src.config import load_config
from empirical_condition_density.src.data import ConditionScheme, build_spiral
from empirical_condition_density.src.evaluation import evaluate_model, evaluation_complete, save_arrays, write_json
from empirical_condition_density.src.metrics import GeometryMetrics
from flow_matching.models import MLP
from empirical_condition_density.src.filesystem import REPO_ROOT
from empirical_condition_density.src.reporting import export_figures, export_tables
from empirical_condition_density.src.training import train


def _checkpoint(run_dir):
    """Ignore interrupted checkpoints that have no completion metadata."""
    choices = []
    for path in run_dir.glob("checkpoint-*/metadata.yaml"):
        metadata = yaml.safe_load(path.read_text())
        choices.append((metadata["step"], path.parent))
    return max(choices, default=(0, None))[1]


def _validate_overwrite(root, config_path):
    """Limit recursive replacement to one experiment or verification run folder."""
    topic_root = REPO_ROOT / "empirical_condition_density"
    allowed_parents = (topic_root / "outputs", topic_root / ".verification")
    allowed = False
    for parent in allowed_parents:
        if root.is_relative_to(parent) and root != parent:
            allowed = True

    settings_path = Path(config_path)
    settings_path = settings_path.resolve()

    #region Output replacement validation
    if not allowed:
        raise ValueError("Overwrite requires a run folder inside this topic's outputs/ or .verification/.")
    if settings_path.is_relative_to(root):
        raise ValueError("Keep the input YAML outside the output folder when overwriting.")
    if root.exists() and not root.is_dir():
        raise ValueError(f"Output path is not a directory: {root}")
    #endregion


def run(config_path):
    """Reuse one target set, train each K, then evaluate final weights on rank zero."""
    config = load_config(config_path)
    torch.set_num_threads(config.launch.cpu_threads)
    accelerator = Accelerator(
        cpu=config.launch.device == "cpu", mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    root = config.output_root
    manifest = root / "experiment.yaml"
    logger = logging.getLogger("empirical_condition_density.experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = []
    started = time.perf_counter()
    try:
        #region Run validation
        if accelerator.num_processes != config.launch.num_processes:
            raise ValueError("Process count differs from YAML; use run.sh to launch the experiment.")
        if config.launch.device == "cuda" and accelerator.device.type != "cuda":
            raise RuntimeError("CUDA was requested but is unavailable.")
        if config.overwrite:
            _validate_overwrite(root, config_path)
        elif manifest.exists():
            saved = yaml.safe_load(manifest.read_text())
            if not config.resume:
                raise FileExistsError(
                    f"Experiment exists: {root}; set overwrite: true, resume: true, or change output_dir."
                )
            if saved["protocol"] != config.protocol():
                raise ValueError("Existing experiment settings differ; use another output_dir.")
        elif any(root.glob("K-*/seed-*/run.yaml")):
            raise FileExistsError("Untagged training runs exist; choose a new experiment output_dir.")
        elif config.mode == "evaluate":
            raise FileNotFoundError("Evaluation requires an existing experiment with trained models.")
        #endregion
        # All ranks finish validation before rank zero replaces shared outputs.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            replaced = config.overwrite and root.exists()
            if replaced:
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            (root / "resolved.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
            if not manifest.exists():
                temporary = manifest.with_suffix(".yaml.tmp")
                temporary.write_text(yaml.safe_dump({"protocol": config.protocol()}, sort_keys=False))
                temporary.replace(manifest)
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(root / "experiment.log")]
            for handler in handlers:
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            if replaced:
                logger.info("Overwrite | replaced previous experiment output | output=%s", root)
            logger.info("Sweep | mode=%s | runs=%s | evaluations=%s | output=%s", config.mode,
                        len(config.condition_counts),
                        len(config.condition_counts) * len(config.evaluation.steps), root)
        accelerator.wait_for_everyone()
        data = build_spiral(config.data)
        geometry = None
        if config.mode != "train" and accelerator.is_main_process:
            # Target distances and fixed projections are shared by every comparison.
            geometry = GeometryMetrics(data.targets.numpy(), config.evaluation.num_directions,
                                       config.evaluation.direction_seed)
            save_arrays(root / "reference.npz", targets=data.targets.numpy(),
                        parameters=data.parameters.numpy(), center=data.center.numpy(),
                        scale=np.asarray(data.scale), directions=geometry.directions)
        rows = []
        total_runs = len(config.condition_counts)
        completed = 0
        seed = config.seed
        # Sweep only K; each trainer receives identical initialization and data RNG seeds.
        for count in config.condition_counts:
            run_dir = root / f"K-{count:05d}" / f"seed-{seed}"
            weights = run_dir / "final-model.safetensors"
            complete = run_dir / "completed.json"
            ready = weights.exists() and complete.exists()
            result = None
            #region Final model validation
            if config.mode == "evaluate" and not ready:
                raise FileNotFoundError(f"Final trained model is missing: {run_dir}")
            #endregion
            if config.mode != "evaluate" and not ready:
                checkpoint = _checkpoint(run_dir) if config.resume else None
                result = train(config.train_config(count), model_config=config.model,
                               spiral_config=config.data, resume_from=checkpoint)
                if accelerator.is_main_process:
                    save_file({k: v.detach().cpu().contiguous() for k, v in result.model.state_dict().items()},
                              str(weights))
                    # A run-local checkpoint name stays valid when results are moved.
                    checkpoint_name = result.checkpoint.name
                    completion_record = {"step": result.step, "checkpoint": checkpoint_name}
                    write_json(complete, completion_record)
                accelerator.wait_for_everyone()
            elif accelerator.is_main_process:
                logger.info("Reusing final model | K=%s | seed=%s", count, seed)
            if config.mode != "train" and accelerator.is_main_process:
                # Evaluation needs no distributed collectives; other ranks wait below.
                folder = run_dir / "evaluation"
                if config.mode != "evaluate" and config.resume and evaluation_complete(
                    folder, config.evaluation,
                ):
                    rows.extend(json.loads((folder / "metrics.json").read_text()))
                    logger.info("Reusing evaluation | K=%s | seed=%s", count, seed)
                else:
                    model = MLP(config.model).to(accelerator.device)
                    model.load_state_dict(load_file(str(weights), device=str(accelerator.device)))
                    rows.extend(evaluate_model(
                        model, data, ConditionScheme(len(data.targets), count),
                        config.evaluation, seed, run_dir, geometry, logger,
                    ))
                    del model
                export_tables(root, rows)
            accelerator.wait_for_everyone()
            del result
            gc.collect()
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
            completed += 1
            if accelerator.is_main_process:
                elapsed = time.perf_counter() - started
                logger.info("Sweep progress | runs=%s/%s | elapsed=%.1fs | ETA=%.1fs",
                            completed, total_runs, elapsed, elapsed * (total_runs - completed) / completed)
        if config.mode != "train" and accelerator.is_main_process:
            summary = export_tables(root, rows)
            export_figures(config, data.targets.numpy(), summary)
        if accelerator.is_main_process:
            write_json(root / f"{config.mode}-complete.json", {
                "training_runs": total_runs, "main_evaluations": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
            })
            logger.info("Complete | mode=%s | elapsed=%.1fs", config.mode, time.perf_counter() - started)
        accelerator.wait_for_everyone()
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("Interrupted | stopping workers; completed checkpoints are retained.")
        raise
    except Exception:
        if accelerator.is_main_process:
            logger.exception("Experiment failed")
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments.config)
    except KeyboardInterrupt:
        raise SystemExit(130)
