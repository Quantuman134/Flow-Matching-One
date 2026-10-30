"""Sequential training/evaluation jobs with portable records and checkpoint resume."""

from dataclasses import asdict, replace
import logging
import os
import signal
import subprocess
import sys
import time

import yaml

from .artifacts import read_json, write_json
from .comparison_config import comparison_protocol
from .filesystem import REPO_ROOT, resolve_project_path


def latest_checkpoint(run_dir):
    """Only published metadata marks a checkpoint that can be resumed."""
    latest = None
    latest_step = -1
    for path in run_dir.glob("checkpoint-*/metadata.yaml"):
        contents = path.read_text()
        metadata = yaml.safe_load(contents)
        if metadata["step"] > latest_step:
            latest_step = metadata["step"]
            latest = path.parent.name
    return latest


def training_complete(run_dir, updates):
    """Recognize final weights published at the requested optimizer update."""
    marker = run_dir / "complete.json"
    if not marker.is_file():
        return False
    record = read_json(marker)
    #region Completion validation
    if record["step"] != updates or record["averaged"] != "averaged.pt":
        raise ValueError(f"Completed training budget/artifacts differ: {run_dir}")
    weights = run_dir / "averaged.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Completed training weights are missing: {weights}")
    #endregion
    return True


def prepare_comparison(settings, config):
    """Pin one protocol, then write one derived training YAML per timestep setting."""
    root = resolve_project_path(settings.output_dir)
    manifest = root / "comparison.json"
    protocol = comparison_protocol(settings, config)
    #region Comparison output validation
    if manifest.is_file():
        existing = read_json(manifest)
        if not settings.resume:
            raise FileExistsError(f"Comparison exists: {root}; enable comparison.resume.")
        if existing["protocol"] != protocol:
            raise ValueError("Comparison settings differ; choose another comparison.output_dir.")
    elif root.exists():
        entries = root.iterdir()
        occupied = any(entries)
        if occupied:
            raise FileExistsError(f"Comparison folder has no matching manifest: {root}")
    if settings.mode in ("evaluate", "report") and not manifest.is_file():
        raise FileNotFoundError("Evaluation/report mode requires an existing comparison manifest.")
    if config.wandb.enabled and config.wandb.mode == "online" and settings.mode != "report":
        if not os.environ.get(config.wandb.api_key_env):
            raise RuntimeError(f"Set {config.wandb.api_key_env} before online training/evaluation.")
    #endregion

    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        write_json(manifest, {"protocol": protocol})
    resolved = asdict(settings)
    resolved["base_config"] = "base.yaml"
    resolved["output_dir"] = "."
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    (root / "resolved-comparison.yaml").write_text(resolved_text)
    base = asdict(config)
    base_text = yaml.safe_dump(base, sort_keys=False)
    (root / "base.yaml").write_text(base_text)

    jobs = []
    for timesteps in settings.timesteps:
        name = f"timesteps-{timesteps}"
        run_dir = root / "runs" / name
        resume_from = None
        ready = training_complete(run_dir, config.training.updates)
        if not ready and settings.mode in ("all", "train"):
            resume_from = latest_checkpoint(run_dir)
            #region Interrupted run validation
            if resume_from is None and run_dir.exists():
                entries = run_dir.iterdir()
                occupied = any(entries)
                if occupied:
                    raise ValueError(f"No complete checkpoint is available in the interrupted run: {run_dir}")
            #endregion
        output_dir = str(run_dir)
        training = replace(
            config.training, timesteps=timesteps, output_dir=output_dir, resume_from=resume_from,
        )
        job = replace(config, training=training)
        path = configs / f"{name}.yaml"
        job_values = asdict(job)
        job_text = yaml.safe_dump(job_values, sort_keys=False)
        path.write_text(job_text)
        jobs.append((path, job))
    return root, jobs


def _terminate_requested(signum, frame):
    """Let the supervisor forward SIGTERM and wait for its active worker job."""
    raise SystemExit(128 + signum)


def run_worker(config_path, comparison_path=None):
    """Forward cancellation to torchrun and wait until it reaps its workers."""
    config_argument = str(config_path)
    command = [sys.executable, "-u", "-m", "imagenet_brightness_flow.scripts.launch", config_argument]
    if comparison_path is not None:
        comparison_argument = str(comparison_path)
        command.extend(["--evaluate", comparison_argument])

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_interrupt = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _terminate_requested)
    child = None
    try:
        # A separate session receives signals only through this supervisor.
        child = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
        returncode = child.wait()
    #region Error handling and cleanup
    except BaseException:
        # Repeated Ctrl+C must not kill torchrun before its own cleanup completes.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_interrupt)
        signal.signal(signal.SIGTERM, previous_term)
    #endregion
    #region Worker exit validation
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    #endregion


def run_comparison(settings, config, comparison_path):
    """Prepare jobs; train/evaluate sequentially; export the final comparison.

    Completed models and evaluation cells are reused. Incomplete training starts
    from its latest complete checkpoint; incomplete evaluation repeats that cell.
    """
    from .reporting import export_comparison

    root, jobs = prepare_comparison(settings, config)
    logger = logging.getLogger("brightness.comparison")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    terminal = logging.StreamHandler(sys.stdout)
    persistent = logging.FileHandler(root / "terminal.log")
    handlers = [terminal, persistent]
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    started = time.perf_counter()
    model_count = len(jobs)
    try:
        if settings.mode != "report":
            for index, (path, job) in enumerate(jobs, start=1):
                run_dir = resolve_project_path(job.training.output_dir)
                ready = training_complete(run_dir, job.training.updates)
                if settings.mode in ("all", "train") and not ready:
                    logger.info("Training job | timesteps=%s | resume=%s", job.training.timesteps, job.training.resume_from)
                    run_worker(path)
                    ready = training_complete(run_dir, job.training.updates)
                #region Final training artifact validation
                if not ready:
                    raise FileNotFoundError(f"Final training artifacts are missing: {run_dir}")
                #endregion
                if settings.mode in ("all", "evaluate"):
                    logger.info("Final evaluation job | timesteps=%s", job.training.timesteps)
                    run_worker(path, comparison_path)
                elapsed = time.perf_counter() - started
                remaining = model_count - index
                eta = elapsed * remaining / index
                logger.info("Comparison progress | models=%s/%s | elapsed=%.1fs | ETA=%.1fs", index, model_count, elapsed, eta)

        if settings.mode != "train":
            export_comparison(root, settings, jobs)
        elapsed = time.perf_counter() - started
        marker = root / f"{settings.mode}-complete.json"
        completion = {"models": model_count, "elapsed_seconds": elapsed}
        write_json(marker, completion)
        logger.info("Comparison complete | mode=%s | output=%s", settings.mode, root)
    #region Error handling and cleanup
    except (KeyboardInterrupt, SystemExit):
        logger.info("Comparison interrupted | completed checkpoints/cells are retained.")
        raise
    except Exception:
        logger.exception("Comparison failed")
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    #endregion
