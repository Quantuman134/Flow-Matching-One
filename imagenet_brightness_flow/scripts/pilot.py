"""Compare global batches at a matched image budget using YAML settings."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import yaml

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.filesystem import REPO_ROOT, resolve_project_path


def write_pilot_configs(config):
    root = resolve_project_path(config.pilot.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Pilot folder is occupied: {root}")
    partition = config.training.processes * config.training.microbatch
    for batch in config.pilot.global_batches:
        if batch % partition:
            raise ValueError("Each pilot batch must be divisible by processes × microbatch.")
    folder = root / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for batch in config.pilot.global_batches:
        updates = config.pilot.image_budget // batch
        run_dir = root / f"batch-{batch}"
        training = replace(
            config.training, global_batch=batch, accumulation=batch // partition,
            updates=updates, output_dir=str(run_dir), resume_from=None,
            checkpoint_interval=updates, log_interval=max(1, updates // 8),
        )
        validation = replace(config.validation, interval=updates)
        wandb = replace(config.wandb, group=root.name)
        job = replace(config, training=training, validation=validation, wandb=wandb)
        path = folder / f"batch-{batch}.yaml"
        path.write_text(yaml.safe_dump(asdict(job), sort_keys=False))
        jobs.append((path, run_dir))
    return root, jobs


def summarize_pilot(root, jobs):
    rows = []
    initialization_hashes = set()
    stream_hashes = set()
    for path, run_dir in jobs:
        config = load_config(path)
        performance = json.loads((run_dir / "performance.json").read_text())
        validation = None
        latest = run_dir / "validation-latest.json"
        if latest.exists():
            reference = json.loads(latest.read_text())["report"]
            validation = json.loads((run_dir / reference).read_text())["overall"]
        rows.append({
            "global_batch": config.training.global_batch, "updates": config.training.updates,
            "config": str(path.relative_to(root)), "run": str(run_dir.relative_to(root)),
            "performance": performance, "validation": validation,
        })
        initialization_hashes.add(performance["initial_model_sha256"])
        stream_hashes.add(performance["stream_sha256"])
    paired = len(initialization_hashes) == 1 and len(stream_hashes) == 1
    report = {"paired_initialization_and_streams": paired, "runs": rows}
    (root / "pilot-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not paired:
        raise ValueError("Pilot runs must share model initialization and consumed data/random streams.")
    return report


def run_pilot(config):
    root, jobs = write_pilot_configs(config)
    for path, run_dir in jobs:
        command = [sys.executable, "-u", "-m", "imagenet_brightness_flow.scripts.launch", str(path)]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    report = summarize_pilot(root, jobs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m imagenet_brightness_flow.scripts.pilot CONFIG.yaml")
    settings = load_config(Path(sys.argv[1]))
    run_pilot(settings)
