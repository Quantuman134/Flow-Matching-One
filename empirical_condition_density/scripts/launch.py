"""Launch the experiment using YAML settings.

1. Validate device availability, process count, and batch divisibility.
2. Choose direct Python execution or multi-process torchrun.
3. Set CPU thread limits and working directory; return the exit status.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

# Direct file execution needs the repository on Python's import path.
if __package__ is None or __package__ == "":
    script_path = Path(__file__)
    script_path = script_path.resolve()
    repository_root = script_path.parents[2]
    repository_path = str(repository_root)
    sys.path.insert(0, repository_path)

from empirical_condition_density.src.config import load_config
from empirical_condition_density.src.filesystem import REPO_ROOT


def launch(path):
    """Validate resources, then start run.py with the YAML-selected process count."""
    path = Path(path)
    path = path.expanduser()
    path = path.resolve()
    config = load_config(path)
    processes = config.launch.num_processes
    use_cuda = config.launch.device != "cpu" and torch.cuda.is_available()
    training = config.train_config(config.condition_counts[0])
    #region Launch validation
    if config.launch.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable.")
    if use_cuda and processes > torch.cuda.device_count():
        raise ValueError("num_processes exceeds the number of visible GPUs.")
    if training.global_batch_size % (processes * training.gradient_accumulation_steps):
        raise ValueError("global_batch_size must divide across processes and accumulation.")
    if training.validation_batch_size % processes:
        raise ValueError("validation_batch_size must divide across processes.")
    #endregion
    command = [sys.executable, "-u"]
    # torchrun assigns ranks and local devices; a single process needs no wrapper.
    if processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc_per_node={processes}"]
    config_path = str(path)
    command += ["-m", "empirical_condition_density.scripts.run", "--config", config_path]
    environment = os.environ.copy()
    # Use the same thread budget for PyTorch and CPU metric libraries.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(config.launch.cpu_threads)
    exit_code = subprocess.call(command, cwd=REPO_ROOT, env=environment)
    return exit_code


if __name__ == "__main__":
    #region Argument validation
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m empirical_condition_density.scripts.launch CONFIG.yaml")
    #endregion
    config_path = sys.argv[1]
    exit_code = launch(config_path)
    raise SystemExit(exit_code)
