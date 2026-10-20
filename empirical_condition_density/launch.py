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

from .config import load_config
from .training import REPO_ROOT


def launch(path):
    """Validate resources, then start run.py with the YAML-selected process count."""
    path = Path(path).expanduser().resolve()
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
    command += ["-m", "empirical_condition_density.run", "--config", str(path)]
    environment = os.environ.copy()
    # Use the same thread budget for PyTorch and CPU metric libraries.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(config.launch.cpu_threads)
    return subprocess.call(command, cwd=REPO_ROOT, env=environment)


if __name__ == "__main__":
    #region Argument validation
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m empirical_condition_density.launch CONFIG.yaml")
    #endregion
    raise SystemExit(launch(sys.argv[1]))
