"""Launch the configured number of workers, preserving torchrun signal handling."""

import argparse
import os
import sys

import torch

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.filesystem import REPO_ROOT, resolve_project_path


def launch(path, evaluation=None):
    """Replace this process with training or final-evaluation workers."""
    path = resolve_project_path(path)
    config = load_config(path)
    settings = config.training
    #region Device validation
    use_cuda = settings.device == "cuda" or (settings.device == "auto" and torch.cuda.is_available())
    if use_cuda and torch.cuda.device_count() < settings.processes:
        raise ValueError("Visible GPUs must cover training.processes.")
    if not use_cuda and settings.processes != 1:
        raise ValueError("The CPU launcher supports one process for lightweight work.")
    #endregion
    command = [sys.executable]
    if settings.processes > 1:
        command.extend([
            "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={settings.processes}",
        ])
    config_path = str(path)
    module = "imagenet_brightness_flow.scripts.train"
    if evaluation is not None:
        module = "imagenet_brightness_flow.scripts.evaluate"
    command.extend(["-m", module, "--config", config_path])
    if evaluation is not None:
        evaluation_path = resolve_project_path(evaluation)
        evaluation_path = str(evaluation_path)
        command.extend(["--comparison", evaluation_path])
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(settings.cpu_threads)
    os.chdir(REPO_ROOT)
    os.execvpe(sys.executable, command, environment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--evaluate", metavar="COMPARISON.yaml")
    arguments = parser.parse_args()
    launch(arguments.config, arguments.evaluate)
