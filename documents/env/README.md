# Environment

## Status

- Environment: `flow-matching-one`; Step 1 CPU data/plotting passed on 2026-10-14, and Step 2 CPU, one-GPU, and two-GPU training checks passed on 2026-10-15.
- Added for Step 2: Accelerate 1.10.1, huggingface-hub 0.34.4, and safetensors 0.6.2; the experiment's current imports work, while the remaining baseline packages have not all been installed or checked.
- Observed host: Linux x86_64, four NVIDIA H100 NVL GPUs, NVIDIA driver 615.71.09, and Conda available on PATH.

## Observed versions

| Component | Installed version |
| --- | --- |
| Python | 3.11.13 |
| PyTorch | 2.9.1+cu128 |
| NumPy | 2.4.6 |
| Matplotlib | 3.10.6 |
| PyYAML | 6.0.2 |
| Accelerate | 1.10.1 |
| huggingface-hub | 0.34.4 |
| safetensors | 0.6.2 |

- NumPy differs from the planned 2.2.6 pin; Step 1 passed with the installed version, and the environment was not downgraded.
- `pip check` passed; the observed package snapshot is stored in `local/flow-matching-one/` and the data artifact metadata records the versions used.

## Baseline

| Component | Selection |
| --- | --- |
| Platform | Linux x86_64 with Bash and Conda |
| Environment name | `flow-matching-one` by default |
| Python / pip | 3.11.13 / 25.2 |
| PyTorch / torchvision | 2.9.1 / 0.24.1 |
| Compute build | CUDA 12.8 (`cu128`) by default; `cu126` and `cpu` selectable |
| Distributed training | Hugging Face Accelerate |
| Remaining dependencies | Exact direct versions in [requirements.txt](requirements.txt) |

- This is an initial pinned baseline, not a claim to use the newest releases; project-specific dependencies will follow the research requirements.
- PyTorch wheels supply the CUDA runtime; the host needs a compatible NVIDIA driver, while a local CUDA toolkit is needed only for custom CUDA builds.
- Other operating systems and GPU backends need separate installation profiles; GPU-count portability remains a requirement for training code.

## Setup

From the repository root:

```bash
bash scripts/setup_env.sh
conda activate flow-matching-one
```

Alternative builds or names:

```bash
bash scripts/setup_env.sh flow-matching-one-cpu cpu
bash scripts/setup_env.sh flow-matching-one-cu126 cu126
```

- The script creates a new environment, installs pinned direct dependencies, runs `pip check`, and checks package imports and one training step.
- CUDA profiles require a working, visible GPU during setup; use the CPU profile for machines without NVIDIA GPUs.
- Existing environments are left untouched; if installation fails, inspect the error and remove the incomplete environment or choose a new name before retrying.

## Rename the existing environment

Leave `flow-matching` before renaming it:

```bash
conda activate base
conda rename --name flow-matching flow-matching-one
conda activate flow-matching-one
```

- Renaming preserves the environment's current packages; it does not complete an interrupted installation.

## Validation and GPU count

After activating the environment:

```bash
# CPU
python scripts/check_env.py --cpu

# One GPU
CUDA_VISIBLE_DEVICES=0 python scripts/check_env.py --require-cuda

# Two GPUs; change the visible device list and nproc_per_node together.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/check_env.py --require-cuda
```

- The multi-GPU check validates a training step, collective communication, and synchronized parameters; it uses Accelerate inside the script and `torchrun` to select the process count.
- These commands validate the complete baseline environment; Step 2 training was checked separately using the packages actually needed by its implementation.

## Reproducibility records

- Successful setup writes a resolved Conda export, `pip freeze`, and host/build information to `local/<environment-name>/`; these machine-specific records are ignored by Git.
- Direct dependencies are pinned; transitive dependencies are resolved at install time, so archive the generated records alongside each experiment for the exact package set.
- Recreate a matching-platform environment with `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 conda env create --name <new-name> --file <snapshot>/environment.yml`; select the recorded compute build instead of `cu128` when different.
- Remove the exported `prefix` entry when transferring machines; Conda exports and pip freezes record versions but do not provide artifact hashes or guarantee future package availability.
- Package snapshots do not guarantee identical results across hardware; experiments must also record configuration, seeds, dataset/model revisions, GPU count, and global batch size.

## References

- [Official PyTorch version/build commands](https://pytorch.org/get-started/previous-versions/) provide the selected PyTorch/torchvision pairing and CUDA/CPU indexes.
- [Accelerate quicktour](https://huggingface.co/docs/accelerate/quicktour) describes portable device placement and distributed training; [launching scripts](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch) covers process launch options.
