# Environment

## Current cluster

- Checked **2026-10-30**: the user-created `flow-matching-one` environment is installed under the cluster's shared Miniconda installation; Python 3.11.13, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, Accelerate 1.10.1, diffusers 0.35.1, and W&B 0.31.0 import successfully.
- NumPy 2.2.6 and Pillow 11.3.0 match the planned pins. `pip check` passed; the pinned brightness VAE configuration and safetensors weights are cached.
- Resolved snapshots: `local/flow-matching-one/environment.yml` and `requirements.lock.txt`, ignored by Git. The configured ImageNet index passed the expected split/class counts; GPU operation requires verification on the allocated compute node.

## Earlier host

- Name: `flow-matching-one`; platform: Linux x86_64; host: four NVIDIA H100 NVL GPUs; driver: **615.71.09**.
- Status: experiment dependencies installed; the full baseline is only partially installed; `pip check` passed.

| Component | Observed version |
| --- | --- |
| Python | 3.11.13 |
| PyTorch | 2.9.1+cu128 |
| torchvision / Pillow | 0.24.1+cu128 / 12.3.0 |
| NumPy / SciPy | 2.4.6 / 1.15.3 |
| Matplotlib / PyYAML | 3.10.6 / 6.0.2 |
| Accelerate | 1.10.1 |
| huggingface-hub / safetensors | 0.34.4 / 0.6.2 |
| diffusers | 0.35.1 |
| importlib_metadata / regex / zipp | 9.0.1 / 2026.9.29 / 4.1.2 |
| gpustat | 1.1.1 |
| W&B | 0.31.0 |

- NumPy: installed **2.4.6**, planned **2.2.6**; no downgrade performed; SciPy accepts the installed version.
- Observed snapshots: `local/flow-matching-one/environment.yml` and `requirements.lock.txt`; ignored by Git.
- Brightness Step 1 observed torchvision/Pillow versions above and passed CPU/CUDA data checks; planned Pillow remains 11.3.0 in the baseline pins. Verification evidence records the tested versions.
- Brightness Step 2: the user approved installing diffusers and its missing dependencies. The versions above passed `pip check`, cached-VAE encoding, SiT-S learning/sampling, and two-GPU training/resume checks. Installation report: `imagenet_brightness_flow/.verification/step2_checks/install-report.json`.
- Brightness Step 3: user-approved W&B installation passed `pip check`, offline logging, and resumed run-ID checks. Package list and installation record: [W&B environment review](wandb-install-review.md).

## Planned baseline

- Python / pip: **3.11.13 / 25.2**; PyTorch / torchvision: **2.9.1 / 0.24.1**.
- Compute builds: **cu128** default; **cu126** and **cpu** selectable; direct dependency pins: [requirements.txt](requirements.txt).
- CUDA wheels supply the runtime; the host needs a compatible driver; custom CUDA builds additionally need a local toolkit.
- Setup profiles cover Linux with Bash/Conda; other platforms need separate profiles.
- Brightness tracking settings: [experiment plan](../experiment-02-brightness/plan.html#wandb-logging).

## Setup

```bash
bash scripts/setup_env.sh
conda activate flow-matching-one
# Alternative: bash scripts/setup_env.sh flow-matching-one-cpu cpu
# Alternative: bash scripts/setup_env.sh flow-matching-one-cu126 cu126
```

- Creates a new environment, installs pins, and validates imports plus one training step; existing environments are left untouched.
- CUDA setup requires a visible GPU; choose the CPU profile on hosts without NVIDIA GPUs.

## Full baseline check

```bash
python scripts/check_env.py --cpu
# CUDA; change visible devices and process count together.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/check_env.py --require-cuda
```

- The full check imports every baseline dependency; install remaining packages before running it in the partially installed environment.
- Distributed checks cover training, communication, and synchronized parameters; implementation-specific checks belong in the [development log](../development-log.md).

## Reproduce

- Successful setup saves Conda/package exports and host information under `local/<environment-name>/`; archive these with experiment outputs.
- Recreate on a matching platform with the recorded compute index:

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    conda env create --name <new-name> --file <snapshot>/environment.yml
```

- Remove the exported `prefix` when moving machines; snapshots preserve resolved versions, not package artifacts or identical results across hardware.
- Run provenance rules: [coding requirements](../coding_requirements.md#logging-and-resume); setup and package-change history: [development log](../development-log.md).

## References

- [PyTorch build/version commands](https://pytorch.org/get-started/previous-versions/).
- [Accelerate quicktour](https://huggingface.co/docs/accelerate/quicktour) and [process launching](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch).
