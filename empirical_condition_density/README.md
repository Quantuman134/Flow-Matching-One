# Empirical condition density

## Status

- Development level: **level-1**; human review after each step.
- Current step: **4 — Full comparison**; implementation is next.
- Progress: Steps 1–2 approved; Step 3 (pilot) skipped by user request; comparison metrics are pending.
- Setting: Gaussian prior → spiral targets.
- Plan: [Experiment 01](../documents/experiment-01-plan.html).

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Fixed spiral, condition encoding, group means, and fresh pair sampling. |
| `visualize.py` | Prior and analytic one-step reference figure. |
| `model.py` | Conditional FP32 MLP and model settings. |
| `flow.py` | Linear interpolation and velocity MSE loss. |
| `sampling.py` | Euler inference with optional trajectory output. |
| `training.py` | Callable trainer, validation, run records, and checkpoint resume. |

## Execution

- Data creation: build the fixed targets in memory when the experiment starts; reuse them across condition counts and sample fresh Gaussian noise for each batch.
- Configuration and launcher: add one experiment YAML configuration and Bash launcher after the full experiment code is assembled.
- Current interface: Python functions and dataclasses; training automatically writes resolved settings and provenance as run records.

## Data and conditions

- Target: three-turn spiral; samples: **50K**; radius before normalization: **0.2 → 2.0**.
- Normalization: zero mean; mean squared distance from the origin: **2**; target coordinates: **FP32**.
- Conditions K: **1, 10, 100, 500, 2,500, 10,000, 50,000**; targets per condition: **N/K**.
- Encoding: integer group index `j` becomes one float, `c = 2(j + 0.5)/K - 1`; condition tensor: **[batch size, 1]**.
- Pair sampling: targets sampled uniformly with replacement; fresh independent Gaussian noise on every call, using a caller-provided CPU `torch.Generator`.

## Review artifact

- Preview: `outputs/empirical_condition_density/data/preview.png`; retained locally from Step 1 verification and ignored by Git.

## Sampling interface

```python
import torch
from empirical_condition_density.data import ConditionScheme, build_spiral, sample_pairs

data = build_spiral()
scheme = ConditionScheme(num_targets=50_000, num_conditions=100)
generator = torch.Generator().manual_seed(0)
batch = sample_pairs(data, scheme, batch_size=512, generator=generator)
```

- `batch.x0` / `batch.x1`: **[512, 2]**, FP32; `batch.condition`: **[512, 1]**, FP32.
- Recreate the generator with the same seed to reproduce pairs across K; keep advancing a generator to obtain fresh pairs.

## Model and objective

| Setting | Value |
| --- | --- |
| Inputs | Position `[B,2]`, time `[B,1]`, condition `[B,1]`; FP32, same device. |
| MLP | Four hidden layers; width 128; SiLU; two velocity outputs; 50,434 parameters. |
| Time | Independent uniform samples in `[0,1)` for each training pair. |
| Path | `x_t = (1-t)x0 + t*x1`; target velocity: `x1-x0`. |
| Loss | Squared Euclidean velocity error, summed over two coordinates and averaged over samples. |
| Defaults | Adam; learning rate 0.001; global batch 512; 20K updates; gradient norm clipped at 1. |
| Precision | FP32 with autocast and TF32 disabled during training and sampling. |

## Training and inference interface

```python
from empirical_condition_density.training import TrainConfig, train
from empirical_condition_density.sampling import sample_euler

result = train(TrainConfig(
    num_conditions=100,
    num_steps=3,  # Short interface check; the default is 20,000 updates.
    device="cpu",
    output_dir="outputs/empirical_condition_density/manual_check",
))
device = next(result.model.parameters()).device
generated = sample_euler(
    result.model, batch.x0.to(device), batch.condition.to(device),
    num_steps=1, return_trajectory=True,
)
```

- `result`: model, completed update count, run directory, and final checkpoint path; `batch` is from the sampling example above.
- `generated.samples`: `[B,2]`; `generated.trajectory`: `[S+1,B,2]` when requested; model evaluations: exactly `S`.
- Sampling: time grid `t=k/S`, `k=0,…,S-1`; the supplied noise and model's original train/eval mode are preserved.
- GPU count: call `train(...)` on every rank under `torchrun`; Accelerate places the model on each rank's device, and the final experiment runner will own process-group teardown.
- Batch partition: `global batch = per-device micro batch × process count × accumulation`; validation batch size must also divide evenly across processes.
- Pair consistency: every rank generates the same global batch, then takes a disjoint slice; initialization, target indices, noise, and times are shared across K for the same seed.

## Run records and resume

- Directory: `<output_dir>/K-<count>/seed-<seed>/`; an existing run requires explicit resume or a different output directory.
- Files: `run.yaml`, `environment.txt`, `metrics.jsonl`, `terminal.log`, and `checkpoint-<update>/`; checkpoint state includes model, optimizer, completed update, and RNG streams.
- Validation: fixed noise/time pairs from a separate RNG; record velocity MSE and one-step error against the exact group mean.
- Resume: pass `resume_from=<checkpoint>` to `train(...)`; model, data, optimizer settings, global batch, accumulation, and process count must match, while total updates may increase.

## Verification

- Passed: 50K unique targets, all seven group sizes, FP32 shapes, encoding examples, and exact K=1 / K=N group-mean references.
- Passed: normalization, repeatable pairs across K, fresh noise for repeated targets, Gaussian moments, invalid-input checks, artifact generation, and dependency consistency.
- Target mean: approximately **(0, 0)**; mean squared radius: **2.0000000002**; preparation and the preview were checked on CPU.
- Step 2 core: architecture, interpolation endpoints, loss normalization, learning on a fixed batch, and exact-target Euler checks for 1–64 steps passed.
- Training: short CPU, one-GPU, and two-GPU runs passed; two-GPU parameters stayed synchronized, with global batch fixed at 512.
- Resume: CPU continuation matched uninterrupted parameters, losses, and gradient norms exactly; two-GPU continuation also passed.
- Guards: invalid shapes/dtypes/settings, incompatible resume, run overwrite, non-finite sampling, precision restoration, and explicit FP32 under a changed default dtype passed.
- Evidence: `outputs/empirical_condition_density/step2_checks/`; these are short verification runs, not pilot results.
