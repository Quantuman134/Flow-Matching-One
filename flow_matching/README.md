# Flow matching toolbox

## Status

- Development level: **1**, following the selected experiment review level.
- Extraction: **reviewed; approved**.
- Consumers: [empirical condition density](../empirical_condition_density/README.md) and [ImageNet brightness](../imagenet_brightness_flow/README.md).

## Components

| Module | Responsibility |
| --- | --- |
| `models/mlp.py` | Conditional vector MLP; configurable state/condition dimensions. |
| `models/sit.py` | SiT-S/2 with a scalar condition and four latent velocity channels. |
| `interpolants.py` | Linear interpolation and target displacement. |
| `losses.py` | Individual loss terms and the combined flow-matching objective. |
| `timesteps.py` | Uniform continuous times or finite left-endpoint grids. |
| `samplers.py` | Euler integration, NFE counts, and optional trajectories. |
| `tests/` | Analytical checks, extraction comparisons, and a small spiral workflow check. |

Experiment folders own dataset loading, source/target pairs, VAE handling, configuration, training orchestration, and evaluation metrics. Run Python imports from the repository root or use the experiment launchers. Runtime dependency: the existing PyTorch installation.

## Interface

| Entry | Convention |
| --- | --- |
| `model(state, times, condition)` | Return velocity with the same shape as `state`. |
| State / endpoints | Matching `[B,...]` tensors; MLP uses `[B,D]`, SiT uses `[B,4,S,S]`. |
| Times | FP32 `[B]`, with 0 at the source and 1 at the target. |
| Conditions | MLP uses `[B,C]`; SiT uses `[B,1]`. Each experiment defines their meaning. |
| `sample_timesteps` | `setting="continuous"` or any positive integer K; grid values are `0, 1/K, …, (K−1)/K`. Every setting consumes B uniform draws. |
| `velocity_mse(prediction, target_velocity)` | Individual velocity error term. `reduction="mean"` averages all squared errors; `"sum_per_sample"` sums features before averaging samples. |
| `flow_matching_loss(prediction, target_velocity)` | Combines loss terms; currently uses velocity MSE. The caller constructs the path and runs the model. |
| `linear_path(source, target, times)` | Returns `source + t × (target−source)` and target velocity `target−source`. |
| `euler_sample` | Any positive integer `nfe`; returns `SamplingResult`. |
| `SamplingResult` | `.samples`, `.num_evaluations`, and optional `.trajectory` with shape `[nfe+1,B,...]`. |
| Precision | FP32 interpolation, loss, and state updates; optional BF16 model autocast. The sampler restores model mode and any scoped matmul precision. |

Model parameter names and default initialization are retained for checkpoint loading. SiT's scalar embedding retains its existing `exposure` parameter names.

## Use

The experiment supplies latent endpoints, scalar conditions, and a caller-owned `torch.Generator`:

```python
from flow_matching.losses import flow_matching_loss
from flow_matching.models import SiT
from flow_matching.interpolants import linear_path
from flow_matching.samplers import euler_sample
from flow_matching.timesteps import sample_timesteps

model = SiT()
model = model.to(source.device)
times = sample_timesteps(len(source), "continuous", generator, source.device)
state, target_velocity = linear_path(source, target, times)
prediction = model(state, times, condition)
loss = flow_matching_loss(prediction, target_velocity)
result = euler_sample(model, source, condition, nfe=4)
generated = result.samples
```

Each experiment selects its loss reduction and sampler matmul precision:

| Experiment | Reduction | Sampler matmul precision |
| --- | --- | --- |
| Empirical condition density | `sum_per_sample` | `highest` |
| ImageNet brightness | `mean` | Caller's setting |

Experiments import the models and flow operations they use directly from this package.

## Verification

```bash
python -m unittest discover -s flow_matching/tests -v
python -m unittest discover -s imagenet_brightness_flow/tests -v
```

Extraction comparisons use local pre-refactor source snapshots under `.verification/refactor/before/`; they are skipped when these snapshots are absent. Other checks run from a fresh checkout using temporary fixtures. Local logs: `.verification/refactor/`.

Verified: nine toolbox/extraction/workflow checks and fourteen brightness CPU checks; real-image SiT/VAE learning and two-GPU launch, relocated resume, and sampling also passed during extraction. Original-code comparisons match brightness exactly and spiral within FP32 tolerances after unifying interpolation arithmetic. Checkpoint continuation under the current code matches uninterrupted training exactly.

Architecture attribution: [third-party notices](THIRD_PARTY_NOTICES.md). Validation history: [development log](../documents/development-log.md).
