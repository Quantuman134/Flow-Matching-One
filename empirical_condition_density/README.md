# Empirical condition density

## Status

- Development level: **level-1**; human review after each step.
- Current step: **1 — Data and conditions**; model training and evaluation are pending.
- Review: Step 1 implemented and awaiting user review; Step 2 has not started.
- Setting: Gaussian prior → spiral targets.
- Plan: [Experiment 01](../documents/experiment-01-plan.html).

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Fixed spiral, condition encoding, group means, and fresh pair sampling. |
| `visualize.py` | Prior and analytic one-step reference figure. |

## Execution

- Data creation: build the fixed targets in memory when the experiment starts; reuse them across condition counts and sample fresh Gaussian noise for each batch.
- Configuration and launcher: add one experiment YAML configuration and Bash launcher after the full experiment code is assembled.

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

## Verification

- Passed: 50K unique targets, all seven group sizes, FP32 shapes, encoding examples, and exact K=1 / K=N group-mean references.
- Passed: normalization, repeatable pairs across K, fresh noise for repeated targets, Gaussian moments, invalid-input checks, artifact generation, and dependency consistency.
- Target mean: approximately **(0, 0)**; mean squared radius: **2.0000000002**; preparation and the preview were checked on CPU.
