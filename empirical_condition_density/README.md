# Empirical condition density

## Status

- Level: **1**; Steps 1–2 approved; Step 3 skipped; revised Step 4 awaits review.
- Full-budget run: completed; seven trained models and 28 main evaluations; results: `outputs/comparison/` within this folder.
- Design and metric definitions: [experiment plan](../documents/experiment-01-plan.html).

## Run

```bash
bash empirical_condition_density/scripts/run.sh [CONFIG.yaml]
```

- Config: [configs/experiment.yaml](configs/experiment.yaml); optional path is relative to the caller; relative outputs resolve from the repository root.
- Training: **50K** targets; **seven** condition counts; **seed 0**; **20K** updates per model; Adam; **no gradient clipping**.
- Inference: Euler steps **1, 4, 16, 64**; **seven** training runs and **28** main evaluations.
- Checkpoints: every **50K** updates plus completion; the current 20K-update budget saves the final checkpoint.
- Devices: `launch.device` selects auto/cpu/cuda; `num_processes` selects GPU count; global batch stays fixed; evaluation uses the main process.

## Code map

```text
empirical_condition_density/
├── src/             # Reusable Python components
├── scripts/         # Run and launch entry points
├── configs/         # YAML settings
├── outputs/         # Experiment results
└── .verification/   # Development checks
```

| File | Responsibility |
| --- | --- |
| `src/data.py` / `src/visualize.py` | Fixed in-memory spiral, conditions, fresh pairs, and analytic reference figure. |
| `src/model.py` / `src/flow.py` / `src/sampling.py` | Conditional MLP, linear-path velocity loss, and Euler integration. |
| `src/training.py` | FP32 Accelerate training, fixed validation, records, and resume. |
| `src/config.py` | Load and validate YAML into dataclasses; Python defaults are fallbacks. |
| `scripts/run.sh` | Select the Conda environment and call `scripts/launch.py`. |
| `scripts/launch.py` | Validate resources and start `scripts/run.py` directly or through torchrun. |
| `scripts/run.py` | Train or reuse each K model, evaluate final weights, and export results. |
| `src/metrics.py` | Nearest-neighbor metrics and fixed-direction sliced Wasserstein-1. |
| `src/evaluation.py` | Schedule sampling, metrics, spread, and coordinate exports. |
| `src/reporting.py` | Single-seed tables, scatter/metric figures, loss curves, and trajectories. |

## Evaluation flow

1. Build balanced conditions and fresh Gaussian noise; reuse noise across K and step counts.
2. Generate **50K** samples per step count; calculate geometric precision/recall, conditional precision, and sliced Wasserstein-1; measure group-mean RMSE at one step.
3. Measure within-condition spread with extra noise draws at training conditions.
4. Save coordinates, metrics, trajectories, timing, and environment records; plots show direct single-seed values.

## Outputs and resume

- Root: `empirical_condition_density/outputs/comparison/` by default; resolved settings, experiment manifest/log, reference targets/projections, and metric/summary JSON and CSV files.
- Models: `K-<count>/seed-<seed>/`; training records, checkpoints, `final-model.safetensors`, and a completion marker.
- Evaluation: each model's `evaluation/`; inputs, outputs by step count, spread samples, trajectories, metrics, and actual environment records; figures: root `figures/`.
- Modes: `all` trains and evaluates; `train` trains only; `evaluate` recomputes evaluation from final weights.
- Resume: set `resume: true`; reuse completed work and resume incomplete training from the latest complete checkpoint; incompatible settings require a new output directory.
- Portability: incomplete training requires the original process count and accumulation; final weights can be evaluated on another GPU count or CPU.

## Verification

- Earlier checks: data geometry, exact metric/velocity references, CPU/two-GPU workflows, and checkpoint recovery; details are in the [development log](../documents/development-log.md).
- Revised checks: single-seed CPU/two-GPU workflows and plots passed; Adam gradients remain unmodified, and checkpoint continuation is exact.
- Local evidence: `empirical_condition_density/.verification/`; includes data previews, implementation checks, and review checks.
