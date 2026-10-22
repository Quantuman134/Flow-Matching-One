# ImageNet brightness flow

## Status

- Development level: **1**, selected by the user; review each implementation step before the next.
- Step 1: **reviewed; approved to proceed**. Steps 2–4 are pending.
- Design and metric definitions: [experiment plan](../documents/experiment-02-brightness/plan.html).

## Runtime data

Settings: [configs/experiment.yaml](configs/experiment.yaml). Dataset paths resolve from the repository root, with `~` expanded.

| Stage | API | Work |
| --- | --- | --- |
| Training startup | `build_datasets(config)` | Index raw filenames in memory, validate counts/classes, and select fixed monitor/preview cases. |
| Source access | `ImageNetSources[index]` | Decode RGB, apply EXIF orientation, resize/crop, and return FP32 sRGB. |
| Training batch | `sample_pairs(...)` | Sample shared horizontal flips and fresh per-image exposure, then construct targets in memory. |
| Validation | `data.validation`, `data.monitor`, `data.preview_sources` | Use all 50,000 validation images, the 1,000-image monitor subset, or 16 fixed preview cases. |

The runtime index exposes relative paths, class mapping, ordered-path fingerprints, and preview conditions for training-run records. Training and evaluation entry points are part of Step 2.

## Components

| Module | Responsibility |
| --- | --- |
| `src/indexing.py` | In-memory source indexes, fixed monitor/preview selection, and fingerprints. |
| `src/data.py` | EXIF/RGB preprocessing, source dataset, and shared horizontal flips with independently sampled exposure. |
| `src/brightness.py` | FP32 sRGB/linear-light conversion, exposure adjustment, and reference saturation masks. |
| `src/metrics.py` | Per-image RGB, raw-output, luminance, latent, and masked errors; exact luminance decompositions. |
| `src/config.py` / `src/paths.py` | Strict YAML settings and repository-relative paths. |
| `tests/test_data_and_metrics.py` | Focused tests using temporary image fixtures. |

`sample_pairs` receives separate caller-owned augmentation and exposure generators. `image_errors` receives raw generated RGB, reference RGB, the configured variance threshold, and an optional saturation mask; it returns per-image tensors, including MSEs and mask counts.

For pooled RMSE, average per-image MSE and take one square root. Use `gain_valid` to restrict all three gain-fit components to matching images; nearly uniform references retain the offset/centered decomposition. Masked squared-error sums and value counts support saturation summaries; empty masks have count zero and NaN per-image MSE.

## Verification

```bash
python -m unittest discover -s imagenet_brightness_flow/tests -v
```

- Eight CPU checks cover known exposure values, identity, clipping, RNG independence, EXIF/RGB preprocessing, deterministic indexing, lazy loading, and analytic error decompositions.
- Runtime indexing retains 1,281,167 training and all 50,000 validation images, with the monitor subset included in final validation.
- Sixteen actual preview sources passed preprocessing and perfect-reference metric checks on CPU; exposure pairs and metrics also passed a lightweight CUDA check.
- Evidence: `.verification/step1_checks/`; validation history: [development log](../documents/development-log.md).
