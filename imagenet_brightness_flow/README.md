# ImageNet brightness flow

## Status

- Development level: **1**, selected by the user; review each implementation step before the next.
- Step 1: **reviewed; approved to proceed**.
- Step 2: **reviewed; approved to proceed**, committed as `ceb2647`.
- Step 3: **reviewed; approved to proceed**; offline logging, distributed validation, and resource pilot verified. Online authentication awaits account configuration.
- Step 4: full training comparison and final evaluation pending.
- Design and metric definitions: [experiment plan](../documents/experiment-02-brightness/plan.html).

## Runtime data

Settings: [configs/experiment.yaml](configs/experiment.yaml). Dataset paths resolve from the repository root, with `~` expanded.

| Stage | API | Work |
| --- | --- | --- |
| Training startup | `build_datasets(config)` | Index raw filenames in memory, validate counts/classes, and select fixed monitor/preview cases. |
| Source access | `ImageNetSources[index]` | Decode RGB, apply EXIF orientation, resize/crop, and return FP32 sRGB. |
| Training batch | `sample_pairs(...)` | Sample shared horizontal flips and fresh per-image exposure, then construct targets in memory. |
| Validation | `data.validation`, `data.monitor`, `data.preview_sources` | Use all 50,000 validation images, the 1,000-image monitor subset, or 16 fixed preview cases. |

The runtime index exposes relative paths, class mapping, ordered-path fingerprints, and preview conditions for training-run records.

## Training and sampling

Edit `training.timesteps` (`continuous`, `16`, `4`, or `1`) and choose a separate `training.output_dir` for each run. The launcher reads the Conda environment, GPU process count, microbatch, and accumulation from YAML.

```bash
bash imagenet_brightness_flow/scripts/train.sh
# Alternate settings; paths containing spaces are supported.
bash imagenet_brightness_flow/scripts/train.sh /absolute/path/settings.yaml
```

Full training uses SiT-S/2 from the [shared flow matching toolbox](../flow_matching/README.md) and the cached, revision-pinned VAE specified in YAML. VAE encoding uses posterior means in FP32; the transformer uses configured autocast precision. Separate initialization, data order, augmentation, exposure, and timestep streams keep the four settings paired.

| Entry | Meaning |
| --- | --- |
| `global_batch` | `microbatch × processes × accumulation`; enforced at startup. |
| `updates` | Completed optimizer updates. |
| `resume_from` | `null` for a fresh run; a bare checkpoint folder name resumes within `output_dir`. |
| `vae.local_files_only` | Load the pinned weights from the local Hugging Face cache. |

Dataset/output paths and resume paths containing directories resolve from the repository root; `~` expands. The Bash launcher's configuration argument resolves from the caller's directory. Resume requires matching model, data, VAE, optimizer, precision, and batch/process settings; the update budget may increase. Prefetched data uses a private cursor, while checkpoints store the committed cursor.

Sample one image with final EMA weights:

```bash
python -m imagenet_brightness_flow.scripts.sample \
  --config imagenet_brightness_flow/configs/experiment.yaml \
  --weights imagenet_brightness_flow/outputs/continuous-seed0/averaged.pt \
  --source /absolute/path/source.JPEG \
  --output imagenet_brightness_flow/outputs/continuous-seed0/example.png \
  --exposure 0.5 --nfe 4
```

## Tracking and validation

Set `wandb.entity` in YAML to your team slug or personal username; its inline comment gives an example. Supply your API key through `WANDB_API_KEY` in the launch environment. `wandb.mode: offline` saves local W&B records for development.

Validation uses EMA weights and the fixed 1,000-source monitor at −1, 0, and +1 EV with NFE 1. Each event saves 16 source/reference/generated panels locally and logs those panels to W&B. FP64 sums/counts combine disjoint distributed partitions; reported RMSE takes the square root after pooling squared errors. The full 50,000-source comparison belongs to Step 4.

Latency comes from these validation passes, including their initial calls. GPU synchronization marks the encoding, integration, and decoding boundaries. Reports show encoding milliseconds per source and integration/decoding milliseconds per generated image; `generation_ms_per_image` includes the source encoding shared across exposures. Times and image counts are summed across GPUs before division; reference construction and metric calculations have separate timing boundaries.

`evaluate()` groups loading, source encoding, reference construction, flow integration, decoding, metric accumulation, and reporting into commented blocks. `_summarize_timings()` combines device timings and builds the latency report.

| Settings | Purpose |
| --- | --- |
| `validation.interval`, `batch_size`, `workers` | Monitor frequency and per-process loading; completion also triggers validation. |
| `wandb.project`, `entity`, `group`, `mode` | Destination and online/offline logging. |
| `pilot.global_batches`, `image_budget`, `warmup_images` | Equal-image resource comparison and throughput warmup. |

Run the resource pilot:

```bash
bash imagenet_brightness_flow/scripts/train.sh --pilot
```

Append a YAML path for custom settings. The pilot runs batches 256 and 1024 sequentially, with the same initial weights and consumed image/augmentation/condition streams. Each run sees 65,536 images; throughput timing excludes the first 2,048. The summary includes training throughput, peak allocated GPU memory, and monitor metrics.

Observed on four H100 NVL GPUs, with microbatch 64, online FP32 VAE encoding, BF16 SiT-S, and offline W&B:

| Global batch | Updates | Training images/s | Peak allocated GiB/GPU | Monitor RGB RMSE [0,1] |
| --- | --- | --- | --- | --- |
| 256 | 256 | 628.3 | 6.05 | 0.13871 |
| 1024 | 64 | 629.3 | 6.05 | 0.13892 |

Both runs matched initialization/consumed-stream hashes, evaluated 3,000 monitor comparisons, and saved 16 panels. Batch 1024 is feasible; measured throughput and memory are similar. Use these short-run RMSEs as implementation diagnostics: the runs have different optimizer/EMA update counts. The proposed full-run batch remains 1024, pending review. Evidence: `.verification/step3_checks/resource-pilot/pilot-summary.json`.

## Outputs

| Artifact | Content |
| --- | --- |
| `run.yaml`, `environment.txt` | Resolved settings, Git/device/package records, dataset fingerprints, class mapping, and fixed preview cases. |
| `terminal.log`, `metrics.jsonl` | Loss, learning rate, processed images, speed, elapsed time, ETA, and checkpoint/error events. |
| `checkpoint-N/` | Model, optimizer, EMA, RNG streams, sample cursor, and update count; `metadata.yaml` marks completion. |
| `latest.json`, `complete.json` | Run-relative artifact names for resume and completion. |
| `averaged.pt` | Final EMA weights, update count, and compatible model/data settings. |
| `validation/update-N/`, `validation-latest.json` | Monitor metrics, 16 PNG panels, and a run-relative report reference. |
| `wandb-run.json`, `wandb/` | W&B identity/destination and local SDK records; checkpoints preserve the run ID and last validation update. |
| `performance.json` | Timed throughput, peak allocated memory, and initialization/consumed-stream hashes. |
| Pilot `configs/`, `batch-B/`, `pilot-summary.json` | Generated run settings, individual runs, and paired resource comparison. |

## Components

| Module | Responsibility |
| --- | --- |
| `src/indexing.py` | In-memory source indexes, fixed monitor/preview selection, and fingerprints. |
| `src/data.py` | EXIF/RGB preprocessing, source dataset, and shared horizontal flips with independently sampled exposure. |
| `src/brightness.py` | FP32 sRGB/linear-light conversion, exposure adjustment, and reference saturation masks. |
| `src/metrics.py` | Per-image RGB, raw-output, luminance, latent, and masked errors; exact luminance decompositions. |
| `src/config.py` / `src/filesystem.py` | Strict YAML settings and repository-relative paths. |
| `flow_matching.models` | Shared SiT-S/2 with scalar conditioning and four velocity channels. |
| `flow_matching.losses`, `.timesteps`, `.samplers` | Shared velocity MSE, timestep sampling, and Euler integration. |
| `src/vae.py` | Frozen VAE encoding and decoding. |
| `src/state.py` / `src/training.py` | Reproducible streams, resumable data order, EMA, Accelerate training, and checkpoint records. |
| `src/aggregation.py`, `validation.py`, `tracking.py` | Streaming statistics, monitor/previews, and W&B logging. |
| `scripts/train.sh`, `launch.py`, `train.py`, `sample.py`, `pilot.py` | YAML launcher, training, single-image generation, and equal-image pilot. |
| `tests/test_data_and_metrics.py` | Focused tests using temporary image fixtures. |
| `tests/test_flow_and_training.py` | Objective/sampler checks, conditioning learning, and exact checkpoint continuation with prefetched data. |
| `tests/test_real_model.py`, `test_entrypoints.py` | Opt-in real-image/model and two-GPU workflow checks. |
| `tests/test_validation_and_tracking.py` | Metric aggregation, fixed panels, run-ID resume, and paired pilot inputs; opt-in real offline SDK workflow. |

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

Step 2 checks passed: fourteen focused CPU tests; a 40-update learning check with full SiT-S and two real 256 × 256 training images; and two-GPU BF16 training, relocated checkpoint resume, and image sampling through the actual entry points. The real-image check reduced velocity MSE from 0.1068 to 0.0382 (last-five mean); this is a small implementation check.

Step 3 checks passed: eighteen focused CPU tests, a real offline W&B training/resume check, and the two-GPU workflow with an uneven 17-source monitor (51 exposure comparisons), 16 panels, and the same run ID after relocation/resume. Evidence: `.verification/step3_checks/`. Online authentication remains pending account configuration.

```bash
BRIGHTNESS_RUN_GPU_CHECKS=1 python -m unittest imagenet_brightness_flow.tests.test_real_model -v
BRIGHTNESS_RUN_LAUNCH_CHECKS=1 python -m unittest imagenet_brightness_flow.tests.test_entrypoints -v
BRIGHTNESS_RUN_TRACKING_CHECKS=1 python -m unittest imagenet_brightness_flow.tests.test_validation_and_tracking.OfflineTrackingTests -v
```

GPU checks use the cached VAE; the launcher check requires two visible GPUs. Evidence: `.verification/step2_checks/`. Shared-component checks and extraction comparisons: [toolbox verification](../flow_matching/README.md#verification). Architecture attribution: [third-party notices](../flow_matching/THIRD_PARTY_NOTICES.md).
