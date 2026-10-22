# Development log

Dated history only; current rules start at the [agent guide](../AGENTS.md), and current experiment status lives in the topic README.

## 2026-10-21 — Runtime data workflow

- Verified integer-size resizing against torchvision's runtime implementation; retained the integer form at the user's request. The preprocessing test passed, and integer/list forms produced identical portrait, landscape, and square results.
- Moved focused checks to `imagenet_brightness_flow/tests/` under the standard unittest runner; added the test/self-check placement rule to the coding requirements.
- Removed standalone preparation commands, CSV manifest I/O, and preparation settings. Added in-memory indexing and runtime dataset construction; source decoding/preprocessing occurs on access, and exposure targets are constructed per training batch. Updated the plan, experiment YAML, and README.
- Validation: eight focused tests passed, including lazy decoding and zero filesystem artifacts during dataset construction. The runtime factory indexed 1,281,167 training and all 50,000 validation images, selected 1,000 monitor and 16 preview cases, and passed metrics checks on 16 real sources. Evidence: `.verification/step1_checks/runtime_cpu.log` and `runtime_data.json`; Step 1 remains awaiting review.

## 2026-10-21 — Environment review rules

- Added environment changes to the coding restrictions: level 0 permits installing needed packages in the specified environment; levels 1–2 require listing packages or new environments and waiting for review before installation or creation.
- Combined implementation review, verification, and environment changes in one table governed by the topic's single development level; linked configuration recording to the existing environment-record requirements.
- Validation: unified-level wording, document link, and whitespace checks passed.

## 2026-10-21 — Brightness Step 1

- Recorded user-selected development level 1 in the plan and new topic README. Implemented strict YAML data settings, source manifests, deterministic monitor/preview selection, EXIF/RGB preprocessing, fresh exposure pairs, and per-image RGB/luminance/latent metrics with saturation and flat-image handling.
- Prepared `imagenet_brightness_flow/outputs/data-v1/`: 1,281,167 training and all 50,000 validation paths, 1,000 monitor sources, and 16 preview cases covering nine exposures. Saved hashes, resolved settings, provenance, and progress logs; sample decoding covered 2,000 images.
- Validation: eight focused CPU checks passed; real manifest hashes, class mappings, monitor membership, 16-source preprocessing, and CPU/CUDA pairing/metrics passed. Bash and direct Python preparation passed from `/tmp` with spaced paths; evidence: `imagenet_brightness_flow/.verification/step1_checks/`.
- Observed torchvision 0.24.1+cu128 and Pillow 12.3.0 in the existing environment; recorded versions without package changes. Step 1 awaits review; Steps 2–4 remain pending.

## 2026-10-21 — W&B and full validation plan

- Added W&B API-key environment entry, run/resume records, and 16 fixed source/reference/output previews at each validation; recorded the SDK as a planned dependency. Final evaluation now includes all 50,000 validation images: 5.4M comparisons and 37.8M per-image velocity evaluations.
- Proposed global batch 1024 with 25,000 updates, retaining 25.6M image presentations; aligned validation/checkpoints to 2,500 updates and EMA decay to 0.9996. Added an equal-image pilot comparison against batch 256 with learning rate 1e-4 as the starting point.
- Replaced notation prose with tables, removed VAE reconstruction measurements, and strengthened the guide's direct-writing rule. Validation: HTML links/navigation, settings consistency, removed-section checks, W&B placeholders, budget arithmetic, and whitespace checks passed. Documentation changes only.

## 2026-10-21 — Brightness plan review

- Changed the plan to SiT-S/2, including width 384, six attention heads, and the matching exposure embedding; confirmed `/data/hkzhang/imagenet_raw` as the selected dataset.
- Renamed “Training times” to “Training timestep sampling”; rewrote evaluation around metric purposes, calculations, scales, and examples, with explained equations and optional decomposition details. Removed the redundant “Controls and interpretation” section; retained VAE reconstruction measurements under evaluation.
- Updated the agent guide's communication rules to require defined symbols and explained metrics before formulas. Validation: architecture checked against official SiT definitions; HTML links, section navigation, metric decomposition examples, and whitespace checks passed. No experiment code or training changed.

## 2026-10-21 — HTML plan presentation

- Converted the [brightness experiment plan](experiment-02-brightness/plan.html) to a self-contained HTML document with section navigation, responsive tables, readable formulas, and print styles; updated links and removed the superseded Markdown version.
- Validation: all original plan content retained; local links, section anchors, document structure, and whitespace checks passed. No experiment settings changed.

## 2026-10-21 — Brightness experiment plan

- Added the [ImageNet brightness plan](experiment-02-brightness/plan.html) in its own document folder; specified SiT-B with a frozen VAE, online exposure targets, four training-time settings, twelve NFE comparison cells, and brightness/spatial error diagnostics. Implementation remains unstarted.
- Inspected dataset entries and sample image headers: `~/ILSVRC` contains pooled latents; raw JPEGs are under `/data/hkzhang/imagenet_raw`, with 1,281,167 training and 50,000 validation images. Checked official SiT and color-space references; linked the plan from the document index and research questions.
- Validation: local document links, exposure identity, both luminance MSE decompositions, training/inference time-grid labels, evaluation counts, and whitespace checks passed. Preserved existing unrelated working-tree edits; no training or environment changes.

## 2026-10-20 — Output overwrite

- Added YAML `overwrite` and enabled it in the default configuration; the main rank replaces the selected output folder before fresh training, including stale checkpoints, logs, and evaluations.
- Kept overwrite separate from resume and evaluation-only mode; restricted replacement to run folders inside the topic's output or verification folders, with input YAML stored outside.
- Validation: CPU and two-GPU replacement, stale-artifact removal, resume, configuration/path guards, legacy YAML, syntax, and whitespace checks passed; evidence: `.verification/overwrite_checks/`. The existing comparison results remain unchanged.

## 2026-10-20 — Relocation path fixes

- Fixed direct Python entry-point imports and centralized output/resume path resolution; new completion records use run-local checkpoint names.
- Repaired seven moved checkpoint references and the saved replay configuration; backed up original metadata in `.verification/path_dependency_checks/metadata_backup/`.
- Reproduced `run.sh` resolving the correct folder and stopping at existing-output protection; preserved the user's 2M-update setting, which needs a new output directory for a fresh run.
- Validation: Bash and direct-Python launchers passed from `/tmp`, including a spaced YAML filename; relocated checkpoints resumed using repository-relative paths and run-local names; seven baseline references, replay settings, syntax, and whitespace checks passed.

## 2026-10-20 — Organize experiment scripts

- Moved reusable Python components to `src/` and executable entry points to `scripts/` within the experiment; updated imports, module commands, and repository-path resolution.
- Added the general folder convention to project requirements and updated the topic guide; configuration, results, and verification artifacts stay in their existing topic folders.
- Validation: relocated imports, repository/output paths, saved-protocol compatibility, syntax, and guide links passed; small CPU and two-GPU launcher runs passed under `.verification/organization_checks/`, including an external working directory and spaces in the YAML path.

## 2026-10-20 — Topic-local artifacts

- Moved experiment results to `empirical_condition_density/outputs/` and verification artifacts to `empirical_condition_density/.verification/`; updated default paths, ignore rules, and documentation.
- Confirmed the existing comparison finished before moving it: seven training runs and 28 main evaluations; retained its original run records.
- Validation: new default paths, saved-protocol compatibility, all model/evaluation artifacts, and Git ignore rules passed checks.

## 2026-10-20 — Separate verification artifacts

- Added the rule to reserve experiment outputs for actual results; development-check artifacts and the old data preview now live in ignored `empirical_condition_density/.verification/`.
- Updated artifact references; the actual `comparison/` results remain in the experiment output directory.

## 2026-10-20 — Explicit evaluation implementation

- Rewrote `evaluation.py` with named intermediate values, separate conversions and file paths, and ordinary loops; removed nested calls, method chains, comprehensions, and calls inside return expressions.
- Preserved evaluation interfaces, metric definitions, and output formats; focused comparisons verify exact metrics and saved arrays for K=1,10,N.
- Noise RNG, JSON output, model state, completion checks, syntax, and explicit-style checks passed; local evidence: `empirical_condition_density/.verification/explicit_style_checks/`.

## 2026-10-20 — Explicit coding style

- Strengthened the coding convention to favor named intermediate values, separate operations, simple loops, and shallow control flow; nested or embedded calls are exceptions rather than the default.

## 2026-10-20 — GPU monitoring dependency

- Recorded the user's pip installation of `gpustat` **1.1.1** in `flow-matching-one`; added its version pin to environment requirements for future setup.
- Refreshed Conda and pip snapshots; `pip check` and the `gpustat --version` command passed.

## 2026-10-20 — Readable function calls

- Added the convention to keep short calls inline and separate long or complex embedded calls into named intermediate results.
- Applied it in `evaluation.py` to metadata, sampling inputs, metric collection, variance calculations, and progress serialization.
- Validation: metrics and saved arrays match the previous implementation exactly for K=1,10,N; model state, completion checks, syntax, and whitespace checks passed.

## 2026-10-20 — Remove unseen-condition evaluation

- Removed `_unseen()`, its sampling branch, settings, metric/figure exports, and completion checks; simplified evaluation signatures and updated callers and experiment documentation.
- Evaluation now uses training conditions with fresh noise, including the conditional-spread check.
- Validation: remaining metrics match previous outputs exactly for K=1,10,N; a small CPU launcher run, completion checks, syntax, and output checks passed.

## 2026-10-19 — Three-guide organization

- Redistributed the agent guide and former `requirements.md` into `AGENTS.md`, `coding_requirements.md`, and `project_requirements.md`; each rule has one owner.
- Removed the former requirements file and updated active references; coding and project requirements remain general, with topic-specific settings kept in topic documents.
- Validation: guide structure, local links, duplicate items, and whitespace checks passed; no code changed.

## 2026-10-19 — One source for working rules

- Consolidated working rules in `requirements.md`; `AGENTS.md` now points there instead of repeating requirements, including the logical-block spacing convention.

## 2026-10-19 — Logical block spacing

- Added the convention to separate logical code blocks with blank lines; applied it in `evaluation.py` and verified the Python syntax tree is unchanged.

## 2026-10-19 — Launcher introduction

- Added a three-point header describing resource validation, process selection, and execution environment handling in `launch.py`; documentation-only change.

## 2026-10-19 — Documentation ownership and review revisions

- Made requirements general, kept research notes focused on hypotheses, and compacted the document index, experiment guide, and environment guide; added ownership, compactness, and code-comment rules to `AGENTS.md`.
- Replaced the seed sweep with scalar `seed: 0`; removed gradient clipping and cross-seed statistics; preserved the YAML's four Euler counts and 50K checkpoint interval.
- Added comments explaining configuration loading, process launch, distributed batches, evaluation stages, metric formulas, and state restoration; revised comparison size: seven models and 28 main evaluations.
- Validation: single-seed CPU/two-GPU workflows and plots passed; gradients above the former clipping threshold reach Adam unchanged, and unclipped checkpoint continuation is exact.
- Checkpoint completion metadata now follows an all-rank save barrier; document ownership, links, syntax, and folding regions passed checks; evidence: `empirical_condition_density/.verification/review_checks/`.

## 2026-10-19 — Step 4: comparison workflow

- Committed approved Step 2 as `577fef6`; implemented the comparison runner, strict YAML configuration, Bash launcher, metric tables, coordinate exports, and figures.
- Added geometric and conditional precision, recall, sliced Wasserstein-1, group-mean RMSE, conditional spread, and separate unseen-condition evaluation; full defaults specify 21 training runs and 147 main evaluations.
- Validation passed for 50K metric references, exact velocity fields at all seven Euler counts, small CPU/two-GPU sweeps, exact distributed checkpoint recovery, portable CPU evaluation, seed summaries, and output guards.
- Fixed a shared-manifest creation race found by the two-GPU check; installed planned SciPy 1.15.3 and refreshed environment snapshots.
- Step 4 implementation awaits level-1 review; the pilot remains skipped, and the full-budget sweep and findings report remain pending.

## 2026-10-19 — Step 2 approval and pilot skipped

- The user approved Step 2 and requested a commit of the reviewed implementation.
- Skip Step 3 (pilot); proceed directly to Step 4 (full comparison), retaining level-1 review after implementation.

## 2026-10-15 — Step 2: model, training, and sampling

- Implemented the conditional MLP, linear flow matching loss, Euler sampler, and FP32 Accelerate trainer with fixed global batches, run records, validation, and checkpoint resume.
- Core checks passed for model dimensions, path endpoints, velocity loss, learning, Euler steps 1–64, invalid inputs, numerical guards, and precision/mode restoration.
- Short CPU, one-GPU, and two-GPU checks passed; CPU resume was bitwise identical to uninterrupted training, and two-GPU resume and parameter synchronization passed.
- Installed the pinned Accelerate dependencies and updated environment records; Step 2 awaits level-1 review, with pilot runs, unified input YAML, and the Bash launcher still pending.

## 2026-10-14 — Validation folding convention

- Adopted labeled folding regions for related validation checks in the general coding conventions; checks on intermediate results stay near their computation.

## 2026-10-14 — Validation folding trial

- Added labeled `#region` / `#endregion` comments around validation checks in `empirical_condition_density/data.py` for IDE folding; this is a style trial for user review.
- Verification: six balanced regions; Python syntax passed and the executable code structure is unchanged.

## 2026-10-14 — Data creation during the experiment

- Removed the standalone preparation command, its YAML configuration, and its Bash launcher; kept reusable data and visualization functions.
- The experiment will build targets in memory at startup; its unified YAML configuration and launcher will be added after the full code is assembled, and Step 1 remains awaiting review.
- Validation: reusable data construction, pair sampling, and visualization imports passed after removing the standalone entry point.

## 2026-10-14 — YAML configuration

- Changed the default configuration convention to YAML and converted the data config, launcher, Python loader, saved configuration, and usage instructions.
- Installed the pinned PyYAML 6.0.2 dependency; the Bash launcher uses Conda's existing YAML reader without changing the base environment.
- Validation: default execution, custom YAML with comments, paths with spaces, alternate working directory, saved YAML, and invalid-input guards passed; refreshed environment snapshots.

## 2026-10-14 — JSON configuration and Bash launchers

- Added the convention to keep task settings in JSON and provide Bash launchers for common one-command workflows.
- Added `empirical_condition_density/run_data.sh`; environment selection and output destination now come from `configs/data.json`, with documented path handling.
- Validation: Bash syntax, default execution, alternate JSON, execution from another directory, paths with spaces, and argument guards passed; Step 1 remains awaiting review.

## 2026-10-14 — Experiment folder and name

- Renamed the experiment to `empirical_condition_density/` at the repository root, removed the empty parent folder, and moved generated artifacts to the matching output directory.
- Updated package commands, imports, repository path resolution, and documentation; Step 1 remains awaiting review.
- Validation: the renamed CLI, imports, sampling interface, output location, folder layout, and documentation links passed.

## 2026-10-14 — Step 1: data and conditions

- Recorded development levels 0–2; this experiment uses level-1 with human review after every step, and Step 1 is awaiting review before Step 2.
- Implemented the 50K spiral, seven condition schemes, fresh Gaussian pair sampling, artifact export, and the reference visualization in `empirical_condition_density/`.
- Validation: target uniqueness, normalization, all group counts, encoding examples, endpoint references, reproducibility, fresh noise, Gaussian moments, invalid inputs, and CPU artifact generation passed; the preview was visually checked.
- Environment: added Matplotlib 3.10.6, recorded observed versions and a package snapshot, and passed `pip check`; no model training or distributed training checks were run.

## 2026-10-14 — Experiment plan revision

- Increased the target dataset to 50,000 points, revised compatible condition counts and evaluation sizes, and explained the spiral and integer-index-to-scalar condition encoding with examples.
- Removed logging requirements from the experiment plan; updated general guides for logging, writing for unfamiliar readers, and separate folders for each experiment or sub-topic.
- Validation: HTML structure, links, JavaScript syntax, all seven 50,000-target groupings, display limits, and condition examples passed; no experiment code was developed.

## 2026-10-14 — Experiment 01 development plan

- Added a standalone HTML plan for Gaussian-to-spiral flow matching with a K-controlled condition sweep, continuous-time MLP training, Euler step comparisons, and explicit metrics.
- Included an interactive analytic group-mean illustration and added persistent logging plus terminal progress to the working rules; experiment implementation and training have not started.
- Validation: HTML structure, internal links, offline assets, JavaScript syntax, and SVG generation at all seven density levels passed; browser appearance has not been reviewed.

## 2026-10-14 — Proposed research direction

- Recorded the user's proposed one-to-one condition-target paradigm and its intended trade-off between condition information and one-step generation.
- Documented an idealized linear-path derivation and open questions about inference conditions, generalization, diversity, and direct prediction; no code was changed.

## 2026-10-14 — Conda environment name

- Changed the setup default and documentation to `flow-matching-one`; added commands to rename the existing `flow-matching` environment.
- Confirmed `flow-matching` exists and `flow-matching-one` does not; the existing environment was not modified, and script syntax and help checks passed.

## 2026-10-14 — Working rules organization

- Grouped `AGENTS.md` into Scope, Communication, Documentation, Environment, and GPU portability; recorded short section titles as a documentation convention.

## 2026-10-14 — Initial organization and environment baseline

- Added persistent working rules, requirements, and a documentation index; the research scope remains pending.
- Added a pinned PyTorch/Hugging Face baseline, a Conda setup script, and a CPU/single-GPU/multi-GPU environment check.
- Recorded Linux x86_64, four NVIDIA H100 NVL GPUs, and driver 615.71.09; no project Conda environment has been installed.
- Validation: syntax, CLI help, invalid-argument handling, and existing-environment protection passed; installation and CPU/GPU/distributed runtime checks remain unverified.
