# Development log

Dated history only; current rules start at the [agent guide](../AGENTS.md), and current experiment status lives in the topic README.

## 2026-10-20 — Explicit evaluation implementation

- Rewrote `evaluation.py` with named intermediate values, separate conversions and file paths, and ordinary loops; removed nested calls, method chains, comprehensions, and calls inside return expressions.
- Preserved evaluation interfaces, metric definitions, and output formats; focused comparisons verify exact metrics and saved arrays for K=1,10,N.
- Noise RNG, JSON output, model state, completion checks, syntax, and explicit-style checks passed; local evidence: `outputs/empirical_condition_density/explicit_style_checks/`.

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
- Checkpoint completion metadata now follows an all-rank save barrier; document ownership, links, syntax, and folding regions passed checks; evidence: `outputs/empirical_condition_density/review_checks/`.

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
