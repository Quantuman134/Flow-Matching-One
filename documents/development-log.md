# Development log

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
