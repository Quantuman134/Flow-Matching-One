# Development log

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
