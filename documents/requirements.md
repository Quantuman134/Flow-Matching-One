# Requirements

## Scope

- The project concerns flow matching; the research goal, model, and dataset are pending the user's introduction.
- Use a general PyTorch training environment with Hugging Face tools as the initial baseline; revise it when the project scope is known.

## Communication and documentation

- Keep chat and documents concise and limited to necessary information; expand when requested.
- Use one consistent term for each concept; keep each documentation item to one or two sentences.
- Group related documentation items under short section titles for quick scanning.
- Maintain `documents/` throughout development, including code changes, environment changes, and important decisions.
- Keep current requirements here, environment information in `documents/env/`, and significant changes with validation in `documents/development-log.md`.

## Environment and portability

- Use `flow-matching-one` as the default Conda environment name.
- Provide a runnable Conda setup script and record dependency versions and setup commands for reproducibility.
- Distinguish the intended environment from the environment actually installed and tested; retain resolved dependency snapshots after installation.
- Training must support different GPU counts without editing source code; expose device selection and process count through runtime configuration.
- Prefer Hugging Face Accelerate for initial distributed training; use its device placement, distributed data loading, gradient synchronization, and main-process output controls.
- Keep per-device batch size explicit and record global batch size as `per-device batch size × process count × gradient accumulation steps`; document changes when GPU count changes.
- Support CPU execution for lightweight utilities and checks when practical; expensive GPU-only code may omit CPU support if documented.

## Terminology

| Term | Meaning |
| --- | --- |
| Flow matching | The project's training approach; the exact formulation is pending. |
| Environment | Python, installed packages, PyTorch compute build, and relevant GPU driver. |
| GPU count | Number of GPUs used by a run, independent of the number installed. |
| Per-device batch size | Samples processed by one training process per step. |
| Global batch size | Samples contributing to one optimizer update across all processes. |
