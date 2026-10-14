# Requirements

## Scope

- Investigate flow matching with a one-to-one condition-target mapping for native one-step generation; see [research direction](research.md) for the proposal, assumptions, and open questions.
- The first experiment uses a 2D Gaussian prior, 50,000 spiral target points, and a conditional MLP with controlled condition density; see the [HTML development plan](experiment-01-plan.html).
- Build the fixed target set in memory at the start of the experiment and reuse it across condition counts; no standalone data-preparation command is needed.
- Implement the experiment one step at a time under level-1; pause for the user's review after each step before proceeding.
- Use a general PyTorch training environment with Hugging Face tools as the initial baseline; revise it when the project scope is known.

## Communication and documentation

- Keep chat and documents concise and limited to necessary information; expand when requested.
- Use one consistent term for each concept; keep each documentation item to one or two sentences.
- Group related documentation items under short section titles for quick scanning.
- Prefer short labeled items or tables for settings; separate descriptions from parameter values, for example `Target: three-turn Archimedean spiral` and `Samples: 50K`.
- Write for readers unfamiliar with the conversation: explain the setting, terms, symbols, and assumptions clearly before presenting formulas.
- Maintain `documents/` throughout development, including code changes, environment changes, and important decisions.
- Keep current requirements here, environment information in `documents/env/`, and significant changes with validation in `documents/development-log.md`.

## Logging

- Keep structured persistent logs for training and experiments, including loss, learning rate, gradient norm, evaluation metrics, timing, checkpoints, events, and errors; make logging frequency configurable.
- Print necessary terminal progress, including step/total, loss, learning rate, running speed, elapsed time, and ETA.
- Save resolved configuration, seeds, Git revision and dirty status, environment, device/GPU count, and global batch size; save model, optimizer, update count, and RNG state for resuming training.
- In distributed runs, aggregate shared metrics and let only the main process write shared outputs; keep these general requirements out of individual experiment plans.

## Development levels

| Level | Implementation | Human review | Verification |
| --- | --- | --- | --- |
| Level-0 | Agent implements directly. | Not required. | Checks appropriate to the change. |
| Level-1 | Agent implements one reviewable step. | Required after each step for this experiment. | Agent runs focused self-checks when necessary; dedicated test code is optional. |
| Level-2 | Agent implements and adds tests for each changed part. | Required. | Run tests related to the changes; use full-stack tests only when necessary. |

- Active level: **level-1** for `empirical_condition_density`; Step 1 is implemented and awaiting user review, and Step 2 has not started.

## Organization

- Put each experiment's scripts and configurations in its own folder at the repository root; the first experiment uses `empirical_condition_density/`.
- Keep related analyses inside their topic folder; give independent sub-topics their own root folder and keep genuinely reusable utilities in a shared location.
- Keep project-wide setup scripts in `scripts/`; do not mix scripts from different experiments there.

## Configuration and execution

- Default configuration format: **YAML**; keep settings needed by a task or experiment in its configuration file.
- Default entry point: **Bash launcher**; provide one-command execution for common tasks and allow selecting an alternate YAML file.
- Keep launchers inside the relevant topic folder; keep experiment parameters in YAML and document the command and path conventions.
- For `empirical_condition_density`, add the unified YAML configuration and Bash launcher after the full experiment code is assembled; intermediate steps provide reusable functions only.

## Code style

- Group related validation checks inside labeled `#region` / `#endregion` comments so they can be collapsed in the IDE.
- Keep checks that depend on intermediate results next to the relevant computation; folding must not change execution or indentation.

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
| Flow matching | The project's training approach; the proposed conditioning paradigm is described in `research.md`. |
| Condition | Information supplied to the model to specify possible targets; the first experiment proposes one normalized floating-point group midpoint. |
| Condition density | Number of distinct conditions K for the same fixed target dataset; larger K means fewer targets per condition. |
| One-step generation | Producing a target sample with one velocity-model evaluation and one Euler update. |
| Environment | Python, installed packages, PyTorch compute build, and relevant GPU driver. |
| GPU count | Number of GPUs used by a run, independent of the number installed. |
| Per-device batch size | Samples processed by one training process per step. |
| Global batch size | Samples contributing to one optimizer update across all processes. |
