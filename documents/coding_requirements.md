# Coding requirements

## Code style

- Name files, modules, functions, and variables for their specific responsibility. Give different concepts distinct names; resolve ambiguous names by renaming them and updating references.
- Add concise docstrings and necessary comments for responsibilities, shapes, mathematical choices, and non-obvious behavior; avoid narrating obvious code.
- Group nearby lines serving one purpose into logical blocks; separate blocks with a blank line.
- Prefer explicit, step-by-step code with descriptive intermediate variables; keep each statement focused on one operation.
- In most cases, avoid nested function calls, chained calls, and calls embedded in arguments, containers, or return expressions; calculate values separately before using them.
- Prefer simple loops and shallow control flow over dense comprehensions and deeply nested blocks; prioritize ease of human review over compact code.
- Fold related validation checks with labeled `#region` / `#endregion` comments; keep intermediate checks near their computation.

## Configuration and execution

- Use YAML for settings and Bash launchers for common one-command tasks; read run settings from YAML rather than duplicating them in launchers.
- Document command and path conventions in the topic README.
- Resolve output and resume paths independently of the working directory; store references between run artifacts relative to their owning run folder so relocation does not break them.
- Provide a runnable Conda setup script for the environment described in the project requirements.

## Development levels

- Apply the single development level recorded in the topic README to implementation, verification, and environment changes. Under levels 1–2, complete one reviewable step and wait for review before proceeding.

| Level | Implementation review | Verification | Environment changes |
| --- | --- | --- | --- |
| 0 | Not required. | Focused checks when necessary. | Install needed packages directly in the specified environment. |
| 1 | After each implementation step. | Focused self-checks when necessary. | List required packages or new environments; wait for review before installation or creation. |
| 2 | After each implementation step. | Add tests for changed parts; use full-stack tests only when necessary. | List required packages or new environments; wait for review before installation or creation. |

- Record the resulting environment configuration following the [environment records](project_requirements.md#environment-records) requirements.
- Put test and self-check code in the topic's `tests/` and use standard test runners. Reserve `scripts/` for experiment entry points and launchers.

## Logging and resume

- Save persistent training and experiment logs; print loss, speed, elapsed time, ETA, and important events or errors; make logging intervals configurable.
- Record resolved settings, RNG seed/state, Git revision, environment, device count, batch sizes, metrics, and checkpoints.
- Save model, optimizer, update count, and RNG state for resume; aggregate distributed metrics and restrict shared writes to the main process.

## Device portability

- Support configurable GPU counts without source edits; avoid fixed device IDs and machine-specific paths, and support CPU for lightweight work where practical.
- Prefer Accelerate for distributed training; record `global batch = per-device microbatch × processes × accumulation`.
