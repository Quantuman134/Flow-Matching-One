# Project working rules

## Scope

- Read `documents/requirements.md` before development; it is the source of truth for user requirements and agreed terminology.
- Do not infer the research method, dataset, or model before the user provides the project details.

## Communication

- Keep chat and documentation concise; use consistent terms and one or two sentences per documentation item.
- Group related documentation items under short section titles.
- Use short labeled items for settings; separate descriptions from parameter values, such as `Target: three-turn Archimedean spiral` and `Samples: 50K`.
- Write for readers unfamiliar with the chat; explain concepts and symbols before using formulas, and prioritize clarity as well as brevity.

## Documentation

- Update relevant files in `documents/` whenever code, environment settings, requirements, or important decisions change; record significant work and validation in `documents/development-log.md`.

## Environment

- Keep environment instructions and observed configuration in `documents/env/`; distinguish planned settings from installed and tested settings.

## Logging

- Keep detailed persistent logs for training and experiments; print necessary progress in the terminal, including loss, running speed, elapsed time, ETA, and important events or errors.
- Keep logging requirements in the general project guides rather than repeating them in experiment plans.

## Organization

- Put scripts and configurations for each experiment, analysis, or sub-topic in its own descriptively named folder at the repository root; keep shared utilities and project setup separate.

## Configuration and execution

- By default, keep code and experiment settings in YAML configuration files and provide Bash launchers for common tasks, so each task normally runs with one command.
- Keep launchers beside their topic's code; read run settings from YAML rather than duplicating them in Bash.
- For the current experiment, build data during the experiment run and add the unified YAML configuration and Bash launcher after the full experiment code is ready.

## Code style

- Wrap related validation checks in labeled `#region` / `#endregion` comments for IDE folding; keep checks that depend on intermediate results near the relevant computation.

## Development levels

- **Level-0:** The agent implements directly; human review is not required.
- **Level-1:** The agent implements, then pauses for human review; run focused self-checks when necessary.
- **Level-2:** The agent implements and adds test code for each changed part, then pauses for human review; run focused tests and use full-stack tests only when necessary.
- **Current experiment:** Level-1; complete one step at a time and wait for the user's review before starting the next step.

## GPU portability

- Support configurable GPU counts for training; avoid fixed device IDs, process counts, or machine-specific paths. Use CPU support for lightweight work where practical; document GPU-only requirements for expensive workloads.
