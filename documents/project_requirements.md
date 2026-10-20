# Project requirements

## Organization

- Give each experiment, analysis, or sub-topic its own descriptive folder at the repository root; keep its scripts, configurations, and launcher together.
- Keep related analyses within their topic; keep shared utilities and project setup separate from topic folders.
- Within each topic, use `src/` for reusable Python code, `scripts/` for entry points and launchers, and `configs/` for settings; keep the README at the topic root.
- Keep each experiment's results in `<topic>/outputs/` and verification artifacts in `<topic>/.verification/`; keep both ignored by Git and separate from each other.

## Documentation

- Update relevant documents after code, environment, requirements, or important decisions change; record significant work and validation in the development log.
- Keep each document focused on its responsibility:

| Document | Responsibility |
| --- | --- |
| `research.md` | Research hypotheses, assumptions, and open questions. |
| Topic plan | Experiment or analysis design and metric definitions. |
| Topic README | Usage, outputs, selected development level, and review status. |
| `env/` | Environment setup, planned settings, and observed configuration. |
| `development-log.md` | Dated changes and validation history. |

## Environment records

- Default environment: `flow-matching-one`; use PyTorch with Hugging Face tools as the initial baseline.
- Distinguish planned settings from installed/tested settings; retain dependency versions, setup commands, and resolved environment snapshots in `documents/env/`.
- Document any GPU-only requirements for expensive workloads.
