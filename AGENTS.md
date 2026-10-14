# Project working rules

## Scope

- Read `documents/requirements.md` before development; it is the source of truth for user requirements and agreed terminology.
- Do not infer the research method, dataset, or model before the user provides the project details.

## Communication

- Keep chat and documentation concise; use consistent terms and one or two sentences per documentation item.
- Group related documentation items under short section titles.

## Documentation

- Update relevant files in `documents/` whenever code, environment settings, requirements, or important decisions change; record significant work and validation in `documents/development-log.md`.

## Environment

- Keep environment instructions and observed configuration in `documents/env/`; distinguish planned settings from installed and tested settings.

## GPU portability

- Support configurable GPU counts for training; avoid fixed device IDs, process counts, or machine-specific paths. Use CPU support for lightweight work where practical; document GPU-only requirements for expensive workloads.
