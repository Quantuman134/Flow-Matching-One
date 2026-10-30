#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
exec bash "$SCRIPT_DIR/train.sh" --compare "$@"
