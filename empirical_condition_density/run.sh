#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash empirical_condition_density/run.sh [CONFIG.yaml]" >&2
    exit 2
fi

TOPIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$TOPIC_DIR")"
CONFIG_PATH="${1:-$TOPIC_DIR/configs/experiment.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Configuration not found: $CONFIG_PATH" >&2
    exit 2
fi
CONFIG_PATH="$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)/$(basename -- "$CONFIG_PATH")"
CONDA_BASE="$(conda info --base)"
# Read the environment name with Conda's YAML reader before selecting Python.
ENV_NAME="$("$CONDA_BASE/bin/python" - "$CONFIG_PATH" <<'PY'
import sys
from ruamel.yaml import YAML

with open(sys.argv[1]) as stream:
    config = YAML(typ="safe").load(stream)
name = config.get("environment") if isinstance(config, dict) else None
if not isinstance(name, str) or not name.strip():
    raise SystemExit("YAML must specify a nonempty environment name.")
print(name)
PY
)"

cd -- "$REPO_ROOT"
exec conda run --no-capture-output --name "$ENV_NAME" \
    python -u -m empirical_condition_density.launch "$CONFIG_PATH"
