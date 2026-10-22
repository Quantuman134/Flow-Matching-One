#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash imagenet_brightness_flow/scripts/train.sh [CONFIG.yaml]" >&2
    exit 2
fi

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" && pwd)"
TOPIC_DIR="$(dirname -- "$SCRIPT_DIR")"
REPO_ROOT="$(dirname -- "$TOPIC_DIR")"
CONFIG_PATH="${1:-$TOPIC_DIR/configs/experiment.yaml}"
CONFIG_DIR="$(dirname -- "$CONFIG_PATH")"
CONFIG_NAME="$(basename -- "$CONFIG_PATH")"
CONFIG_DIR="$(cd -- "$CONFIG_DIR" && pwd)"
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
CONDA_BASE="$(conda info --base)"
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
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
exec python -u -m imagenet_brightness_flow.scripts.launch "$CONFIG_PATH"
