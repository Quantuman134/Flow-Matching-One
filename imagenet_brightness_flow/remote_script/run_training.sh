#!/usr/bin/env bash
# Run the configured comparison inside a node allocated by ~/get_node.sh.
set -eo pipefail

if (( $# > 1 )); then
    echo "Usage: bash remote_script/run_training.sh [COMPARISON.yaml]" >&2
    exit 2
fi

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" && pwd)"
TOPIC_DIR="$(dirname -- "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$TOPIC_DIR/configs/comparison.yaml}"

# Noninteractive srun shells may need the user's Conda initialization.
# Load shell settings before enabling nounset, which shell profiles may not support.
if ! command -v conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        CONDA_BIN_DIR="$(dirname -- "$CONDA_EXE")"
        export PATH="$CONDA_BIN_DIR:$PATH"
    elif [[ -r "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
fi
set -euo pipefail

#region Launch environment validation
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is unavailable. Initialize Conda before calling ~/get_node.sh." >&2
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Comparison configuration does not exist: $CONFIG_PATH" >&2
    exit 1
fi
#endregion

# Make Conda available to the existing Bash launcher even when it is a function.
CONDA_BASE="$(conda info --base)"
export PATH="$CONDA_BASE/bin:$PATH"

echo "Brightness comparison | host=$(hostname) | Slurm job=${SLURM_JOB_ID:-none}"
echo "Comparison configuration: $CONFIG_PATH"

# The YAML owns GPU/batch settings, comparison mode, outputs, and resume behavior.
# Replace this shell so cancellation reaches the existing comparison supervisor.
exec bash "$TOPIC_DIR/scripts/compare.sh" "$CONFIG_PATH"
