#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [ENV_NAME] [cu128|cu126|cpu]
Defaults: flow-matching-one cu128. Requires Conda and Linux x86_64.
Creates a new environment; refuses to modify an existing environment.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# > 2 )); then usage >&2; exit 2; fi
ENV_NAME=${1:-flow-matching-one}
TORCH_BUILD=${2:-cu128}
if [[ ! $ENV_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo 'ENV_NAME must contain only letters, digits, underscores, or hyphens.' >&2
    exit 2
fi
if [[ $ENV_NAME == base || $ENV_NAME == root ]]; then
    echo 'Choose a project environment name other than base or root.' >&2
    exit 2
fi
case "$TORCH_BUILD" in
    cu128|cu126|cpu) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'This baseline supports Linux x86_64; select platform-specific wheels for other systems.' >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo 'Install Miniconda/Anaconda and put conda on PATH first.' >&2
    exit 1
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_BASE=$(conda info --base)
CONDA_PYTHON="$CONDA_BASE/bin/python"
if conda env list --json | "$CONDA_PYTHON" -c '
import json, pathlib, sys
name = sys.argv[1]
sys.exit(0 if any(pathlib.Path(p).name == name for p in json.load(sys.stdin)["envs"]) else 1)
' "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists; choose a new name." >&2
    exit 1
fi

conda create --yes --name "$ENV_NAME" --override-channels --channel conda-forge \
    python=3.11.13 pip=25.2
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url "https://download.pytorch.org/whl/$TORCH_BUILD"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
    -r "$REPO_ROOT/documents/env/requirements.txt"
conda run --no-capture-output --name "$ENV_NAME" python -m pip check

CHECK_ARGS=()
if [[ $TORCH_BUILD == cpu ]]; then CHECK_ARGS+=(--cpu); else CHECK_ARGS+=(--require-cuda); fi
conda run --no-capture-output --name "$ENV_NAME" python \
    "$REPO_ROOT/scripts/check_env.py" "${CHECK_ARGS[@]}"

SNAPSHOT_DIR="$REPO_ROOT/documents/env/local/$ENV_NAME"
mkdir -p "$SNAPSHOT_DIR"
conda env export --name "$ENV_NAME" > "$SNAPSHOT_DIR/environment.yml"
conda run --name "$ENV_NAME" python -m pip freeze > "$SNAPSHOT_DIR/requirements.lock.txt"
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    uname -sm
    echo "PyTorch build: $TORCH_BUILD"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version --format=csv || true
    fi
} > "$SNAPSHOT_DIR/host.txt"
echo "Setup complete. Activate with: conda activate $ENV_NAME"
echo "Resolved environment recorded in $SNAPSHOT_DIR"
