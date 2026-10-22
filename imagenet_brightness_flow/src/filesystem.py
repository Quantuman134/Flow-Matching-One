"""Repository-relative dataset and configuration paths."""

from pathlib import Path


MODULE_PATH = Path(__file__)
MODULE_PATH = MODULE_PATH.resolve()
TOPIC_ROOT = MODULE_PATH.parents[1]
REPO_ROOT = MODULE_PATH.parents[2]


def resolve_project_path(value: str | Path) -> Path:
    """Expand user paths and anchor relative paths at the repository root."""
    path = Path(value)
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    return path
