"""Repository-relative paths shared by configuration, training, and launchers."""

from pathlib import Path


MODULE_PATH = Path(__file__)
MODULE_PATH = MODULE_PATH.resolve()
REPO_ROOT = MODULE_PATH.parents[2]


def resolve_project_path(value):
    """Resolve an output or checkpoint path independently of the working directory."""
    path = Path(value)
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path

    path = path.resolve()
    return path


def resolve_checkpoint_path(value, run_dir):
    """Accept an absolute path, a repository-relative path, or a run-local name."""
    path = Path(value)
    path = path.expanduser()

    if not path.is_absolute():
        num_parts = len(path.parts)
        if num_parts == 1:
            run_root = resolve_project_path(run_dir)
            path = run_root / path
        else:
            path = REPO_ROOT / path

    path = path.resolve()
    return path
