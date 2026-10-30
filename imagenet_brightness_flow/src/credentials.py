"""Load local W&B credentials into the environment, outside saved run settings."""

import os
from pathlib import Path

import yaml

from .filesystem import REPO_ROOT


def load_wandb_credentials(api_key_env, path=None):
    """Read .secret/wandb.yaml; existing nonempty environment values take priority."""
    if path is None:
        path = REPO_ROOT / ".secret" / "wandb.yaml"
    path = Path(path)
    if not path.is_file():
        return

    # Parsing errors must not echo private YAML contents into terminal logs.
    try:
        contents = path.read_text()
        values = yaml.safe_load(contents)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("Unable to read private W&B settings in .secret/wandb.yaml.") from None

    #region Private setting validation
    if not isinstance(values, dict) or set(values) - {"api_key", "entity"}:
        raise ValueError("Private W&B settings support only api_key and entity.")
    for name in ("api_key", "entity"):
        value = values.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Private W&B {name} must be null or a nonempty string.")
    #endregion

    # Credentials stay in process state; configuration exports contain no API key.
    variables = {"api_key": api_key_env, "entity": "WANDB_ENTITY"}
    for name, variable in variables.items():
        value = values.get(name)
        if value is not None and not os.environ.get(variable):
            os.environ[variable] = value
