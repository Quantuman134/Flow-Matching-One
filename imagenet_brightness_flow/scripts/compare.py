"""Train the timestep comparison and export full-split evaluation reports."""

import argparse

from imagenet_brightness_flow.src.comparison import run_comparison
from imagenet_brightness_flow.src.comparison_config import load_comparison
from imagenet_brightness_flow.src.filesystem import resolve_project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    arguments = parser.parse_args()
    path = resolve_project_path(arguments.config)
    settings, config = load_comparison(path)
    try:
        run_comparison(settings, config, path)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
