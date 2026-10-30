"""Evaluate one run's final EMA weights on the complete validation split."""

import argparse

from imagenet_brightness_flow.src.comparison_config import load_comparison
from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.final_evaluation import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    settings, _ = load_comparison(arguments.comparison)
    try:
        evaluate_run(config, settings)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
