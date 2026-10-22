"""Train one ImageNet brightness-flow timestep setting from YAML."""

import argparse

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    try:
        train(config)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
