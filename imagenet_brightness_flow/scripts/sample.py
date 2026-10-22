"""Apply averaged brightness-flow weights to a source image."""

import argparse
from dataclasses import asdict

from PIL import Image
import torch

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.data import preprocess_image
from flow_matching.models import SiT
from flow_matching.samplers import euler_sample
from imagenet_brightness_flow.src.filesystem import resolve_project_path
from imagenet_brightness_flow.src.vae import FrozenVAE


def sample(config, weights_path, source_path, output_path, exposure, nfe):
    weights_path = resolve_project_path(weights_path)
    source_path = resolve_project_path(source_path)
    output_path = resolve_project_path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    requested = config.training.device
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    model = SiT(latent_size=config.data.image_size // 8)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    signature = weights["signature"]
    if signature["vae"] != asdict(config.vae):
        raise ValueError("Sampling VAE settings differ from the saved weights.")
    model.load_state_dict(weights["model"])
    model = model.to(device)
    vae = FrozenVAE.from_config(config.vae, device)
    source = preprocess_image(source_path, config.data.image_size)
    source = source.unsqueeze(0).to(device)
    condition = torch.tensor([[exposure]], dtype=torch.float32, device=device)
    if not torch.isfinite(condition).all() or not config.data.exposure_min <= exposure <= config.data.exposure_max:
        raise ValueError("Exposure must be finite and within configured limits.")
    latent = vae.encode(source)
    transported = euler_sample(model, latent, condition, nfe, config.training.mixed_precision)
    generated = vae.decode(transported.samples)
    pixels = generated[0].clamp(0, 1)
    pixels = (pixels * 255).round().to(torch.uint8)
    pixels = pixels.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(pixels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposure", type=float, required=True)
    parser.add_argument("--nfe", type=int, choices=[1, 4, 16], default=1)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    sample(config, arguments.weights, arguments.source, arguments.output, arguments.exposure, arguments.nfe)


if __name__ == "__main__":
    main()
