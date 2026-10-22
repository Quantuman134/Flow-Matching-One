"""Opt-in check of the cached pretrained VAE and full SiT-S on real images."""

import json
import os
import unittest

import torch

from imagenet_brightness_flow.src.brightness import apply_exposure
from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.data import preprocess_image
from flow_matching.losses import flow_matching_loss
from flow_matching.models import SiT
from flow_matching.interpolants import linear_path
from flow_matching.samplers import euler_sample
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT, resolve_project_path
from imagenet_brightness_flow.src.vae import FrozenVAE


@unittest.skipUnless(os.environ.get("BRIGHTNESS_RUN_GPU_CHECKS") == "1", "Opt-in real-model GPU check")
class RealModelTests(unittest.TestCase):
    def test_real_vae_sit_learning_and_sampling(self):
        self.assertTrue(torch.cuda.is_available())
        torch.set_num_threads(2)
        torch.manual_seed(0)
        config = load_config(TOPIC_ROOT / "configs" / "experiment.yaml")
        data_root = resolve_project_path(config.data.root)
        paths = sorted((data_root / "train" / "0000").iterdir())
        images = []
        for path in paths[:2]:
            image = preprocess_image(path, 256)
            images.append(image)
        source = torch.stack(images).to("cuda")
        exposure = torch.tensor([[-1.0], [1.0]], device="cuda")
        reference = apply_exposure(source, exposure)
        vae = FrozenVAE.from_config(config.vae, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            source_latent, target_latent = vae.encode_pairs(source, reference.image)
        self.assertEqual(source_latent.shape, (2, 4, 32, 32))
        self.assertEqual(source_latent.dtype, torch.float32)
        self.assertFalse(source_latent.requires_grad)
        repeated = vae.encode(source)
        torch.testing.assert_close(source_latent, repeated, rtol=0, atol=0)
        model = SiT().to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        times = torch.zeros(2, device="cuda")
        losses = []
        for _ in range(40):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state, target_velocity = linear_path(source_latent, target_latent, times)
                prediction = model(state, times, exposure)
                loss = flow_matching_loss(prediction, target_velocity)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(sum(losses[-5:]) / 5, losses[0] * 0.8)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
        nfe_results = {}
        for nfe in (1, 4, 16):
            generated = euler_sample(model, source_latent, exposure, nfe, "bf16")
            decoded = vae.decode(generated.samples)
            self.assertEqual(decoded.shape, source.shape)
            self.assertEqual(decoded.dtype, torch.float32)
            self.assertTrue(torch.isfinite(decoded).all())
            latent_mse = (generated.samples - target_latent).square().mean().item()
            nfe_results[nfe] = {"latent_mse": latent_mse}
        report = {
            "device": torch.cuda.get_device_name(), "vae_revision": config.vae.revision,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "sources": [], "updates": len(losses), "learning_rate": 1e-4,
            "initial_loss": losses[0], "last_five_mean_loss": sum(losses[-5:]) / 5,
            "nfe_results": nfe_results, "losses": losses,
        }
        for path in paths[:2]:
            report["sources"].append(str(path.relative_to(data_root)))
        folder = TOPIC_ROOT / ".verification" / "step2_checks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "real_model.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
