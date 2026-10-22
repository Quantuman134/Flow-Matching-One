"""Frozen FP32 VAE with deterministic posterior-mean endpoints."""

import torch
from torch import nn


class FrozenVAE(nn.Module):
    def __init__(self, autoencoder, scale=0.18215, batch_size=16):
        super().__init__()
        self.autoencoder = autoencoder.float()
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.scale = scale
        self.batch_size = batch_size
        self.train(False)

    @classmethod
    def from_config(cls, config, device):
        from diffusers import AutoencoderKL

        autoencoder = AutoencoderKL.from_pretrained(
            config.model_id, revision=config.revision, local_files_only=config.local_files_only,
            torch_dtype=torch.float32, use_safetensors=True,
        )
        autoencoder = autoencoder.to(device)
        wrapper = cls(autoencoder, config.scale, config.batch_size)
        return wrapper

    def train(self, mode=True):
        """Keep the frozen encoder/decoder in evaluation mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, images):
        chunks = []
        with torch.autocast(images.device.type, enabled=False):
            for batch in images.split(self.batch_size):
                normalized = 2 * batch.float() - 1
                encoded = self.autoencoder.encode(normalized)
                latent = encoded.latent_dist.mean * self.scale
                chunks.append(latent)
        latents = torch.cat(chunks, dim=0)
        return latents

    @torch.no_grad()
    def encode_pairs(self, source, reference):
        source_latent = self.encode(source)
        reference_latent = self.encode(reference)
        return source_latent, reference_latent

    @torch.no_grad()
    def decode(self, latents):
        """Return raw FP32 sRGB; metric/preview consumers choose their clipping."""
        chunks = []
        with torch.autocast(latents.device.type, enabled=False):
            for batch in latents.split(self.batch_size):
                unscaled = batch.float() / self.scale
                decoded = self.autoencoder.decode(unscaled)
                rgb = (decoded.sample.float() + 1) / 2
                chunks.append(rgb)
        images = torch.cat(chunks, dim=0)
        return images
