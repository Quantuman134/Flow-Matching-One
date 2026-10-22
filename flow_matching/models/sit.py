"""SiT-S/2 with a scalar condition, implemented with PyTorch.

Architecture reference: https://github.com/willisma/SiT/blob/main/models.py
The class embedding is replaced by a scalar MLP and the output has four
velocity channels. Smaller constructor dimensions support focused CPU checks.
"""

import math

import torch
from torch import nn
from torch.nn import functional as functional


def timestep_features(times: torch.Tensor, width: int = 256) -> torch.Tensor:
    """Continuous sinusoidal features, using the SiT frequency convention."""
    frequencies = torch.arange(width // 2, device=times.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000) * frequencies / (width // 2))
    angles = times[:, None].float() * frequencies[None]
    cosine = angles.cos()
    sine = angles.sin()
    features = torch.cat([cosine, sine], dim=-1)
    return features


def position_features(grid_size: int, width: int) -> torch.Tensor:
    """Fixed 2-D sine/cosine positions, horizontal coordinates first."""
    coordinates = torch.arange(grid_size, dtype=torch.float64)
    vertical, horizontal = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = torch.arange(width // 4, dtype=torch.float64)
    frequencies = 10000 ** (-frequencies / (width // 4))
    components = []
    for grid in (horizontal, vertical):
        positions = grid.reshape(-1, 1)
        angles = positions * frequencies[None]
        components.extend([angles.sin(), angles.cos()])
    features = torch.cat(components, dim=1)
    features = features.float().unsqueeze(0)
    return features


def modulate(tokens, shift, scale):
    scaled = tokens * (1 + scale[:, None])
    shifted = scaled + shift[:, None]
    return shifted


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens):
        batch, length, width = tokens.shape
        combined = self.qkv(tokens)
        combined = combined.reshape(batch, length, 3, self.heads, self.head_width)
        combined = combined.permute(2, 0, 3, 1, 4)
        query, key, value = combined.unbind(0)
        attended = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        output = self.projection(attended)
        return output


class SiTBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.attention = Attention(width, heads)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width),
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, tokens, condition):
        parameters = self.condition(condition)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parameters.chunk(6, dim=1)
        normalized = self.attention_norm(tokens)
        normalized = modulate(normalized, shift_a, scale_a)
        attended = self.attention(normalized)
        tokens = tokens + gate_a[:, None] * attended

        normalized = self.mlp_norm(tokens)
        normalized = modulate(normalized, shift_m, scale_m)
        transformed = self.mlp(normalized)
        tokens = tokens + gate_m[:, None] * transformed
        return tokens


class SiT(nn.Module):
    """Predict [B,4,S,S] velocity from latent, [B] time, and [B,1] condition.

    The exposure parameter names preserve existing brightness checkpoints.
    """

    def __init__(self, latent_size=32, width=384, depth=12, heads=6, patch_size=2):
        super().__init__()
        #region Architecture validation
        for value in (latent_size, width, depth, heads, patch_size):
            if type(value) is not int or value < 1:
                raise ValueError("Architecture dimensions must be positive integers.")
        if latent_size % patch_size or width % 4 or width % heads:
            raise ValueError("Require divisible latent patches, 2-D positions, and attention heads.")
        #endregion
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.patch = nn.Conv2d(4, width, kernel_size=patch_size, stride=patch_size)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.exposure = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = SiTBlock(width, heads)
            self.blocks.append(block)
        self.output_norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.output_condition = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, 4 * patch_size * patch_size)
        positions = position_features(latent_size // patch_size, width)
        self.register_buffer("positions", positions)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        patch_weights = self.patch.weight.view(self.patch.weight.shape[0], -1)
        nn.init.xavier_uniform_(patch_weights)
        nn.init.zeros_(self.patch.bias)
        for embedding in (self.time, self.exposure):
            nn.init.normal_(embedding[0].weight, std=0.02)
            nn.init.normal_(embedding[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.condition[-1].weight)
            nn.init.zeros_(block.condition[-1].bias)
        nn.init.zeros_(self.output_condition[-1].weight)
        nn.init.zeros_(self.output_condition[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, times, condition):
        batch = latent.shape[0]
        #region Input shape validation
        if latent.shape != (batch, 4, self.latent_size, self.latent_size):
            raise ValueError("latent must have shape [B,4,latent_size,latent_size].")
        if times.shape != (batch,) or condition.shape != (batch, 1):
            raise ValueError("times and condition must have shapes [B] and [B,1].")
        #endregion
        patches = self.patch(latent)
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self.positions
        time_features = timestep_features(times)
        time_embedding = self.time(time_features)
        scalar_condition = condition.float()
        condition = time_embedding + self.exposure(scalar_condition)
        for block in self.blocks:
            tokens = block(tokens, condition)
        parameters = self.output_condition(condition)
        shift, scale = parameters.chunk(2, dim=1)
        tokens = self.output_norm(tokens)
        tokens = modulate(tokens, shift, scale)
        patches = self.output(tokens)

        grid = self.latent_size // self.patch_size
        patches = patches.reshape(batch, grid, grid, self.patch_size, self.patch_size, 4)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        velocity = patches.reshape(batch, 4, self.latent_size, self.latent_size)
        return velocity
