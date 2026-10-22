"""Deterministic source preprocessing and fresh online brightness pairs."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms

from .brightness import apply_exposure
from .config import DataConfig, InputConfig
from .indexing import ImageNetIndex, ImageRecord, index_imagenet


def preprocess_image(path: Path, image_size: int) -> torch.Tensor:
    """Return [3,S,S] FP32 sRGB after orientation, bicubic resize, and center crop."""
    #region Preprocessing validation
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    #endregion
    with Image.open(path) as decoded:
        oriented = ImageOps.exif_transpose(decoded)
        rgb = oriented.convert("RGB")
        resized = transforms.resize(
            rgb, image_size, interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        cropped = transforms.center_crop(resized, [image_size, image_size])
        values = transforms.pil_to_tensor(cropped)
    image = values.to(torch.float32)
    image = image / 255
    return image


class ImageNetSources(Dataset):
    """Decode and preprocess each requested source image during a run."""

    def __init__(self, root: Path, records: list[ImageRecord], image_size: int):
        self.root = root
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record.relative_path
        image = preprocess_image(path, self.image_size)
        sample = {
            "image": image,
            "relative_path": record.relative_path,
            "label": record.label,
        }
        return sample


@dataclass(frozen=True)
class ImageNetData:
    index: ImageNetIndex
    training: ImageNetSources
    validation: ImageNetSources
    monitor: ImageNetSources
    preview_sources: ImageNetSources


def build_datasets(config: InputConfig) -> ImageNetData:
    """Build runtime datasets; pixels are decoded lazily by dataset access."""
    index = index_imagenet(config.data, config.seed)
    image_size = config.data.image_size
    training = ImageNetSources(index.root, index.training, image_size)
    validation = ImageNetSources(index.root, index.validation, image_size)
    monitor = ImageNetSources(index.root, index.monitor, image_size)
    preview_records = []
    for preview in index.previews:
        preview_records.append(preview.record)
    preview_sources = ImageNetSources(index.root, preview_records, image_size)
    data = ImageNetData(index, training, validation, monitor, preview_sources)
    return data


@dataclass(frozen=True)
class BrightnessPairs:
    source: torch.Tensor           # [B,3,H,W]; includes shared spatial augmentation.
    reference: torch.Tensor        # [B,3,H,W]; freshly constructed FP32 target.
    exposure: torch.Tensor         # [B,1]; relative exposure in stops.
    saturated: torch.Tensor        # [B,3,H,W]; reference saturation mask.
    horizontal_flip: torch.Tensor  # [B]; sampled augmentation flags.


def sample_pairs(
    sources: torch.Tensor,
    config: DataConfig,
    augmentation_generator: torch.Generator,
    exposure_generator: torch.Generator,
) -> BrightnessPairs:
    """Sample independent per-image flips and exposures using caller-owned RNGs."""
    #region Source validation
    if sources.ndim != 4 or sources.shape[1] != 3 or sources.shape[0] == 0:
        raise ValueError("sources must have nonempty shape [B,3,H,W].")
    if not sources.is_floating_point():
        raise ValueError("sources must contain floating-point sRGB values.")
    if augmentation_generator is exposure_generator:
        raise ValueError("Augmentation and exposure require separate RNG streams.")
    #endregion
    batch_size = sources.shape[0]
    augmentation_values = torch.rand(
        batch_size, generator=augmentation_generator, device=augmentation_generator.device,
    )
    flips = augmentation_values < config.train_flip_probability
    flips = flips.to(sources.device)
    flip_mask = flips[:, None, None, None]
    mirrored = sources.flip(-1)
    source = torch.where(flip_mask, mirrored, sources)
    source = source.to(torch.float32)

    exposure_values = torch.rand(
        (batch_size, 1), generator=exposure_generator, device=exposure_generator.device,
    )
    exposure_range = config.exposure_max - config.exposure_min
    exposure = config.exposure_min + exposure_values * exposure_range
    exposure = exposure.to(sources.device)
    result = apply_exposure(source, exposure)
    pairs = BrightnessPairs(source, result.image, exposure, result.saturated, flips)
    return pairs
