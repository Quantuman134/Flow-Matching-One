"""In-memory ImageNet indexing and fixed validation selection at training startup."""

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import DataConfig
from .paths import resolve_project_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """A dataset-root-relative image path and its sorted class index."""

    relative_path: str
    label: int

    def __post_init__(self):
        path = PurePosixPath(self.relative_path)
        #region Record validation
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
            raise ValueError("Image paths must follow split/class/filename below the dataset root.")
        if path.parts[0] not in ("train", "val"):
            raise ValueError("Image paths must start with train/ or val/.")
        if type(self.label) is not int or self.label < 0:
            raise ValueError("Image labels must be nonnegative integers.")
        #endregion


@dataclass(frozen=True, slots=True)
class PreviewCase:
    record: ImageRecord
    exposure: float


def scan_split(root: Path, split: str) -> tuple[list[ImageRecord], list[str]]:
    """Index class folders lexically; image pixels remain on disk."""
    directory = root / split
    #region Dataset validation
    if split not in ("train", "val") or not directory.is_dir():
        raise ValueError(f"Missing ImageNet split: {directory}")
    #endregion
    classes = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                classes.append(entry.name)
    classes.sort()

    records = []
    for label, class_name in enumerate(classes):
        class_dir = directory / class_name
        names = []
        with os.scandir(class_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    raise ValueError(f"Unexpected nested directory in ImageNet class: {entry.path}")
                suffix = Path(entry.name).suffix
                suffix = suffix.lower()
                if entry.is_file() and suffix in IMAGE_SUFFIXES:
                    names.append(entry.name)
        names.sort()
        if not names:
            raise ValueError(f"ImageNet class has no supported images: {class_dir}")
        for name in names:
            relative_path = f"{split}/{class_name}/{name}"
            record = ImageRecord(relative_path, label)
            records.append(record)

    LOGGER.info("Indexed %s: %d images, %d classes", split, len(records), len(classes))
    return records, classes


def select_per_class(records: list[ImageRecord], count: int, seed: int) -> list[ImageRecord]:
    """Select a fixed number per class using a private seeded RNG."""
    #region Selection validation
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer.")
    #endregion
    groups = {}
    for record in records:
        if record.label not in groups:
            groups[record.label] = []
        groups[record.label].append(record)

    generator = random.Random(seed)
    selected = []
    labels = sorted(groups)
    for label in labels:
        group = groups[label]
        group = sorted(group, key=lambda record: record.relative_path)
        if count > len(group):
            raise ValueError(f"Class {label} has {len(group)} images; requested {count}.")
        choices = generator.sample(group, count)
        choices.sort(key=lambda record: record.relative_path)
        selected.extend(choices)
    return selected


def select_previews(
    monitor: list[ImageRecord], count: int, exposures: list[float], seed: int,
) -> list[PreviewCase]:
    """Use distinct monitor sources and cycle through every exposure value."""
    #region Preview validation
    if not exposures or not len(exposures) <= count <= len(monitor):
        raise ValueError("Preview count must cover exposures and fit inside the monitor subset.")
    #endregion
    generator = random.Random(seed)
    choices = generator.sample(monitor, count)
    previews = []
    for index, record in enumerate(choices):
        exposure_index = index % len(exposures)
        exposure = float(exposures[exposure_index])
        preview = PreviewCase(record, exposure)
        previews.append(preview)
    return previews


def fingerprint_records(records: list[ImageRecord]) -> str:
    """Hash ordered relative paths and labels for the training run record."""
    digest = hashlib.sha256()
    for record in records:
        path_bytes = record.relative_path.encode("utf-8")
        label_bytes = str(record.label)
        label_bytes = label_bytes.encode("ascii")
        digest.update(path_bytes)
        digest.update(b"\0")
        digest.update(label_bytes)
        digest.update(b"\n")
    fingerprint = digest.hexdigest()
    return fingerprint


@dataclass(frozen=True)
class ImageNetIndex:
    root: Path
    classes: list[str]
    training: list[ImageRecord]
    validation: list[ImageRecord]
    monitor: list[ImageRecord]
    previews: list[PreviewCase]
    fingerprints: dict[str, str]


def index_imagenet(config: DataConfig, seed: int) -> ImageNetIndex:
    """Discover source files and validation cases in memory when a run starts."""
    root = resolve_project_path(config.root)
    training, training_classes = scan_split(root, "train")
    validation, validation_classes = scan_split(root, "val")

    #region Dataset validation
    if training_classes != validation_classes:
        raise ValueError("Training and validation class folders must match.")
    if len(training_classes) != config.expected_classes:
        raise ValueError("ImageNet class count differs from data.expected_classes.")
    if len(training) != config.expected_train_images:
        raise ValueError("Training image count differs from data.expected_train_images.")
    if len(validation) != config.expected_validation_images:
        raise ValueError("Validation image count differs from data.expected_validation_images.")
    #endregion
    monitor = select_per_class(validation, config.monitor_per_class, seed)
    previews = select_previews(monitor, config.preview_images, config.evaluation_exposures, seed + 1)
    preview_sources = []
    for preview in previews:
        preview_sources.append(preview.record)
    fingerprints = {}
    fingerprints["training"] = fingerprint_records(training)
    fingerprints["validation"] = fingerprint_records(validation)
    fingerprints["monitor"] = fingerprint_records(monitor)
    fingerprints["preview_sources"] = fingerprint_records(preview_sources)
    index = ImageNetIndex(root, training_classes, training, validation, monitor, previews, fingerprints)
    return index
