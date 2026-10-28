"""Focused CPU tests for runtime data loading, exposure, and image errors."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from imagenet_brightness_flow.src.brightness import apply_exposure, linear_to_srgb, srgb_to_linear
from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.data import build_datasets, preprocess_image, sample_pairs
from imagenet_brightness_flow.src.indexing import (
    ImageRecord, fingerprint_records, scan_split, select_per_class, select_previews,
)
from imagenet_brightness_flow.src.metrics import image_errors, latent_mse, luminance_decomposition
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT


class DataAndMetricTests(unittest.TestCase):
    def setUp(self):
        verification_root = TOPIC_ROOT / ".verification" / "step1_checks"
        verification_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=verification_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config_path = TOPIC_ROOT / "configs" / "experiment.yaml"
        self.config = load_config(config_path)
        self.config = replace(
            self.config, validation=replace(self.config.validation, enabled=False),
            wandb=replace(self.config.wandb, enabled=False),
        )

    def test_known_exposure_and_identity(self):
        source = torch.ones(3, 2, 2)
        identity = apply_exposure(source, 0)
        self.assertTrue(torch.equal(identity.image, source))
        half = apply_exposure(source, -1)
        expected = torch.full_like(source, 0.7353569830524495)
        torch.testing.assert_close(half.image, expected)
        self.assertFalse(half.saturated.any())
        double = apply_exposure(source, 1)
        torch.testing.assert_close(double.image, source)
        self.assertTrue(double.saturated.all())

        ramp = torch.linspace(0, 1, 256)
        linear = srgb_to_linear(ramp)
        recovered = linear_to_srgb(linear)
        torch.testing.assert_close(recovered, ramp, rtol=0, atol=2e-7)

    def test_batch_conditions_and_validation(self):
        source = torch.ones(3, 3, 2, 2)
        exposure = torch.tensor([[-1.0], [0.0], [1.0]])
        adjusted = apply_exposure(source, exposure)
        self.assertEqual(adjusted.image.dtype, torch.float32)
        self.assertLess(adjusted.image[0].mean(), adjusted.image[1].mean())
        self.assertFalse(adjusted.saturated[0].any())
        self.assertTrue(adjusted.saturated[2].all())
        with self.assertRaises(ValueError):
            apply_exposure(source, torch.ones(2))
        with self.assertRaises(ValueError):
            apply_exposure(source + 1, 0)
        with self.assertRaises(ValueError):
            apply_exposure(source, float("nan"))

    def test_pairs_have_separate_reproducible_rng_streams(self):
        source = torch.linspace(0, 1, 24)
        source = source.reshape(2, 3, 2, 2)
        before = source.clone()
        augmentation = torch.Generator().manual_seed(11)
        exposure = torch.Generator().manual_seed(22)
        first = sample_pairs(source, self.config.data, augmentation, exposure)
        augmentation = torch.Generator().manual_seed(11)
        exposure = torch.Generator().manual_seed(22)
        repeated = sample_pairs(source, self.config.data, augmentation, exposure)
        torch.testing.assert_close(first.source, repeated.source)
        torch.testing.assert_close(first.reference, repeated.reference)
        torch.testing.assert_close(first.exposure, repeated.exposure)
        next_pair = sample_pairs(source, self.config.data, augmentation, exposure)
        self.assertFalse(torch.equal(next_pair.exposure, repeated.exposure))
        torch.testing.assert_close(source, before)
        expected = apply_exposure(first.source, first.exposure)
        torch.testing.assert_close(first.reference, expected.image)

    def test_preprocessing_honors_exif_and_rgb_shape(self):
        pixels = np.zeros((20, 36, 3), dtype=np.uint8)
        pixels[:, :18, 0] = 255
        pixels[:, 18:, 2] = 255
        image = Image.fromarray(pixels)
        exif = image.getexif()
        exif[274] = 6  # Clockwise quarter-turn: left/right colors become top/bottom.
        path = self.root / "oriented.jpg"
        image.save(path, quality=100, exif=exif)
        source = preprocess_image(path, 16)
        self.assertEqual(source.shape, (3, 16, 16))
        self.assertEqual(source.dtype, torch.float32)
        self.assertGreater(source[0, :4].mean(), source[2, :4].mean())
        self.assertGreater(source[2, -4:].mean(), source[0, -4:].mean())
        grayscale = Image.new("L", (30, 20), 128)
        path = self.root / "gray.png"
        grayscale.save(path)
        source = preprocess_image(path, 16)
        torch.testing.assert_close(source[0], source[1])
        torch.testing.assert_close(source[1], source[2])

    def _fixture_records(self):
        for split, count in [("train", 3), ("val", 2)]:
            for label in range(3):
                directory = self.root / "raw" / split / f"{label:04d}"
                directory.mkdir(parents=True)
                for index in range(count):
                    path = directory / f"{index}.png"
                    image = Image.new("RGB", (12, 10), (label * 40, index * 30, 100))
                    image.save(path)
        root = self.root / "raw"
        records, classes = scan_split(root, "val")
        return root, records, classes

    def test_index_selection_is_sorted_and_reproducible(self):
        root, records, classes = self._fixture_records()
        self.assertEqual(classes, ["0000", "0001", "0002"])
        self.assertEqual(len(records), 6)
        monitor = select_per_class(records, 1, 0)
        repeated = select_per_class(records, 1, 0)
        self.assertEqual(monitor, repeated)
        self.assertEqual(len(monitor), 3)
        self.assertTrue(set(monitor).issubset(records))
        previews = select_previews(monitor, 3, [-1, 0, 1], 1)
        self.assertEqual([case.exposure for case in previews], [-1, 0, 1])
        first = fingerprint_records(records)
        repeated, classes = scan_split(root, "val")
        second = fingerprint_records(repeated)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            ImageRecord("../escape.jpg", 0)

    def test_runtime_datasets_are_lazy_and_preserve_validation(self):
        root, records, classes = self._fixture_records()
        data = replace(
            self.config.data, root=str(root), image_size=16, evaluation_exposures=[-1, 0, 1],
            preview_images=3, expected_classes=3, expected_train_images=9,
            expected_validation_images=6,
        )
        config = replace(self.config, data=data)
        config_path = self.root / "settings with spaces.yaml"
        values = asdict(config)
        with config_path.open("w") as stream:
            yaml.safe_dump(values, stream)
        loaded_config = load_config(config_path)
        before = sorted(self.root.rglob("*"))
        with patch("imagenet_brightness_flow.src.data.preprocess_image", wraps=preprocess_image) as decode:
            datasets = build_datasets(loaded_config)
            self.assertEqual(decode.call_count, 0)
            source = datasets.training[0]
            self.assertEqual(decode.call_count, 1)
        after = sorted(self.root.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(source["image"].shape, (3, 16, 16))
        self.assertEqual(len(datasets.training), 9)
        self.assertEqual(len(datasets.validation), 6)
        self.assertEqual(len(datasets.monitor), 3)
        self.assertTrue(set(datasets.index.monitor).issubset(datasets.index.validation))
        self.assertEqual(len(datasets.preview_sources), 3)
        bad_data = replace(data, expected_train_images=10)
        bad_config = replace(config, data=bad_data)
        with self.assertRaises(ValueError):
            build_datasets(bad_config)

    def test_luminance_offset_gain_local_and_flat_cases(self):
        reference = torch.linspace(0.1, 0.5, 256, dtype=torch.float64)
        reference = reference.reshape(1, 16, 16)
        for output in [reference + 0.1, 0.7 * reference]:
            values = luminance_decomposition(output, reference, 1e-8)
            total = values["mean_brightness_mse"] + values["global_gain_mse"]
            total = total + values["remaining_spatial_mse"]
            torch.testing.assert_close(total, values["luminance_mse"])
            self.assertLess(values["remaining_spatial_mse"].item(), 1e-25)
        output = reference.clone()
        output[:, 4:8, 4:8] += 0.1
        values = luminance_decomposition(output, reference, 1e-8)
        self.assertGreater(values["remaining_spatial_mse"].item(), 0)
        total = values["mean_brightness_mse"] + values["centered_luminance_mse"]
        torch.testing.assert_close(total, values["luminance_mse"])
        flat = torch.full_like(reference, 0.5)
        values = luminance_decomposition(flat + 0.1, flat, 1e-8)
        self.assertFalse(values["gain_valid"].any())
        self.assertTrue(values["global_gain_mse"].isnan().all())
        torch.testing.assert_close(values["mean_brightness_mse"], values["luminance_mse"])

    def test_clipping_masks_and_latent_errors(self):
        reference = torch.ones(2, 3, 2, 2)
        reference[0] = 0.5
        output = torch.full_like(reference, 1.25)
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[1] = True
        values = image_errors(output, reference, 1e-8, mask)
        expected = torch.tensor([0.25, 0], dtype=torch.float64)
        torch.testing.assert_close(values["rgb_mse"], expected)
        expected = torch.tensor([0.5625, 0.0625], dtype=torch.float64)
        torch.testing.assert_close(values["raw_rgb_mse"], expected)
        self.assertTrue(values["range_violation_fraction"].eq(1).all())
        self.assertTrue(values["saturated_rgb_mse"][0].isnan())
        self.assertEqual(values["saturated_value_count"][1].item(), 12)
        self.assertEqual(values["unsaturated_value_count"][0].item(), 12)
        latent_reference = torch.zeros(2, 4, 2, 2)
        latent_output = torch.ones_like(latent_reference)
        latent_values = latent_mse(latent_output, latent_reference)
        torch.testing.assert_close(latent_values, torch.ones(2, dtype=torch.float64))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
