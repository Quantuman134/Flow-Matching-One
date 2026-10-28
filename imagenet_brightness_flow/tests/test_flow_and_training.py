"""Focused model/objective/sampler and checkpoint-continuation checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.data import build_datasets
from flow_matching.losses import flow_matching_loss
from flow_matching.models import SiT
from flow_matching.interpolants import linear_path
from flow_matching.samplers import euler_sample
from flow_matching.timesteps import sample_timesteps
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT
from imagenet_brightness_flow.src.state import MicrobatchSampler, SampleOrder, TrainingState
from imagenet_brightness_flow.src.training import train
from imagenet_brightness_flow.src.vae import FrozenVAE


class FixtureAutoencoder(nn.Module):
    """Deterministic downsample/upsample fixture with a trainable dummy weight."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def encode(self, images):
        reduced = torch.nn.functional.avg_pool2d(images, 8)
        channel = reduced[:, :1]
        mean = torch.cat([reduced, channel], dim=1) * self.weight
        distribution = SimpleNamespace(mean=mean)
        return SimpleNamespace(latent_dist=distribution)

    def decode(self, latents):
        values = latents[:, :3] * self.weight
        images = torch.nn.functional.interpolate(values, scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=images)


class FlowAndTrainingTests(unittest.TestCase):
    def setUp(self):
        root = TOPIC_ROOT / ".verification" / "step2_checks"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(TOPIC_ROOT / "configs" / "experiment.yaml")
        self.config = replace(
            self.config, validation=replace(self.config.validation, enabled=False),
            wandb=replace(self.config.wandb, enabled=False),
        )

    def test_timestep_grids_share_draws_and_k1_consumes_rng(self):
        samples = {}
        states = []
        for setting in ("continuous", 16, 4, 1):
            generator = torch.Generator().manual_seed(91)
            samples[setting] = sample_timesteps(2000, setting, generator)
            states.append(generator.get_state())
        for setting in (16, 4, 1):
            expected = torch.floor(samples["continuous"] * setting) / setting
            torch.testing.assert_close(samples[setting], expected, rtol=0, atol=0)
            self.assertEqual(len(samples[setting].unique()), setting)
        for state in states:
            self.assertTrue(torch.equal(state, states[0]))
        self.assertTrue(samples["continuous"].ge(0).all())
        self.assertTrue(samples["continuous"].lt(1).all())

    def test_exact_flow_objective_and_euler_nfe(self):
        class Drift(nn.Module):
            def __init__(self):
                super().__init__()
                self.times = []

            def forward(self, state, times, exposure):
                self.times.append(times.detach().clone())
                return torch.ones_like(state) * exposure[:, :, None, None]

        source = torch.zeros(2, 4, 4, 4)
        exposure = torch.tensor([[0.5], [-0.25]])
        reference = source + exposure[:, :, None, None]
        times = torch.tensor([0.125, 0.75])
        model = Drift()
        state, target_velocity = linear_path(source, reference, times)
        prediction = model(state, times, exposure)
        loss = flow_matching_loss(prediction, target_velocity)
        self.assertEqual(loss.item(), 0)
        for nfe in (1, 4, 16):
            model.times.clear()
            output = euler_sample(model, source, exposure, nfe)
            torch.testing.assert_close(output.samples, reference)
            self.assertEqual(len(model.times), nfe)
            for index, actual in enumerate(model.times):
                self.assertTrue(actual.eq(index / nfe).all())
            self.assertTrue(model.training)

        class GrowingDrift(nn.Module):
            def forward(self, state, times, exposure):
                return state

        model = GrowingDrift()
        source = torch.ones_like(source)
        for nfe in (1, 4, 16):
            output = euler_sample(model, source, exposure, nfe)
            expected = source * (1 + 1 / nfe) ** nfe
            torch.testing.assert_close(output.samples, expected)

    def test_model_shape_zero_initialization_and_exposure_learning(self):
        torch.manual_seed(3)
        model = SiT(latent_size=4, width=32, depth=2, heads=4)
        source = torch.randn(1, 4, 4, 4)
        source = source.repeat(4, 1, 1, 1)
        times = torch.zeros(4)
        exposure = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
        initial = model(source, times, exposure)
        self.assertEqual(initial.shape, source.shape)
        self.assertTrue(initial.eq(0).all())
        reference = source + exposure[:, :, None, None] * 0.2
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        state, target_velocity = linear_path(source, reference, times)
        prediction = model(state, times, exposure)
        before_loss = flow_matching_loss(prediction, target_velocity)
        before = before_loss.item()
        for _ in range(120):
            optimizer.zero_grad()
            prediction = model(state, times, exposure)
            loss = flow_matching_loss(prediction, target_velocity)
            loss.backward()
            optimizer.step()
        prediction = model(state, times, exposure)
        after_loss = flow_matching_loss(prediction, target_velocity)
        after = after_loss.item()
        self.assertLess(after, before * 0.1)
        changed = model(source, times, -exposure)
        original = model(source, times, exposure)
        self.assertGreater((changed - original).abs().mean().item(), 0.01)
        self.assertGreater(model.exposure[0].weight.grad.abs().sum().item(), 0)

    def test_frozen_vae_means_scaling_and_fp32(self):
        vae = FrozenVAE(FixtureAutoencoder(), batch_size=1)
        vae.train()
        self.assertFalse(vae.autoencoder.training)
        self.assertFalse(any(parameter.requires_grad for parameter in vae.parameters()))
        source = torch.full((2, 3, 16, 16), 0.75, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded, reference = vae.encode_pairs(source, source)
            decoded = vae.decode(encoded)
        self.assertEqual(encoded.shape, (2, 4, 2, 2))
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertFalse(encoded.requires_grad)
        torch.testing.assert_close(encoded, reference, rtol=0, atol=0)
        torch.testing.assert_close(encoded, torch.full_like(encoded, 0.5 * 0.18215))
        torch.testing.assert_close(decoded, source)

    def test_prefetch_cursor_and_global_condition_streams(self):
        config = replace(self.config.training, global_batch=8, microbatch=2, processes=2, accumulation=2)
        order = SampleOrder(11, 3)
        first_rank = iter(MicrobatchSampler(order, config, 0))
        second_rank = iter(MicrobatchSampler(order, config, 1))
        observed = []
        for _ in range(2):
            observed.extend(next(first_rank))
            observed.extend(next(second_rank))
        self.assertEqual(observed, order.take(8))
        checkpoint = order.state_dict()
        for _ in range(8):
            next(first_rank)  # Worker prefetch must leave the committed cursor unchanged.
        self.assertEqual(checkpoint, order.state_dict())
        restored = SampleOrder(11, 3)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.take(30), order.take(30))

        states = []
        for setting in ("continuous", 16, 4, 1):
            state = TrainingState(0, 11)
            conditions = state.draw_conditions(8, self.config.data, setting)
            states.append((state, conditions))
        for state, conditions in states:
            self.assertTrue(torch.equal(conditions[0], states[0][1][0]))
            self.assertTrue(torch.equal(conditions[1], states[0][1][1]))
            restored = TrainingState(0, 11)
            restored.load_state_dict(state.state_dict())
            expected = state.draw_conditions(8, self.config.data, "continuous")
            actual = restored.draw_conditions(8, self.config.data, "continuous")
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def _fixture_config(self, output_name, updates, resume_from=None, workers=0):
        raw = self.root / "raw"
        for split, count in (("train", 5), ("val", 3)):
            folder = raw / split / "0000"
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = Image.new("RGB", (20, 18), (30 + 20 * index, 50, 90))
                image.save(folder / f"{index}.png")
        data = replace(
            self.config.data, root=str(raw), image_size=16, evaluation_exposures=[0],
            preview_images=1, expected_classes=1, expected_train_images=5,
            expected_validation_images=3,
        )
        training = replace(
            self.config.training, updates=updates, global_batch=4, microbatch=2, accumulation=2,
            processes=1, device="cpu", mixed_precision="no", workers=workers,
            log_interval=1, checkpoint_interval=2, output_dir=str(self.root / output_name),
            resume_from=resume_from,
        )
        return replace(self.config, data=data, training=training)

    def _train_fixture(self, config):
        torch.manual_seed(config.seed)
        model = SiT(latent_size=2, width=16, depth=1, heads=2)
        vae = FrozenVAE(FixtureAutoencoder())
        datasets = build_datasets(config)
        result = train(config, datasets=datasets, vae=vae, model=model)
        return result

    def test_checkpoint_resume_matches_uninterrupted_with_prefetch(self):
        direct_config = self._fixture_config("direct", 4, workers=2)
        direct = self._train_fixture(direct_config)
        partial_config = self._fixture_config("continued", 2, workers=2)
        partial = self._train_fixture(partial_config)
        continued_config = replace(
            partial_config, training=replace(partial_config.training, updates=4, resume_from=partial.checkpoint.name),
        )
        continued = self._train_fixture(continued_config)
        self.assertEqual(continued.step, 4)
        for name, expected in direct.model.state_dict().items():
            actual = continued.model.state_dict()[name]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, expected in direct.averaged.state_dict().items():
            actual = continued.averaged.state_dict()[name]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        direct_optimizer = torch.load(direct.checkpoint / "optimizer.bin", weights_only=True)
        continued_optimizer = torch.load(continued.checkpoint / "optimizer.bin", weights_only=True)
        for identifier, expected in direct_optimizer["state"].items():
            for name, value in expected.items():
                actual = continued_optimizer["state"][identifier][name]
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
        latest = json.loads((continued.checkpoint.parent / "latest.json").read_text())
        self.assertEqual(latest["checkpoint"], continued.checkpoint.name)
        with self.assertRaises(FileExistsError):
            self._train_fixture(direct_config)
        mismatched = replace(continued_config, seed=17)
        with self.assertRaises(ValueError):
            self._train_fixture(mismatched)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
