"""Focused checks of dimension-independent paths, loss scales, and sampling."""

import unittest

import torch
from torch import nn

from flow_matching.losses import flow_matching_loss, velocity_mse
from flow_matching.models import MLP, MLPConfig, SiT
from flow_matching.interpolants import linear_path
from flow_matching.samplers import euler_sample
from flow_matching.timesteps import sample_timesteps


class ComponentsTests(unittest.TestCase):
    def test_linear_path_vector_and_image_broadcasting(self):
        for shape in ((3, 5), (3, 4, 2, 2)):
            source = torch.full(shape, 0.25)
            target = torch.full(shape, 0.75)
            times = torch.tensor([0.0, 0.5, 1.0])
            state, velocity = linear_path(source, target, times)
            torch.testing.assert_close(state[0], source[0])
            self.assertTrue(state[1].eq(0.5).all())
            torch.testing.assert_close(state[2], target[2])
            self.assertTrue(velocity.eq(0.5).all())
        with self.assertRaises(ValueError):
            linear_path(source, target, times[:, None])

    def test_reduction_scales_and_fp32_gradient(self):
        prediction = torch.ones(2, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        mean = velocity_mse(prediction, target)
        summed = velocity_mse(prediction, target, reduction="sum_per_sample")
        self.assertEqual(mean.item(), 1)
        self.assertEqual(summed.item(), 3)
        mean.backward()
        expected = torch.full_like(prediction, 1 / 3)
        torch.testing.assert_close(prediction.grad, expected)
        low_precision = prediction.detach().to(torch.bfloat16)
        self.assertEqual(velocity_mse(low_precision, target).dtype, torch.float32)
        with self.assertRaises(ValueError):
            velocity_mse(prediction, target, reduction="sum")

    def test_generic_mlp_dimensions_and_times(self):
        config = MLPConfig(hidden_dim=16, num_hidden_layers=2)
        model = MLP(config, state_dim=5, condition_dim=3)
        source = torch.randn(2, 5)
        target = torch.randn_like(source)
        times = torch.tensor([0.25, 0.75])
        condition = torch.randn(2, 3)
        state, target_velocity = linear_path(source, target, times)
        prediction = model(state, times, condition)
        loss = flow_matching_loss(prediction, target_velocity)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        output = model(source, times, condition)
        self.assertEqual(output.shape, source.shape)
        with self.assertRaises(ValueError):
            model(source, times[:, None], condition)
        self.assertIsInstance(SiT(latent_size=2, width=16, depth=1, heads=2), nn.Module)

    def test_any_positive_grid_and_paired_rng(self):
        generators = []
        for _ in range(3):
            generator = torch.Generator().manual_seed(7)
            generators.append(generator)
        continuous = sample_timesteps(100, "continuous", generators[0])
        thirds = sample_timesteps(100, 3, generators[1])
        single = sample_timesteps(100, 1, generators[2])
        expected = torch.floor(continuous * 3) / 3
        torch.testing.assert_close(thirds, expected, rtol=0, atol=0)
        self.assertTrue(single.eq(0).all())
        for generator in generators:
            self.assertTrue(torch.equal(generator.get_state(), generators[0].get_state()))
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                sample_timesteps(2, invalid, generators[0])

    def test_euler_trajectory_nfe_and_caller_state(self):
        class Drift(nn.Module):
            def __init__(self):
                super().__init__()
                self.times = []

            def forward(self, state, times, condition):
                self.times.append(times.clone())
                return torch.ones_like(state)

        for shape in ((2, 5), (2, 4, 3, 3)):
            model = Drift()
            model.eval()
            source = torch.zeros(shape)
            condition = torch.zeros(2, 1)
            previous_precision = torch.get_float32_matmul_precision()
            result = euler_sample(model, source, condition, 3, return_trajectory=True, matmul_precision="highest")
            self.assertFalse(model.training)
            self.assertEqual(result.num_evaluations, 3)
            self.assertEqual(result.trajectory.shape, (4, *shape))
            torch.testing.assert_close(result.samples, torch.ones_like(source))
            self.assertTrue(source.eq(0).all())
            self.assertEqual(torch.get_float32_matmul_precision(), previous_precision)
            for index, times in enumerate(model.times):
                self.assertEqual(times.shape, (2,))
                self.assertTrue(times.eq(index / 3).all())

    def test_sampler_restores_mode_and_precision_on_failure(self):
        class InvalidVelocity(nn.Module):
            def forward(self, state, times, condition):
                return torch.full_like(state, float("nan"))

        model = InvalidVelocity()
        previous_precision = torch.get_float32_matmul_precision()
        source = torch.zeros(2, 3)
        condition = torch.zeros(2, 1)
        with self.assertRaises(FloatingPointError):
            euler_sample(model, source, condition, matmul_precision="highest")
        self.assertTrue(model.training)
        self.assertEqual(torch.get_float32_matmul_precision(), previous_precision)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
