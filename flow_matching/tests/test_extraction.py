"""Compare extracted components with local snapshots of the original code."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from flow_matching.losses import flow_matching_loss
from flow_matching.models import MLP, MLPConfig, SiT
from flow_matching.interpolants import linear_path
from flow_matching.samplers import euler_sample
from flow_matching.timesteps import sample_timesteps


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BEFORE = PACKAGE_ROOT / ".verification" / "refactor" / "before"


def load_original(topic, name):
    path = BEFORE / topic / f"{name}.py"
    module_name = f"toolbox_original_{topic}_{name}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@unittest.skipUnless((BEFORE / "spiral" / "model.py").exists(), "Local pre-extraction snapshots required")
class ExtractionTests(unittest.TestCase):
    def test_spiral_weights_loss_gradients_updates_and_trajectory(self):
        original_models = load_original("spiral", "model")
        original_flow = load_original("spiral", "flow")
        original_sampling = load_original("spiral", "sampling")
        torch.manual_seed(19)
        original = original_models.ConditionalMLP(original_models.ModelConfig(16, 2))
        torch.manual_seed(19)
        shared = MLP(MLPConfig(16, 2))
        for name, expected in original.state_dict().items():
            torch.testing.assert_close(shared.state_dict()[name], expected, rtol=0, atol=0)
        original_optimizer = torch.optim.Adam(original.parameters(), lr=0.001)
        shared_optimizer = torch.optim.Adam(shared.parameters(), lr=0.001)
        for _ in range(3):
            source = torch.randn(8, 2)
            target = torch.randn_like(source)
            condition = torch.randn(8, 1)
            times = torch.rand(8)
            old_loss = original_flow.flow_matching_loss(original, source, target, times[:, None], condition)
            state, target_velocity = linear_path(source, target, times)
            prediction = shared(state, times, condition)
            new_loss = flow_matching_loss(prediction, target_velocity, reduction="sum_per_sample")
            # Allow FP32 rounding from the original spiral interpolation arithmetic.
            torch.testing.assert_close(new_loss, old_loss, rtol=1e-5, atol=1e-7)
            old_loss.backward()
            new_loss.backward()
            for old_parameter, parameter in zip(original.parameters(), shared.parameters(), strict=True):
                torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=1e-5, atol=1e-7)
            original_optimizer.step()
            shared_optimizer.step()
            original_optimizer.zero_grad()
            shared_optimizer.zero_grad()
        for nfe in (1, 4, 16, 64):
            old_result = original_sampling.sample_euler(original, source, condition, nfe, return_trajectory=True)
            new_result = euler_sample(
                shared, source, condition, nfe, return_trajectory=True, matmul_precision="highest",
            )
            torch.testing.assert_close(new_result.samples, old_result.samples, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(new_result.trajectory, old_result.trajectory, rtol=1e-5, atol=1e-7)
        replacement = MLP(MLPConfig(16, 2))
        replacement.load_state_dict(original.state_dict(), strict=True)

    def test_brightness_weights_loss_gradients_updates_and_samples(self):
        original_models = load_original("brightness", "model")
        original_flow = load_original("brightness", "flow")
        dimensions = {"latent_size": 4, "width": 32, "depth": 2, "heads": 4}
        torch.manual_seed(17)
        original = original_models.ExposureSiT(**dimensions)
        torch.manual_seed(17)
        shared = SiT(**dimensions)
        for name, expected in original.state_dict().items():
            torch.testing.assert_close(shared.state_dict()[name], expected, rtol=0, atol=0)
        # Exercise conditioning and attention through nonzero output/modulation weights.
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.normal_(std=0.03)
        shared.load_state_dict(original.state_dict(), strict=True)
        source = torch.randn(2, 4, 4, 4)
        target = torch.randn_like(source)
        condition = torch.tensor([[-0.75], [0.5]])
        times = torch.tensor([0.25, 0.75])
        old_loss = original_flow.flow_matching_loss(original, source, target, times, condition)
        state, target_velocity = linear_path(source, target, times)
        prediction = shared(state, times, condition)
        new_loss = flow_matching_loss(prediction, target_velocity)
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        old_loss.backward()
        new_loss.backward()
        for old_parameter, parameter in zip(original.parameters(), shared.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=0, atol=0)
        original_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4)
        shared_optimizer = torch.optim.AdamW(shared.parameters(), lr=1e-4)
        original_optimizer.step()
        shared_optimizer.step()
        for nfe in (1, 4, 16):
            expected = original_flow.euler_sample(original, source, condition, nfe)
            actual = euler_sample(shared, source, condition, nfe)
            torch.testing.assert_close(actual.samples, expected, rtol=0, atol=0)
        for setting in ("continuous", 16, 4, 1):
            old_generator = torch.Generator().manual_seed(31)
            new_generator = torch.Generator().manual_seed(31)
            expected = original_flow.sample_timesteps(32, setting, old_generator)
            actual = sample_timesteps(32, setting, new_generator)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(old_generator.get_state(), new_generator.get_state()))

if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
