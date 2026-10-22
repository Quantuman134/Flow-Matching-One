"""Small spiral training/resume check through direct toolbox imports."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from empirical_condition_density.src.data import SpiralConfig
from empirical_condition_density.src.evaluation import generate
from empirical_condition_density.src.training import TrainConfig, train
from flow_matching.models import MLPConfig


class SpiralWorkflowTests(unittest.TestCase):
    def test_spiral_checkpoint_continuation_and_evaluation(self):
        evidence = Path(__file__).resolve().parents[1] / ".verification" / "refactor"
        evidence.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence) as temporary:
            root = Path(temporary)
            config = TrainConfig(
                num_conditions=2, num_steps=4, global_batch_size=8,
                gradient_accumulation_steps=2, device="cpu", log_interval=2,
                validation_interval=2, validation_batch_size=8, checkpoint_interval=2,
                output_dir=str(root / "direct"),
            )
            model_config = MLPConfig(hidden_dim=16, num_hidden_layers=2)
            spiral_config = SpiralConfig(num_targets=24)
            direct = train(config, model_config=model_config, spiral_config=spiral_config)
            partial_config = replace(config, num_steps=2, output_dir=str(root / "continued"))
            partial = train(partial_config, model_config=model_config, spiral_config=spiral_config)
            continued_config = replace(partial_config, num_steps=4)
            continued = train(
                continued_config, model_config=model_config, spiral_config=spiral_config,
                resume_from=partial.checkpoint,
            )
            for name, expected in direct.model.state_dict().items():
                actual = continued.model.state_dict()[name]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            noise = torch.randn(6, 2)
            conditions = torch.zeros(6, 1)
            samples = generate(continued.model, noise, conditions, steps=4, batch_size=3)
            self.assertEqual(samples.shape, (6, 2))
            self.assertTrue(torch.isfinite(torch.from_numpy(samples)).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
