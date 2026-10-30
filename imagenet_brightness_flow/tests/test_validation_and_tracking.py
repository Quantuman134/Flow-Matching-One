"""Streaming aggregation, fixed previews, W&B IDs, and matched pilot inputs."""

from dataclasses import replace
import json
import itertools
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from imagenet_brightness_flow.scripts.pilot import write_pilot_configs
from imagenet_brightness_flow.src.aggregation import MetricAccumulator
from imagenet_brightness_flow.src.brightness import apply_exposure
from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.data import build_datasets
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT
from imagenet_brightness_flow.src.metrics import image_errors
from imagenet_brightness_flow.src.state import TrainingState
from imagenet_brightness_flow.src.tracking import WandbTracker
from imagenet_brightness_flow.src.validation import evaluate, render_previews, save_validation
from imagenet_brightness_flow.src.vae import FrozenVAE
from imagenet_brightness_flow.tests.test_flow_and_training import FixtureAutoencoder


class LocalAccelerator:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1
    is_main_process = True

    def reduce(self, values, reduction):
        return values

    def gather(self, values):
        return values


class ExactExposureVelocity(nn.Module):
    def forward(self, source, times, exposure):
        rgb = (source[:, :3] / 0.18215 + 1) / 2
        rgb = rgb.clamp(0, 1)
        adjusted = apply_exposure(rgb, exposure)
        displacement = (adjusted.image - rgb) * (2 * 0.18215)
        return torch.cat([displacement, displacement[:, :1]], dim=1)


class ValidationFixture(unittest.TestCase):
    def setUp(self):
        evidence = TOPIC_ROOT / ".verification" / "step3_checks"
        evidence.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=evidence)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(TOPIC_ROOT / "configs" / "experiment.yaml")

    def fixture_config(self):
        raw = self.root / "raw"
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[:8, :8] = [30, 60, 90]
        pixels[:8, 8:] = [90, 130, 170]
        pixels[8:, :8] = [200, 180, 160]
        pixels[8:, 8:] = [240, 220, 200]
        for split in ("train", "val"):
            for label in range(17):
                folder = raw / split / f"{label:04d}"
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(2):
                    image = Image.fromarray(pixels)
                    image.save(folder / f"{index}.png")
        data = replace(
            self.config.data, root=str(raw), image_size=16,
            expected_classes=17, expected_train_images=34, expected_validation_images=34,
        )
        training = replace(
            self.config.training, device="cpu", mixed_precision="no", processes=1,
            global_batch=4, microbatch=2, accumulation=2, workers=0,
            updates=2, checkpoint_interval=2, log_interval=1, output_dir=str(self.root / "training"),
        )
        validation = replace(self.config.validation, batch_size=4, workers=0, interval=2)
        wandb = replace(self.config.wandb, enabled=False, mode="offline")
        return replace(self.config, data=data, training=training, validation=validation, wandb=wandb)

class ValidationAndTrackingTests(ValidationFixture):
    def test_streamed_aggregation_counts_masks_and_decompositions(self):
        reference = torch.linspace(0.1, 0.7, 48).reshape(1, 3, 4, 4)
        reference = reference.repeat(3, 1, 1, 1)
        reference[0] = 0.3
        predicted = reference + 0.1
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[1] = True
        values = image_errors(predicted, reference, 1e-8, mask)
        values["latent_mse"] = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
        direct = MetricAccumulator([0, 1], "cpu")
        streamed = MetricAccumulator([0, 1], "cpu")
        for exposure in range(2):
            direct.update(exposure, values)
            for index in range(3):
                selected = {}
                for name, value in values.items():
                    selected[name] = value[index:index + 1]
                streamed.update(exposure, selected)
        torch.testing.assert_close(streamed.totals, direct.totals, rtol=1e-14, atol=1e-14)
        report = streamed.summary()
        overall = report["overall"]
        self.assertEqual(overall["comparisons"], 6)
        self.assertAlmostEqual(overall["rgb_rmse"], 0.1, places=6)
        self.assertAlmostEqual(overall["latent_rmse"], (0.14 / 3) ** 0.5)
        fitted = overall["gain_fit"]
        total = fitted["mean_brightness_mse"] + fitted["global_gain_mse"] + fitted["remaining_spatial_mse"]
        self.assertAlmostEqual(total, fitted["luminance_mse"], places=14)
        self.assertEqual(fitted["count"], 4)
        self.assertEqual(overall["flat_reference"]["count"], 2)
        self.assertEqual(overall["saturation"]["saturated"]["value_count"], 96)
        self.assertEqual(overall["saturation"]["unsaturated"]["value_count"], 192)

    def test_monitor_metrics_previews_and_rng_independence(self):
        config = self.fixture_config()
        datasets = build_datasets(config)
        vae = FrozenVAE(FixtureAutoencoder())
        model = ExactExposureVelocity()
        before = torch.random.get_rng_state()
        clock = itertools.count()
        with (
            patch.object(vae, "encode", wraps=vae.encode) as encode,
            patch.object(vae, "decode", wraps=vae.decode) as decode,
            patch("imagenet_brightness_flow.src.validation.time.perf_counter", side_effect=clock),
        ):
            report = evaluate(model, vae, datasets.monitor, [-1, 0, 1], config, LocalAccelerator())
        # Five source batches, each with three exposure references and generations.
        self.assertEqual(encode.call_count, 20)
        self.assertEqual(decode.call_count, 15)
        latency = report["latency"]
        self.assertEqual(latency["source_images"], 17)
        self.assertEqual(latency["generated_images"], 51)
        self.assertAlmostEqual(latency["source_encoding_ms_per_source"], 5000 / 17)
        self.assertAlmostEqual(latency["integration_ms_per_image"], 15000 / 51)
        self.assertAlmostEqual(latency["decoding_ms_per_image"], 15000 / 51)
        self.assertAlmostEqual(latency["generation_ms_per_image"], 35000 / 51)
        panels = render_previews(model, vae, datasets, config, torch.device("cpu"), step=2, nfe=1)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(report["source_images"], 17)
        self.assertEqual(report["overall"]["comparisons"], 51)
        self.assertLess(report["overall"]["rgb_rmse"], 2e-7)
        self.assertLess(report["overall"]["identity_rmse"], 2e-7)
        for values in report["by_exposure"].values():
            self.assertEqual(values["comparisons"], 17)
        self.assertEqual(len(panels), 16)
        for panel in panels:
            self.assertEqual(panel.pixels.shape, (16, 48, 3))
            self.assertEqual(panel.pixels.dtype, np.uint8)
            self.assertIn("update=2", panel.caption)
        save_validation(self.root, 2, report, panels)
        path = self.root / "validation" / "update-0000002" / "metrics.json"
        saved = json.loads(path.read_text())
        self.assertEqual(len(saved["previews"]), 16)
        self.assertEqual(len(list(path.parent.glob('preview-*.png'))), 16)

    def test_tracking_run_id_resume_and_exactly_sixteen_images(self):
        config = self.fixture_config()
        config = replace(config, wandb=replace(config.wandb, enabled=True, mode="online", entity="unit-test-team"))
        datasets = build_datasets(config)
        panels = render_previews(
            ExactExposureVelocity(), FrozenVAE(FixtureAutoencoder()), datasets, config,
            torch.device("cpu"), step=2, nfe=1,
        )
        state = TrainingState(0, 34)
        state.wandb_run_id = "restored-run"
        state.validation_step = 2
        restored = TrainingState(0, 34)
        restored.load_state_dict(state.state_dict())
        self.assertEqual(restored.validation_step, 2)
        run = SimpleNamespace(
            id="restored-run", entity="unit-test-team", url="https://wandb.ai/unit-test-team/test/restored-run",
            define_metric=Mock(), log=Mock(), finish=Mock(),
        )
        sdk = SimpleNamespace(login=Mock(), init=Mock(return_value=run), Image=Mock(side_effect=lambda pixels, caption: caption))
        with patch.dict('sys.modules', {'wandb': sdk}), patch.dict(os.environ, {'WANDB_API_KEY': 'unit-test-secret'}):
            tracker = WandbTracker.start(config, restored, self.root, resumed=True)
            tracker.validation(2, 8, {"overall": {"rgb_rmse": 0.1}}, panels)
        self.assertEqual(sdk.init.call_args.kwargs['resume'], 'must')
        self.assertEqual(sdk.init.call_args.kwargs['id'], 'restored-run')
        self.assertNotIn('unit-test-secret', json.dumps(sdk.init.call_args.kwargs['config']))
        payload = run.log.call_args.args[0]
        self.assertEqual(len(payload['validation/previews']), 16)
        self.assertEqual(payload['optimizer_update'], 2)
        self.assertNotIn('step', run.log.call_args.kwargs)
        with self.assertRaises(ValueError):
            tracker.validation(2, 8, {}, panels[:15])

    def test_equal_image_pilot_configs_and_consumed_streams(self):
        config = self.fixture_config()
        pilot = replace(config.pilot, global_batches=[4, 8], image_budget=32, warmup_images=8, output_dir=str(self.root / 'pilot'))
        config = replace(config, pilot=pilot)
        root, jobs = write_pilot_configs(config)
        digests = []
        for path, run_dir in jobs:
            job = load_config(path)
            self.assertEqual(job.training.updates * job.training.global_batch, 32)
            self.assertEqual(job.validation.interval, job.training.updates)
            state = TrainingState(job.seed, 34)
            for _ in range(job.training.updates):
                state.draw_conditions(job.training.global_batch, job.data, job.training.timesteps)
                state.order.take(job.training.global_batch)
                state.step += 1
            digests.append(state.stream_digest())
        self.assertEqual(digests[0], digests[1])


@unittest.skipUnless(os.environ.get('BRIGHTNESS_RUN_TRACKING_CHECKS') == '1', 'Opt-in real offline W&B workflow')
class OfflineTrackingTests(ValidationFixture):
    def test_offline_training_and_resumed_id(self):
        import wandb
        from flow_matching.models import SiT
        from imagenet_brightness_flow.src.training import train

        self.addCleanup(wandb.teardown, exit_code=0)
        config = self.fixture_config()
        config = replace(config, wandb=replace(config.wandb, enabled=True))
        torch.manual_seed(config.seed)
        model = SiT(latent_size=2, width=16, depth=1, heads=2)
        vae = FrozenVAE(FixtureAutoencoder())
        partial = train(config, model=model, vae=vae)
        run_record = Path(config.training.output_dir) / 'wandb-run.json'
        first_id = json.loads(run_record.read_text())['run_id']
        continued = replace(config.training, updates=3, resume_from=partial.checkpoint.name)
        config = replace(config, training=continued)
        model = SiT(latent_size=2, width=16, depth=1, heads=2)
        result = train(config, model=model, vae=vae)
        self.assertEqual(json.loads(run_record.read_text())['run_id'], first_id)
        state = torch.load(result.checkpoint / 'custom_checkpoint_0.pkl', weights_only=False)
        self.assertEqual(state['wandb_run_id'], first_id)
        self.assertEqual(state['validation_step'], 3)
        path = run_record.parent / 'validation' / 'update-0000003' / 'metrics.json'
        self.assertEqual(len(json.loads(path.read_text())['previews']), 16)

        # Final evaluation reopens the same SDK run and keeps NFE previews separate.
        from imagenet_brightness_flow.src.comparison_config import load_comparison
        from imagenet_brightness_flow.src.final_evaluation import evaluate_run

        settings, _ = load_comparison(TOPIC_ROOT / 'configs/comparison.yaml')
        settings = replace(settings, nfe=[1, 4], batch_size=8, workers=0, log_interval=10)
        model = SiT(latent_size=2, width=16, depth=1, heads=2)
        evaluate_run(config, settings, model=model, vae=vae)
        identity = json.loads(run_record.read_text())
        self.assertEqual(identity['run_id'], first_id)
        for nfe in settings.nfe:
            path = run_record.parent / 'final-evaluation' / f'nfe-{nfe}' / 'metrics.json'
            report = json.loads(path.read_text())
            self.assertEqual(len(report['previews']), 16)
            self.assertEqual(report['overall']['comparisons'], 34 * 9)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
