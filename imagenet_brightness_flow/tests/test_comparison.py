"""Full-split cells, timestep flags, portable resume, and comparison reports."""

from dataclasses import asdict, replace
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from imagenet_brightness_flow.src.artifacts import write_json
from imagenet_brightness_flow.src.comparison import latest_checkpoint, prepare_comparison
from imagenet_brightness_flow.src.comparison_config import load_comparison, timestep_coverage
from imagenet_brightness_flow.src.data import build_datasets
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT
from imagenet_brightness_flow.src.final_evaluation import evaluate_run
from imagenet_brightness_flow.src.reporting import export_comparison
from imagenet_brightness_flow.src.training import run_signature
from imagenet_brightness_flow.src.tracking import WandbTracker
from imagenet_brightness_flow.src.vae import FrozenVAE
from imagenet_brightness_flow.tests.test_flow_and_training import FixtureAutoencoder
from imagenet_brightness_flow.tests.test_validation_and_tracking import (
    ExactExposureVelocity, LocalAccelerator, ValidationFixture,
)


class FixtureAccelerator(LocalAccelerator):
    def wait_for_everyone(self):
        pass


class ComparisonTests(ValidationFixture):
    def setUp(self):
        # All fixture data and generated reports stay in this topic's verification folder.
        evidence = TOPIC_ROOT / ".verification" / "step4_checks"
        evidence.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=evidence)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _, self.config = load_comparison(TOPIC_ROOT / "configs" / "comparison.yaml")
        self.config = self.fixture_config()
        self.settings, _ = load_comparison(TOPIC_ROOT / "configs" / "comparison.yaml")
        self.settings = replace(
            self.settings, output_dir=str(self.root / "comparison"), batch_size=8,
            workers=0, log_interval=10,
        )

    def make_completed_models(self):
        root, jobs = prepare_comparison(self.settings, self.config)
        datasets = build_datasets(self.config)
        for _, job in jobs:
            model = ExactExposureVelocity()
            run_dir = Path(job.training.output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            signature = run_signature(job, datasets, model)
            weights = {"model": model.state_dict(), "signature": signature, "step": job.training.updates}
            torch.save(weights, run_dir / "averaged.pt")
            write_json(run_dir / "complete.json", {
                "step": job.training.updates, "averaged": "averaged.pt", "checkpoint": "fixture-checkpoint",
            })
            write_json(run_dir / "performance.json", {
                "initial_model_sha256": "fixture-initialization", "stream_sha256": "fixture-stream",
            })
            events = [
                {"event": "train", "step": 1, "velocity_mse": 0.1},
                {"event": "validation", "step": 2, "rgb_rmse": 0.05},
            ]
            with (run_dir / "metrics.jsonl").open("w") as stream:
                for event in events:
                    stream.write(json.dumps(event) + "\n")
        return root, jobs, datasets

    def evaluate_jobs(self, jobs, datasets, settings=None):
        if settings is None:
            settings = self.settings
        for _, job in jobs:
            evaluate_run(
                job, settings, datasets=datasets, model=ExactExposureVelocity(),
                vae=FrozenVAE(FixtureAutoencoder()), accelerator=FixtureAccelerator(),
            )

    def test_protocol_configs_and_all_timestep_flags(self):
        root, jobs = prepare_comparison(self.settings, self.config)
        self.assertEqual(len(jobs), 4)
        for path, job in jobs:
            self.assertTrue(path.is_file())
            self.assertEqual(job.seed, self.config.seed)
            self.assertEqual(job.training.global_batch, self.config.training.global_batch)
            self.assertEqual(job.training.learning_rate, self.config.training.learning_rate)
        restored, _ = load_comparison(root / "resolved-comparison.yaml")
        self.assertEqual(restored.output_dir, str(root))
        flags = []
        for timesteps in self.settings.timesteps:
            for nfe in self.settings.nfe:
                coverage = timestep_coverage(timesteps, nfe)
                self.assertEqual(coverage['euler_times'][0], 0)
                self.assertLess(coverage['euler_times'][-1], 1)
                if coverage['includes_unsupervised_timesteps']:
                    flags.append((timesteps, nfe))
        self.assertEqual(flags, [(4, 16), (1, 4), (1, 16)])
        incompatible = replace(self.config.training, updates=3)
        incompatible = replace(self.config, training=incompatible)
        with self.assertRaises(ValueError):
            prepare_comparison(self.settings, incompatible)
        with self.assertRaises(FileExistsError):
            prepare_comparison(replace(self.settings, resume=False), self.config)

    def test_only_complete_checkpoints_resume(self):
        root, jobs = prepare_comparison(self.settings, self.config)
        run_dir = Path(jobs[0][1].training.output_dir)
        partial = run_dir / 'checkpoint-0000002'
        partial.mkdir(parents=True)
        complete = run_dir / 'checkpoint-0000001'
        complete.mkdir()
        (complete / 'metadata.yaml').write_text('step: 1\n')
        self.assertEqual(latest_checkpoint(run_dir), complete.name)
        _, resumed = prepare_comparison(self.settings, self.config)
        self.assertEqual(resumed[0][1].training.resume_from, complete.name)

    def test_twelve_full_split_cells_reports_relocation_and_resume(self):
        root, jobs, datasets = self.make_completed_models()
        self.evaluate_jobs(jobs, datasets)
        summary = export_comparison(root, self.settings, jobs)
        self.assertEqual(len(summary['cells']), 12)
        count = self.config.data.expected_validation_images
        exposure_count = len(self.config.data.evaluation_exposures)
        for cell in summary['cells']:
            report = cell['report']
            self.assertEqual(report['source_images'], count)
            self.assertEqual(report['overall']['comparisons'], count * exposure_count)
            self.assertEqual(len(report['previews']), 16)
            for metrics in report['by_exposure'].values():
                self.assertEqual(metrics['comparisons'], count)
            if report['nfe'] == 1:
                self.assertLess(report['overall']['rgb_rmse'], 2e-7)
        with (root / 'tables/overall.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        with (root / 'tables/by-exposure.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12 * exposure_count)
        self.assertEqual(len(list((root / 'figures').glob('*.png'))), 4)
        self.assertIn('†', (root / 'report.md').read_text())
        # Retain a small, explicitly synthetic report preview for visual review.
        preview = TOPIC_ROOT / '.verification/step4_checks/fixture-report-preview'
        shutil.copytree(root / 'figures', preview / 'figures', dirs_exist_ok=True)
        shutil.copytree(root / 'tables', preview / 'tables', dirs_exist_ok=True)
        shutil.copyfile(root / 'report.md', preview / 'report.md')

        # Completed cell references remain valid after the entire comparison moves.
        moved = root.with_name('relocated comparison')
        root.rename(moved)
        restored_settings, restored_base = load_comparison(moved / 'resolved-comparison.yaml')
        self.assertEqual(restored_settings.output_dir, str(moved))
        self.assertEqual(restored_base.seed, self.config.seed)
        moved_settings = replace(self.settings, output_dir=str(moved))
        _, moved_jobs = prepare_comparison(moved_settings, self.config)
        with patch('imagenet_brightness_flow.src.final_evaluation.evaluate') as evaluate:
            self.evaluate_jobs(moved_jobs, datasets, moved_settings)
            evaluate.assert_not_called()
        export_comparison(moved, moved_settings, moved_jobs)

        # A missing panel repeats its whole cell; its siblings remain reusable.
        missing = moved / 'runs/timesteps-1/final-evaluation/nfe-1/preview-00.png'
        missing.unlink()
        from imagenet_brightness_flow.src.validation import evaluate as actual_evaluate
        with patch('imagenet_brightness_flow.src.final_evaluation.evaluate', wraps=actual_evaluate) as evaluate:
            self.evaluate_jobs(moved_jobs, datasets, moved_settings)
            self.assertEqual(evaluate.call_count, 1)
        self.assertTrue(missing.is_file())

        # Replacing the weight file invalidates completed reports even at the same path.
        weights_path = moved / 'runs/timesteps-1/averaged.pt'
        weights = torch.load(weights_path, weights_only=True)
        weights['changed_file'] = True
        torch.save(weights, weights_path)
        with self.assertRaises(ValueError):
            self.evaluate_jobs(moved_jobs[-1:], datasets, moved_settings)

    def test_final_tracking_keeps_nfe_previews_separate(self):
        from imagenet_brightness_flow.src.validation import render_previews

        datasets = build_datasets(self.config)
        panels = render_previews(
            ExactExposureVelocity(), FrozenVAE(FixtureAutoencoder()), datasets,
            self.config, torch.device('cpu'), step=2, nfe=1,
        )
        run = SimpleNamespace(log=Mock())
        sdk = SimpleNamespace(Image=Mock(side_effect=lambda pixels, caption: caption))
        tracker = WandbTracker(run, sdk, 16)
        for nfe in (1, 4, 16):
            tracker.final_evaluation(2, 8, nfe, {'overall': {'rgb_rmse': 0.1}}, panels)
            payload = run.log.call_args.args[0]
            self.assertEqual(len(payload[f'final/nfe-{nfe}/previews']), 16)
            self.assertEqual(payload[f'final/nfe-{nfe}/overall/rgb_rmse'], 0.1)
            self.assertEqual(payload['optimizer_update'], 2)
        with self.assertRaises(ValueError):
            tracker.final_evaluation(2, 8, 1, {}, panels[:15])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
