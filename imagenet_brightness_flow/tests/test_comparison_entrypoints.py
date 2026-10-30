"""Opt-in real two-GPU comparison, relocated report, and interrupt cleanup."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from imagenet_brightness_flow.src.comparison_config import load_comparison
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT


@unittest.skipUnless(os.environ.get('BRIGHTNESS_RUN_COMPARISON_CHECKS') == '1', 'Opt-in two-GPU comparison')
class ComparisonEntrypoints(unittest.TestCase):
    def test_comparison_relocated_report_and_interrupt(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        evidence = TOPIC_ROOT / '.verification/step4_checks'
        evidence.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence) as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            ramp = np.arange(40 * 36 * 3, dtype=np.uint8)
            pixels = ramp.reshape(36, 40, 3)
            for split, count in (('train', 2), ('val', 1)):
                for label in range(17):
                    folder = raw / split / f'{label:04d}'
                    folder.mkdir(parents=True)
                    for index in range(count):
                        image = Image.fromarray(pixels)
                        image.save(folder / f'{index}.png')
            settings, base = load_comparison(TOPIC_ROOT / 'configs/comparison.yaml')
            data = replace(base.data, root=str(raw), image_size=32, expected_classes=17,
                           expected_train_images=34, expected_validation_images=17)
            training = replace(base.training, updates=2, global_batch=4, microbatch=1,
                               accumulation=2, processes=2, workers=0, log_interval=1,
                               checkpoint_interval=2)
            config = replace(base, data=data, training=training,
                             validation=replace(base.validation, enabled=False),
                             wandb=replace(base.wandb, enabled=False, mode='offline'))
            base_path = root / 'base with spaces.yaml'
            base_path.write_text(yaml.safe_dump(asdict(config)))
            settings = replace(settings, base_config=str(base_path), timesteps=[1], nfe=[1, 4],
                               batch_size=4, workers=0, log_interval=2,
                               output_dir=str(root / 'comparison'))
            path = root / 'comparison with spaces.yaml'
            path.write_text(yaml.safe_dump(asdict(settings)))
            command = ['bash', str(TOPIC_ROOT / 'scripts/compare.sh'), str(path)]
            self.execute(command, evidence / 'two-gpu-comparison.log')
            original = Path(settings.output_dir)
            summary = json.loads((original / 'summary.json').read_text())
            self.assertEqual(len(summary['cells']), 2)
            for cell in summary['cells']:
                report = cell['report']
                self.assertEqual(report['source_images'], 17)
                self.assertEqual(report['overall']['comparisons'], 153)
                self.assertEqual(report['provenance']['processes'], 2)
                self.assertEqual(len(report['previews']), 16)
            self.assertTrue(summary['cells'][1]['report']['includes_unsupervised_timesteps'])

            # Report mode reuses run-relative artifacts after moving the comparison.
            moved = root / 'relocated comparison'
            original.rename(moved)
            settings = replace(settings, mode='report', output_dir=str(moved))
            path.write_text(yaml.safe_dump(asdict(settings)))
            self.execute(command, evidence / 'relocated-report.log')
            self.assertTrue((moved / 'report-complete.json').is_file())

            # A new long-budget fixture is interrupted immediately after checkpoint 2.
            training = replace(training, updates=100000)
            config = replace(config, training=training)
            base_path.write_text(yaml.safe_dump(asdict(config)))
            settings = replace(settings, mode='train', output_dir=str(root / 'interrupted'))
            path.write_text(yaml.safe_dump(asdict(settings)))
            job_path = Path(settings.output_dir) / 'configs/timesteps-1.yaml'
            marker = Path(settings.output_dir) / 'runs/timesteps-1/checkpoint-0000002/metadata.yaml'
            log_path = evidence / 'comparison-interrupt.log'
            with log_path.open('w') as stream:
                process = subprocess.Popen(command, cwd='/tmp', stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                started = time.monotonic()
                try:
                    while not marker.is_file():
                        if process.poll() is not None:
                            self.fail(log_path.read_text()[-5000:])
                        if time.monotonic() - started > 90:
                            self.fail('Timed out waiting for a complete fixture checkpoint')
                        time.sleep(0.1)
                    workers = self.workers(job_path)
                    self.assertEqual(len(workers), 2)
                    os.kill(process.pid, signal.SIGINT)
                    exit_code = process.wait(timeout=45)
                    self.assertEqual(exit_code, 130)
                    self.assertFalse(self.workers(job_path))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=45)
            self.assertNotIn('Broken pipe', log_path.read_text())
            record = {'processes': 2, 'source_images': 17, 'exposures': 9, 'cells': 2,
                      'comparisons_per_cell': 153, 'previews_per_cell': 16,
                      'relocated_report': True, 'interrupted_exit_code': exit_code,
                      'remaining_workers': self.workers(job_path)}
            (evidence / 'entrypoints.json').write_text(json.dumps(record, indent=2) + '\n')

    def workers(self, config_path):
        expected = os.fsencode(config_path)
        result = []
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            try:
                arguments = (folder / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if b'imagenet_brightness_flow.scripts.train' in arguments and expected in arguments:
                if b'torch.distributed.run' not in arguments:
                    result.append(int(folder.name))
        return result

    def execute(self, command, log):
        with log.open('w') as stream:
            result = subprocess.run(command, cwd='/tmp', stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        self.assertEqual(result.returncode, 0, log.read_text()[-8000:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
