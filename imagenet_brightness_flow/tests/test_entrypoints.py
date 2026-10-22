"""Opt-in two-GPU launcher, relocated resume, and image sampling check."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image
import torch
import yaml

from imagenet_brightness_flow.src.config import load_config
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT


@unittest.skipUnless(os.environ.get("BRIGHTNESS_RUN_LAUNCH_CHECKS") == "1", "Opt-in two-GPU entry-point check")
class EntrypointTests(unittest.TestCase):
    def test_two_gpu_launch_relocated_resume_and_sample(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        evidence = TOPIC_ROOT / ".verification" / "step2_checks"
        evidence.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence) as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for split, count in (("train", 6), ("val", 3)):
                folder = raw / split / "0000"
                folder.mkdir(parents=True)
                for index in range(count):
                    image = Image.new("RGB", (40, 36), (40 + 20 * index, 90, 60))
                    image.save(folder / f"{index}.png")
            config = load_config(TOPIC_ROOT / "configs" / "experiment.yaml")
            data = replace(
                config.data, root=str(raw), image_size=32, evaluation_exposures=[0],
                preview_images=1, expected_classes=1, expected_train_images=6,
                expected_validation_images=3,
            )
            training = replace(
                config.training, updates=2, global_batch=4, microbatch=1, accumulation=2,
                processes=2, workers=1, log_interval=1, checkpoint_interval=2,
                output_dir=str(root / "original run"),
            )
            config = replace(config, data=data, training=training)
            config_path = root / "settings with spaces.yaml"
            config_path.write_text(yaml.safe_dump(asdict(config)))
            launcher = TOPIC_ROOT / "scripts" / "train.sh"
            command = ["bash", str(launcher), str(config_path)]
            self._execute(command, evidence / "launch_fresh.log")
            original = Path(training.output_dir)
            moved = root / "relocated run"
            original.rename(moved)
            continued = replace(training, updates=3, output_dir=str(moved), resume_from="checkpoint-0000002")
            config = replace(config, training=continued)
            config_path.write_text(yaml.safe_dump(asdict(config)))
            self._execute(command, evidence / "launch_resume.log")
            complete = json.loads((moved / "complete.json").read_text())
            self.assertEqual(complete["step"], 3)
            self.assertEqual(complete["checkpoint"], "checkpoint-0000003")
            output = root / "sample.png"
            sample_command = [
                os.sys.executable, "-m", "imagenet_brightness_flow.scripts.sample",
                "--config", str(config_path), "--weights", str(moved / "averaged.pt"),
                "--source", str(raw / "val" / "0000" / "0.png"),
                "--output", str(output), "--exposure", "0.5", "--nfe", "4",
            ]
            self._execute(sample_command, evidence / "sample.log", cwd=TOPIC_ROOT.parent)
            with Image.open(output) as generated:
                self.assertEqual(generated.size, (32, 32))
                self.assertEqual(generated.mode, "RGB")
            report = {
                "processes": 2, "precision": "bf16", "microbatch": 1, "accumulation": 2,
                "global_batch": 4, "initial_updates": 2, "resumed_updates": 3,
                "relocated_resume": True, "sample_nfe": 4, "sample_size": [32, 32],
            }
            (evidence / "entrypoints.json").write_text(json.dumps(report, indent=2) + "\n")

    def _execute(self, command, log_path, cwd="/tmp"):
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "2"
        environment["MKL_NUM_THREADS"] = "2"
        with log_path.open("w") as stream:
            result = subprocess.run(
                command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                timeout=180,
            )
        self.assertEqual(result.returncode, 0, log_path.read_text()[-12000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
