"""Check private W&B loading, environment precedence, and credential isolation."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from imagenet_brightness_flow.src.config import WandbConfig, _settings, load_config
from imagenet_brightness_flow.src.credentials import load_wandb_credentials
from imagenet_brightness_flow.src.filesystem import TOPIC_ROOT
from imagenet_brightness_flow.src.tracking import WandbTracker


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private = self.root / ".secret" / "wandb.yaml"
        self.private.parent.mkdir()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_missing_file_keeps_environment_unchanged(self):
        load_wandb_credentials("WANDB_API_KEY", self.private)
        self.assertNotIn("WANDB_API_KEY", os.environ)
        self.assertNotIn("WANDB_ENTITY", os.environ)

    def test_private_file_loads_key_and_entity(self):
        self.private.write_text("api_key: fixture-api-key\nentity: fixture-team\n")
        load_wandb_credentials("WANDB_API_KEY", self.private)
        self.assertEqual(os.environ["WANDB_API_KEY"], "fixture-api-key")
        self.assertEqual(os.environ["WANDB_ENTITY"], "fixture-team")

    def test_existing_environment_values_take_priority(self):
        self.private.write_text("api_key: fixture-api-key\nentity: fixture-team\n")
        os.environ["WANDB_API_KEY"] = "exported-key"
        os.environ["WANDB_ENTITY"] = "exported-team"
        load_wandb_credentials("WANDB_API_KEY", self.private)
        self.assertEqual(os.environ["WANDB_API_KEY"], "exported-key")
        self.assertEqual(os.environ["WANDB_ENTITY"], "exported-team")

    def test_null_settings_and_custom_key_variable(self):
        self.private.write_text("api_key: null\nentity: null\n")
        load_wandb_credentials("CUSTOM_WANDB_KEY", self.private)
        self.assertNotIn("CUSTOM_WANDB_KEY", os.environ)
        self.private.write_text("api_key: fixture-api-key\n")
        load_wandb_credentials("CUSTOM_WANDB_KEY", self.private)
        self.assertEqual(os.environ["CUSTOM_WANDB_KEY"], "fixture-api-key")

    def test_invalid_private_settings_do_not_echo_contents(self):
        cases = (
            "api_key: [fixture-private-value\n",
            "api_key: [fixture-private-value]\n",
            "fixture-private-value: true\n",
        )
        for contents in cases:
            with self.subTest(contents=contents):
                self.private.write_text(contents)
                with self.assertRaises(ValueError) as caught:
                    load_wandb_credentials("WANDB_API_KEY", self.private)
                self.assertNotIn("fixture-private-value", str(caught.exception))

    def test_config_and_tracking_keep_credentials_out_of_saved_settings(self):
        self.private.write_text("api_key: fixture-api-key\nentity: fixture-team\n")
        config_path = TOPIC_ROOT / "configs" / "experiment.yaml"
        with patch("imagenet_brightness_flow.src.credentials.REPO_ROOT", self.root):
            config = load_config(config_path)
        serialized = json.dumps(asdict(config))
        self.assertNotIn("fixture-api-key", serialized)
        self.assertNotIn("fixture-team", serialized)
        self.assertIsNone(config.wandb.entity)
        self.assertEqual(config.wandb.api_key_env, "WANDB_API_KEY")

        run = SimpleNamespace(
            id="fixture-run", entity="fixture-team", url="fixture-url",
            define_metric=Mock(),
        )
        sdk = SimpleNamespace(login=Mock(), init=Mock(return_value=run))
        state = SimpleNamespace(wandb_run_id=None)
        with patch.dict("sys.modules", {"wandb": sdk}):
            WandbTracker.start(config, state, self.root, resumed=False)
        sdk.login.assert_called_once_with(key="fixture-api-key", verify=True)
        self.assertEqual(sdk.init.call_args.kwargs["entity"], "fixture-team")
        saved_config = json.dumps(sdk.init.call_args.kwargs["config"])
        self.assertNotIn("fixture-api-key", saved_config)
        self.assertNotIn("fixture-team", saved_config)
        identity = (self.root / "wandb-run.json").read_text()
        self.assertNotIn("fixture-api-key", identity)

    def test_shared_yaml_omits_account_settings_and_retains_strict_validation(self):
        path = TOPIC_ROOT / "configs" / "experiment.yaml"
        values = yaml.safe_load(path.read_text())["wandb"]
        self.assertNotIn("entity", values)
        self.assertNotIn("api_key_env", values)
        settings = _settings(WandbConfig, values)
        self.assertIsNone(settings.entity)
        self.assertEqual(settings.api_key_env, "WANDB_API_KEY")

        missing = dict(values)
        missing.pop("project")
        with self.assertRaises(ValueError):
            _settings(WandbConfig, missing)
        unknown = dict(values, project_typo="fixture")
        with self.assertRaises(ValueError):
            _settings(WandbConfig, unknown)

    def test_previous_config_exports_remain_loadable(self):
        path = TOPIC_ROOT / "configs" / "experiment.yaml"
        values = yaml.safe_load(path.read_text())["wandb"]
        values["entity"] = "previous-team"
        values["api_key_env"] = "PREVIOUS_KEY_VARIABLE"
        settings = _settings(WandbConfig, values)
        self.assertEqual(settings.entity, "previous-team")
        self.assertEqual(settings.api_key_env, "PREVIOUS_KEY_VARIABLE")


if __name__ == "__main__":
    unittest.main()
