import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm import (
    aiter_fmoe_config,
)


class _FakeDistribution:
    def __init__(self, root: Path, version: str):
        self._root = root
        self.version = version

    def locate_file(self, path: str) -> Path:
        return self._root / path


class AiterFmoeConfigTest(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.pop("AITER_CONFIG_FMOE", None)
        aiter_fmoe_config._CONFIG_STATUS = None

    def tearDown(self):
        if self._old_env is not None:
            os.environ["AITER_CONFIG_FMOE"] = self._old_env
        else:
            os.environ.pop("AITER_CONFIG_FMOE", None)
        aiter_fmoe_config._CONFIG_STATUS = None

    def _fixture(self, root: Path):
        config_dir = root / "aiter" / "configs"
        model_config_dir = config_dir / "model_configs"
        model_config_dir.mkdir(parents=True)
        default_config = config_dir / "tuned_fmoe.csv"
        default_config.write_text("shape,run_1stage\ndefault,0\n")
        model_config = model_config_dir / "model_tuned_fmoe.csv"
        model_config.write_text("shape,run_1stage\nmodel,0\n")
        override_config = root / "rtp_override.csv"
        override_config.write_text("shape,run_1stage\ntarget,1\n")
        return default_config, model_config, override_config

    def test_configures_additive_override_and_keeps_model_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, model_config, override_config = self._fixture(root)
            default_hash = hashlib.sha256(default_config.read_bytes()).hexdigest()
            distribution = _FakeDistribution(
                root, aiter_fmoe_config._SUPPORTED_AITER_VERSION
            )
            with mock.patch.object(
                aiter_fmoe_config.importlib.metadata,
                "distribution",
                return_value=distribution,
            ), mock.patch.object(
                aiter_fmoe_config,
                "_SUPPORTED_DEFAULT_CONFIG_SHA256",
                default_hash,
            ), mock.patch.object(
                aiter_fmoe_config, "_OVERRIDE_CONFIG", override_config
            ):
                status = aiter_fmoe_config.configure_aiter_fmoe_overrides()

            self.assertTrue(status.applied)
            self.assertEqual(
                os.environ["AITER_CONFIG_FMOE"].split(os.pathsep),
                [str(default_config), str(model_config), str(override_config)],
            )

    def test_version_change_requires_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._fixture(root)
            distribution = _FakeDistribution(root, "new-aiter-version")
            with mock.patch.object(
                aiter_fmoe_config.importlib.metadata,
                "distribution",
                return_value=distribution,
            ):
                status = aiter_fmoe_config.configure_aiter_fmoe_overrides()

            self.assertFalse(status.applied)
            self.assertIn("review", status.reason)
            with self.assertRaisesRegex(RuntimeError, "Revalidate"):
                aiter_fmoe_config.require_aiter_fmoe_overrides_for_qwen35_tp4()

    def test_explicit_config_must_include_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, _, override_config = self._fixture(root)
            default_hash = hashlib.sha256(default_config.read_bytes()).hexdigest()
            distribution = _FakeDistribution(
                root, aiter_fmoe_config._SUPPORTED_AITER_VERSION
            )
            os.environ["AITER_CONFIG_FMOE"] = str(default_config)
            with mock.patch.object(
                aiter_fmoe_config.importlib.metadata,
                "distribution",
                return_value=distribution,
            ), mock.patch.object(
                aiter_fmoe_config,
                "_SUPPORTED_DEFAULT_CONFIG_SHA256",
                default_hash,
            ), mock.patch.object(
                aiter_fmoe_config, "_OVERRIDE_CONFIG", override_config
            ):
                status = aiter_fmoe_config.configure_aiter_fmoe_overrides()

            self.assertFalse(status.applied)
            self.assertIn("explicitly set", status.reason)


if __name__ == "__main__":
    unittest.main()
