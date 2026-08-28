import csv
import hashlib
import os
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

from rtp_llm.models_py.kernel_tuning.aiter import fmoe


class _FakeDistribution:
    def __init__(self, root: Path, version: str):
        self._root = root
        self.version = version

    def locate_file(self, path: str) -> Path:
        return self._root / path


class AiterFmoeTuningTest(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.pop("AITER_CONFIG_FMOE", None)
        fmoe._CONFIG_STATUS = None

    def tearDown(self):
        if self._old_env is not None:
            os.environ["AITER_CONFIG_FMOE"] = self._old_env
        else:
            os.environ.pop("AITER_CONFIG_FMOE", None)
        fmoe._CONFIG_STATUS = None

    def _write_config(
        self, path: Path, tokens: tuple[int, ...] = (64,), tag: str = ""
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "cu_num": "80",
            "model_dim": "2048",
            "inter_dim": "128",
            "expert": "256",
            "topk": "8",
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float8_e4m3fnuz",
            "q_dtype_w": "torch.float8_e4m3fnuz",
            "q_type": "QuantType.per_Token",
            "use_g1u1": "1",
            "doweight_stage1": "0",
            "_tag": tag,
        }
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(
                destination, fieldnames=(*fmoe._DISPATCH_KEY_FIELDS, "_tag")
            )
            writer.writeheader()
            for token in tokens:
                writer.writerow({**row, "token": str(token)})

    def _fixture(self, root: Path, stock_token: int = 64):
        config_dir = root / "aiter" / "configs"
        default_config = config_dir / "tuned_fmoe.csv"
        model_config = config_dir / "model_configs" / "model_tuned_fmoe.csv"
        overlay_config = root / "rtp_overlay.csv"
        self._write_config(default_config, (stock_token,))
        self._write_config(model_config, (stock_token + 1,))
        self._write_config(overlay_config, fmoe._AFFECTED_TOKEN_BUCKETS)
        return default_config, model_config, overlay_config

    def _configure(self, root: Path, default_config: Path, overlay_config: Path):
        default_hash = hashlib.sha256(default_config.read_bytes()).hexdigest()
        distribution = _FakeDistribution(root, fmoe._SUPPORTED_AITER_VERSION)
        return mock.patch.multiple(
            fmoe,
            _SUPPORTED_DEFAULT_CONFIG_SHA256=default_hash,
            _OVERLAY_CONFIG=overlay_config,
        ), mock.patch.object(
            fmoe.importlib.metadata, "distribution", return_value=distribution
        )

    def test_configures_additive_overlay_and_keeps_model_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, model_config, overlay_config = self._fixture(root)
            config_patch, distribution_patch = self._configure(
                root, default_config, overlay_config
            )
            with config_patch, distribution_patch:
                status = fmoe.configure_aiter_fmoe_overlays()

            self.assertTrue(status.applied)
            self.assertEqual(
                os.environ["AITER_CONFIG_FMOE"].split(os.pathsep),
                [str(default_config), str(model_config), str(overlay_config)],
            )

    def test_stock_dispatch_collision_requires_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, _, overlay_config = self._fixture(root, stock_token=1)
            config_patch, distribution_patch = self._configure(
                root, default_config, overlay_config
            )
            with config_patch, distribution_patch:
                status = fmoe.configure_aiter_fmoe_overlays()

            self.assertFalse(status.applied)
            self.assertIn("overlap", status.reason)

    def test_nonempty_overlay_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, _, overlay_config = self._fixture(root)
            self._write_config(
                overlay_config,
                fmoe._AFFECTED_TOKEN_BUCKETS,
                tag="not-used-for-normal-dispatch",
            )
            config_patch, distribution_patch = self._configure(
                root, default_config, overlay_config
            )
            with config_patch, distribution_patch:
                status = fmoe.configure_aiter_fmoe_overlays()

            self.assertFalse(status.applied)
            self.assertIn("excludes tagged rows", status.reason)

    def test_version_change_requires_review_for_affected_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._fixture(root)
            distribution = _FakeDistribution(root, "new-aiter-version")
            with mock.patch.object(
                fmoe.importlib.metadata,
                "distribution",
                return_value=distribution,
            ):
                status = fmoe.configure_aiter_fmoe_overlays()

            self.assertFalse(status.applied)
            self.assertIn("review", status.reason)
            with self.assertRaisesRegex(RuntimeError, "Revalidate"):
                fmoe.require_aiter_fmoe_tuning(self._affected_signature())

    def test_explicit_config_must_include_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_config, _, overlay_config = self._fixture(root)
            config_patch, distribution_patch = self._configure(
                root, default_config, overlay_config
            )
            os.environ["AITER_CONFIG_FMOE"] = str(default_config)
            with config_patch, distribution_patch:
                status = fmoe.configure_aiter_fmoe_overlays()

            self.assertFalse(status.applied)
            self.assertIn("explicitly set", status.reason)

    def test_signature_is_shape_based_not_model_or_parallelism_based(self):
        signature = self._affected_signature()
        self.assertTrue(fmoe.is_affected_aiter_fmoe_signature(signature))
        self.assertNotIn("model", {field.name for field in fields(signature)})
        self.assertNotIn("tp_size", {field.name for field in fields(signature)})
        self.assertNotIn("ep_size", {field.name for field in fields(signature)})

        for name, value in (
            ("gfx", "gfx950"),
            ("cu_num", 79),
            ("model_dim", 4096),
            ("inter_dim", 256),
            ("expert", 128),
            ("topk", 4),
            ("dtype", "torch.float16"),
            ("use_g1u1", 0),
        ):
            with self.subTest(field=name):
                self.assertFalse(
                    fmoe.is_affected_aiter_fmoe_signature(
                        replace(signature, **{name: value})
                    )
                )

    def test_unaffected_signature_does_not_require_an_active_overlay(self):
        fmoe._CONFIG_STATUS = fmoe._status(False, "inactive")
        fmoe.require_aiter_fmoe_tuning(
            replace(self._affected_signature(), inter_dim=256)
        )

    def test_bundled_overlay_matches_declared_dispatch_rows(self):
        self.assertEqual(
            len(fmoe._validated_overlay_dispatch_keys(fmoe._OVERLAY_CONFIG)),
            len(fmoe._AFFECTED_TOKEN_BUCKETS),
        )

    @staticmethod
    def _affected_signature():
        return next(iter(fmoe._AFFECTED_WORKLOAD_SIGNATURES))


if __name__ == "__main__":
    unittest.main()
