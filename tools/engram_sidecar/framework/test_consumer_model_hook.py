"""CPU harness-boundary tests; this does not execute or mock model forward."""
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "consumer_model_hook_under_test", Path(__file__).with_name("consumer_model_hook.py")
)
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
try:
    import torch
except ImportError:
    torch = None


class ComparisonBoundTest(unittest.TestCase):
    def test_zero_has_only_fixed_absolute_floor(self):
        self.assertEqual(h.bf16_bound(0), 1e-5)
        self.assertEqual(h.bf16_bound(-0.0), 1e-5)

    def test_bf16_two_ulp_bound_tracks_exponent_and_sign(self):
        self.assertEqual(h.bf16_bound(1.0), 2 / 128)
        self.assertEqual(h.bf16_bound(-1.0), 2 / 128)
        self.assertEqual(h.bf16_bound(0.5), 2 / 256)
        self.assertEqual(h.bf16_bound(4.0), 2 / 32)
        self.assertEqual(h.bf16_bound(1e-12), 1e-5)

    def test_nonfinite_bound_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                h.bf16_bound(value)


class FixtureBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="engram-consumer-boundary-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / "consumer_checkpoint"
        self.checkpoint.mkdir()
        self.index = {
            f"layers.{layer}.engram.{suffix}": f"layer-{layer}.safetensors"
            for layer in (0, 1) for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")
        }
        self.write("consumer_checkpoint/config.json", {"checkpoint_path": "/old/location"})
        self.write("consumer_checkpoint/model.safetensors.index.json", {"weight_map": self.index})
        self.write("consumer_checkpoint/layer-0.safetensors", b"fixture boundary placeholder")
        self.write("consumer_checkpoint/layer-1.safetensors", b"fixture boundary placeholder")
        self.write("oracle_inputs.safetensors", b"fixture boundary placeholder")
        self.seal()

    def write(self, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        (self.root / name).write_bytes(raw)

    def seal(self):
        self.metadata = dict(fixture_component_only=True, files_sha256={
            str(path.relative_to(self.root)): h._digest(path.read_bytes())
            for path in self.root.rglob("*") if path.is_file() and path.name != "fixture.json"
        })
        self.write("fixture.json", self.metadata)

    def test_checks_files_and_rebinds_only_runtime_checkpoint_path(self):
        config, _ = h.check_consumer_fixture(self.root)
        self.assertEqual(config["checkpoint_path"], str(self.checkpoint))
        self.assertEqual(json.loads((self.checkpoint / "config.json").read_text())["checkpoint_path"], "/old/location")

    def test_refuses_copied_owner_checkpoint(self):
        (self.root / "owner_checkpoint").mkdir()
        with self.assertRaisesRegex(ValueError, "OWNER_CHECKPOINT_PRESENT"):
            h.check_consumer_fixture(self.root)

    def test_consumer_file_tamper_rejected(self):
        self.write("consumer_checkpoint/layer-1.safetensors", b"changed")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            h.check_consumer_fixture(self.root)

    def test_consumer_index_with_table_rejected_even_with_fresh_hash(self):
        self.index["layers.0.engram.embed.weight"] = "layer-0.safetensors"
        self.write("consumer_checkpoint/model.safetensors.index.json", {"weight_map": self.index})
        self.seal()
        with self.assertRaisesRegex(ValueError, "only eight"):
            h.check_consumer_fixture(self.root)

    def test_undeclared_table_file_rejected(self):
        self.write("consumer_checkpoint/other.safetensors", b"unexpected weight")
        with self.assertRaisesRegex(ValueError, "undeclared tensor"):
            h.check_consumer_fixture(self.root)

    def test_metadata_cannot_read_outside_fixture(self):
        self.metadata["files_sha256"]["consumer_checkpoint/../../outside"] = "unused"
        self.write("fixture.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "relative path"):
            h.check_consumer_fixture(self.root)

    def test_missing_native_fails_before_importing_rtp(self):
        native_dir = self.root / "rtp_llm" / "libs"
        native_dir.mkdir(parents=True)
        (native_dir / "libth_transformer_config.so").touch()
        with mock.patch.object(h, "REPO", self.root), \
             mock.patch.object(h.sys, "path", list(sys.path)), \
             mock.patch.object(h.importlib.util, "find_spec", return_value=None), \
             mock.patch.object(h.importlib, "import_module") as imported:
            with self.assertRaisesRegex(RuntimeError, "REAL_RTP_NATIVE_REQUIRED"):
                h.require_real_native()
        imported.assert_not_called()

    def test_python_fake_native_is_rejected(self):
        native_dir = self.root / "rtp_llm" / "libs"
        native_dir.mkdir(parents=True)
        (native_dir / "libth_transformer_config.so").touch()
        fake = SimpleNamespace(origin="/tmp/libth_transformer_config.py",
                               loader=importlib.machinery.SourceFileLoader("fake", "/tmp/fake.py"))
        with mock.patch.object(h, "REPO", self.root), \
             mock.patch.object(h.sys, "path", list(sys.path)), \
             mock.patch.object(h.importlib.util, "find_spec", return_value=fake), \
             mock.patch.object(h.importlib, "import_module") as imported:
            with self.assertRaisesRegex(RuntimeError, "REAL_RTP_NATIVE_REQUIRED"):
                h.require_real_native()
        imported.assert_not_called()

    def test_external_native_spec_cannot_bypass_missing_bounded_repo_library(self):
        external = self.root / "external" / "libth_transformer_config.so"
        external.parent.mkdir()
        external.touch()
        fake = SimpleNamespace(
            origin=str(external),
            loader=importlib.machinery.ExtensionFileLoader("libth_transformer_config", str(external)),
        )
        with mock.patch.object(h, "REPO", self.root), \
             mock.patch.object(h.sys, "path", [str(external.parent), *sys.path]), \
             mock.patch.object(h.importlib.util, "find_spec", return_value=fake) as found, \
             mock.patch.object(h.importlib, "import_module") as imported:
            with self.assertRaisesRegex(RuntimeError, "bounded config library missing"):
                h.require_real_native()
        found.assert_not_called()
        imported.assert_not_called()


@unittest.skipIf(torch is None, "CPU torch required, no GPU is used")
class TensorComparisonTest(unittest.TestCase):
    def test_accepts_two_ulp_but_rejects_three(self):
        expected = torch.tensor([1.0, 0.5, -1.0], dtype=torch.bfloat16)
        good = torch.tensor([1 + 2 / 128, 0.5 + 2 / 256, -1 - 2 / 128], dtype=torch.bfloat16)
        self.assertLessEqual(h.compare_bf16(good, expected, "two ULP")["max_bound_ratio"], 1)
        bad = torch.tensor([1 + 3 / 128, 0.5, -1.0], dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, "bound exceeded"):
            h.compare_bf16(bad, expected, "three ULP")

    def test_zero_does_not_get_unit_exponent_tolerance(self):
        expected = torch.zeros(1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, "bound exceeded"):
            h.compare_bf16(torch.tensor([0.001], dtype=torch.bfloat16), expected, "zero")

    def test_nonfinite_shape_and_dtype_fail(self):
        expected = torch.zeros(1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            h.compare_bf16(torch.tensor([float("nan")], dtype=torch.bfloat16), expected, "nan")
        with self.assertRaisesRegex(AssertionError, "shape mismatch"):
            h.compare_bf16(torch.zeros(2, dtype=torch.bfloat16), expected, "shape")
        with self.assertRaisesRegex(AssertionError, "dtype"):
            h.compare_bf16(torch.zeros(1), expected, "dtype")

    def test_empty_batch_reports_no_compared_elements(self):
        empty = torch.empty((0, 2, 128), dtype=torch.bfloat16)
        self.assertEqual(h.compare_bf16(empty, empty, "empty")["elements"], 0)


if __name__ == "__main__":
    unittest.main()
