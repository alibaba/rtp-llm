import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from rtp_llm.model_loader.nvfp4_cache import Nvfp4ExpertCache, quantize_expert_to_nvfp4


class Nvfp4CacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Nvfp4ExpertCache(self.temp.name)
        self.weight = torch.arange(128, dtype=torch.bfloat16).reshape(4, 32) / 100
        self.args = dict(
            block_size=16,
            input_scale=32 / 2688,
            identity=dict(layer=0, expert=1, weight="w1"),
        )

    def test_reload_is_exact_without_requantizing(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            side_effect=AssertionError("cache hit must not quantize"),
        ):
            got = self.cache.get_or_quantize(self.weight, **self.args)
        for left, right in zip(got, expected):
            self.assertTrue(
                torch.equal(left.float().reshape(-1), right.float().reshape(-1))
            )
        manifest = self.cache.manifest()
        self.assertEqual(len(manifest["entries"]), 1)
        entry = manifest["entries"][0]
        data = (Path(self.temp.name) / entry["file"]).read_bytes()
        self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())

    def test_source_and_activation_contract_invalidate_cache(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        self.cache.get_or_quantize(self.weight + 1, **self.args)
        self.cache.get_or_quantize(
            self.weight, **{**self.args, "input_scale": 1 / 2688}
        )
        self.assertEqual(len(self.cache.manifest()["entries"]), 3)

    def test_algorithm_revision_invalidates_cache(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.ALGORITHM_SHA",
            "revised-rounding-contract",
        ):
            self.cache.get_or_quantize(self.weight, **self.args)
        self.assertEqual(len(self.cache.manifest()["entries"]), 2)

    def test_invalid_scale_or_block_size_is_rejected(self):
        for changes in (
            dict(block_size=32),
            dict(input_scale=0),
            dict(input_scale=float("nan")),
        ):
            with self.assertRaisesRegex(ValueError, "block16 and positive finite"):
                self.cache.get_or_quantize(self.weight, **{**self.args, **changes})
        self.assertFalse(list(Path(self.temp.name).glob("*.safetensors")))

    def _assert_rebuilt(self, expected):
        with self.assertLogs(level="WARNING") as logs, mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            wraps=quantize_expert_to_nvfp4,
        ) as quantize:
            got = self.cache.get_or_quantize(self.weight, **self.args)
        quantize.assert_called_once_with(self.weight, self.args["block_size"])
        self.assertIn("Rebuilding invalid NVFP4 cache", "\n".join(logs.output))
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            side_effect=AssertionError("rebuilt cache must be reusable"),
        ):
            reloaded = Nvfp4ExpertCache(self.temp.name).get_or_quantize(
                self.weight, **self.args
            )
        for actual in (got, reloaded):
            for left, right in zip(actual, expected):
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.shape, right.shape)
                self.assertTrue(
                    torch.equal(
                        left.reshape(-1).view(torch.uint8),
                        right.reshape(-1).view(torch.uint8),
                    )
                )
        self.assertEqual(len(self.cache.manifest()["entries"]), 1)
        self.assertFalse(list(Path(self.temp.name).glob(".nvfp4-*")))

    def test_corrupted_payload_is_rebuilt(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        with path.open("r+b") as f:
            f.seek(-1, 2)
            value = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([value[0] ^ 1]))
        self._assert_rebuilt(expected)

    def test_truncated_file_is_rebuilt(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        path.write_bytes(b"truncated")
        self._assert_rebuilt(expected)

    def test_invalid_metadata_is_rebuilt(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors = {name: f.get_tensor(name).clone() for name in f.keys()}
            metadata = f.metadata()
        for invalid in (
            None,
            {**metadata, "contract": "wrong contract"},
            {"contract": metadata["contract"]},
            {**metadata, "tensor_sha256": "invalid json"},
            {**metadata, "tensor_sha256": "{}"},
        ):
            with self.subTest(metadata=invalid):
                save_file(tensors, str(path), metadata=invalid)
                self._assert_rebuilt(expected)

    def test_invalid_tensors_are_rebuilt(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors = {name: f.get_tensor(name).clone() for name in f.keys()}
            metadata = f.metadata()
        for changed in (
            {name: value for name, value in tensors.items() if name != "packed"},
            {**tensors, "packed": tensors["packed"][:1]},
            {**tensors, "block_scale": tensors["block_scale"].float()},
            {**tensors, "global_scale": torch.tensor([float("nan")])},
        ):
            with self.subTest(tensors=tuple(changed)):
                save_file(changed, str(path), metadata=metadata)
                self._assert_rebuilt(expected)

    def test_read_permission_error_is_not_requantized(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.safe_open",
            side_effect=PermissionError("cache is unreadable"),
        ), mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4"
        ) as quantize:
            with self.assertRaisesRegex(PermissionError, "cache is unreadable"):
                self.cache.get_or_quantize(self.weight, **self.args)
        quantize.assert_not_called()

    @unittest.skipIf(os.geteuid() == 0, "requires non-root file permissions")
    def test_unreadable_file_is_not_requantized(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4"
        ) as quantize:
            with self.assertRaises(OSError):
                self.cache.get_or_quantize(self.weight, **self.args)
        quantize.assert_not_called()

    def test_failed_rebuild_preserves_entry_and_can_retry(self):
        expected = self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        damaged = b"truncated"
        path.write_bytes(damaged)
        with self.assertLogs(level="WARNING"), mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.os.replace",
            side_effect=OSError("cache publication failed"),
        ):
            with self.assertRaisesRegex(OSError, "cache publication failed"):
                self.cache.get_or_quantize(self.weight, **self.args)
        self.assertEqual(path.read_bytes(), damaged)
        self.assertFalse(list(Path(self.temp.name).glob(".nvfp4-*")))
        self._assert_rebuilt(expected)

    def test_quantization_failure_is_not_retried(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        path.write_bytes(b"truncated")
        with self.assertLogs(level="WARNING"), mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            side_effect=RuntimeError("quantization failed"),
        ) as quantize:
            with self.assertRaisesRegex(RuntimeError, "quantization failed"):
                self.cache.get_or_quantize(self.weight, **self.args)
        quantize.assert_called_once()

    def test_failed_rebuild_preserves_concurrent_publication(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        valid = path.read_bytes()
        path.write_bytes(b"truncated")

        def publish_before_read_error(*args, **kwargs):
            path.write_bytes(valid)
            raise SafetensorError("invalid header from the old entry")

        with self.assertLogs(level="WARNING"), mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.safe_open",
            side_effect=publish_before_read_error,
        ), mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            side_effect=RuntimeError("quantization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "quantization failed"):
                self.cache.get_or_quantize(self.weight, **self.args)
        self.assertEqual(path.read_bytes(), valid)
        with mock.patch(
            "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
            side_effect=AssertionError("concurrent publication must be reusable"),
        ):
            self.cache.get_or_quantize(self.weight, **self.args)


if __name__ == "__main__":
    unittest.main()
