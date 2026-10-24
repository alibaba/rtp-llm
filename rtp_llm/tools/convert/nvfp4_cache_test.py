import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.model_loader.nvfp4_cache import Nvfp4ExpertCache


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

    def test_corrupted_payload_is_rejected(self):
        self.cache.get_or_quantize(self.weight, **self.args)
        path = next(Path(self.temp.name).glob("*.safetensors"))
        with path.open("r+b") as f:
            f.seek(-1, 2)
            value = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum|scales"):
            self.cache.get_or_quantize(self.weight, **self.args)


if __name__ == "__main__":
    unittest.main()
