"""Real safetensors NVFP4 artifact tests; all numeric work executes on CPU."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.new_loader_test import fp4_moe_test as fixture
from rtp_llm.models_py.quant_methods import nvfp4_cache
from rtp_llm.models_py.quant_methods.nvfp4_cache import Nvfp4ExpertCache


class Nvfp4ArtifactTest(unittest.TestCase):
    def _load(self, directory):
        return fixture.Fp4MoETest()._load(
            fixture._weights(True),
            online=True,
            online_cache_dir=directory,
            activation_amax=32,
        )

    def _same(self, actual, expected):
        for name, weight in expected.named_parameters():
            torch.testing.assert_close(dict(actual.named_parameters())[name], weight)
        for name, value in expected.named_buffers():
            torch.testing.assert_close(
                dict(actual.named_buffers())[name].reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            )

    def test_real_checkpoint_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                nvfp4_cache,
                "quantize_expert_to_nvfp4",
                wraps=nvfp4_cache.quantize_expert_to_nvfp4,
            ) as quantize:
                expected = self._load(directory)
                self.assertEqual(quantize.call_count, 4)
            self.assertEqual(len(list(Path(directory).glob("*.safetensors"))), 4)
            with mock.patch.object(
                nvfp4_cache,
                "quantize_expert_to_nvfp4",
                side_effect=AssertionError("cache miss"),
            ):
                self._same(self._load(directory), expected)

    def test_real_checkpoint_corrupt_payload_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self._load(directory)
            path = next(Path(directory).glob("*.safetensors"))
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            tensors["packed"].reshape(-1)[0] ^= 1
            save_file(tensors, path, metadata=metadata)
            with mock.patch.object(
                nvfp4_cache,
                "quantize_expert_to_nvfp4",
                wraps=nvfp4_cache.quantize_expert_to_nvfp4,
            ) as quantize:
                self._same(self._load(directory), expected)
                self.assertEqual(quantize.call_count, 1)
            self.assertFalse(list(Path(directory).glob(".nvfp4-*")))

    def test_real_checkpoint_truncated_artifact_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self._load(directory)
            path = next(Path(directory).glob("*.safetensors"))
            path.write_bytes(b"truncated")
            self._same(self._load(directory), expected)

    def test_full_checkpoint_amax_is_part_of_cache_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Nvfp4ExpertCache(directory)
            weight = torch.ones(8, 16, dtype=torch.bfloat16)
            with mock.patch.object(
                nvfp4_cache,
                "quantize_expert_to_nvfp4",
                wraps=nvfp4_cache.quantize_expert_to_nvfp4,
            ) as quantize:
                first = cache.get_or_quantize(
                    weight,
                    block_size=16,
                    input_scale=1.0,
                    identity={"tp_rank": 0},
                    amax=torch.tensor(1.0),
                )
                second = cache.get_or_quantize(
                    weight,
                    block_size=16,
                    input_scale=1.0,
                    identity={"tp_rank": 0},
                    amax=torch.tensor(2.0),
                )
                self.assertEqual(quantize.call_count, 2)
            self.assertEqual(len(list(Path(directory).glob("*.safetensors"))), 2)
            torch.testing.assert_close(second[2], first[2] * 2)

    def test_publication_failure_keeps_correct_in_memory_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self._load(None)
            with mock.patch.object(
                nvfp4_cache.os, "replace", side_effect=OSError("read only")
            ):
                self._same(self._load(directory), expected)
            self.assertFalse(list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
