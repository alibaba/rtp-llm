"""Tests for the production MegaMoE output-buffer cache."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _MEGA_OUTPUT_CACHE,
    _get_or_create_mega_output,
)


class MegaMoEStaticBufferTest(unittest.TestCase):
    def setUp(self):
        _MEGA_OUTPUT_CACHE.clear()

    def tearDown(self):
        _MEGA_OUTPUT_CACHE.clear()

    def test_reuses_production_cache_and_live_slice_storage(self):
        buf = _get_or_create_mega_output(64, 128, torch.bfloat16, torch.device("cpu"))
        reused = _get_or_create_mega_output(
            16, 128, torch.bfloat16, torch.device("cpu")
        )
        live = reused[:16]

        self.assertIs(reused, buf)
        self.assertEqual(live.shape, (16, 128))
        self.assertEqual(live.data_ptr(), buf.data_ptr())
        live.fill_(1)
        self.assertTrue(torch.all(buf[:16] == 1))

    def test_grows_capacity_and_replaces_cached_storage(self):
        initial = _get_or_create_mega_output(8, 64, torch.bfloat16, torch.device("cpu"))
        grown = _get_or_create_mega_output(32, 64, torch.bfloat16, torch.device("cpu"))
        reused = _get_or_create_mega_output(16, 64, torch.bfloat16, torch.device("cpu"))

        self.assertIsNot(grown, initial)
        self.assertEqual(grown.shape, (32, 64))
        self.assertIs(reused, grown)

    def test_cache_key_separates_hidden_size_and_dtype(self):
        base = _get_or_create_mega_output(4, 32, torch.bfloat16, torch.device("cpu"))
        different_hidden = _get_or_create_mega_output(
            4, 64, torch.bfloat16, torch.device("cpu")
        )
        different_dtype = _get_or_create_mega_output(
            4, 32, torch.float32, torch.device("cpu")
        )

        self.assertIsNot(base, different_hidden)
        self.assertIsNot(base, different_dtype)


if __name__ == "__main__":
    unittest.main()
