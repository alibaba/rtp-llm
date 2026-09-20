"""Tests for the production MegaMoE output-buffer cache."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _MEGA_FP8_BUF_CACHE,
    _MEGA_OUTPUT_CACHE,
    _get_or_create_mega_fp8_buf,
    _get_or_create_mega_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
    _MEGA_FP8_SE_BUF_CACHE,
    _get_or_create_mega_fp8_se_buf,
)


class MegaMoEStaticBufferTest(unittest.TestCase):
    def setUp(self):
        _MEGA_OUTPUT_CACHE.clear()
        _MEGA_FP8_BUF_CACHE.clear()
        _MEGA_FP8_SE_BUF_CACHE.clear()

    def tearDown(self):
        _MEGA_OUTPUT_CACHE.clear()
        _MEGA_FP8_BUF_CACHE.clear()
        _MEGA_FP8_SE_BUF_CACHE.clear()

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

    def test_fp8_routed_buffer_is_cached(self):
        buf = SimpleNamespace(buffer=torch.empty(16, dtype=torch.uint8))
        factory = Mock(return_value=buf)
        deep_gemm = SimpleNamespace(
            mega_fp8=SimpleNamespace(get_symm_buffer_for_mega_moe_fp8=factory)
        )
        group = object()

        with patch.dict("sys.modules", {"deep_gemm": deep_gemm}):
            first = _get_or_create_mega_fp8_buf(group, 512, 256, 8, 4096, 1024)
            second = _get_or_create_mega_fp8_buf(group, 512, 256, 8, 4096, 1024)

        self.assertIs(first, buf)
        self.assertIs(second, buf)
        factory.assert_called_once_with(
            group,
            512,
            256,
            8,
            4096,
            1024,
            num_shared_experts=0,
            use_fp8_dispatch=True,
            activation="swiglu",
        )

    def test_fp8_se_buffer_has_an_independent_cache(self):
        buf = SimpleNamespace(
            buffer=torch.empty(16, dtype=torch.uint8),
            shared_l1_acts_sf=object(),
        )
        factory = Mock(return_value=buf)
        deep_gemm = SimpleNamespace(
            mega_fp8=SimpleNamespace(get_symm_buffer_for_mega_moe_fp8=factory)
        )
        group = object()

        with patch.dict("sys.modules", {"deep_gemm": deep_gemm}):
            first = _get_or_create_mega_fp8_se_buf(
                group, 512, 256, 8, 4096, 1024, 1
            )
            second = _get_or_create_mega_fp8_se_buf(
                group, 512, 256, 8, 4096, 1024, 1
            )

        self.assertIs(first, buf)
        self.assertIs(second, buf)
        factory.assert_called_once_with(
            group,
            512,
            256,
            8,
            4096,
            1024,
            num_shared_experts=1,
            use_fp8_dispatch=True,
            activation="swiglu",
        )


if __name__ == "__main__":
    unittest.main()
