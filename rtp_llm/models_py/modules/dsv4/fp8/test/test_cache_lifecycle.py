"""Lifecycle tests for shared DSV4 RoPE/compressor caches."""

from __future__ import annotations

import gc
import unittest
import weakref
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8 import compressor as compressor_module
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8


class DSV4CacheLifecycleTest(unittest.TestCase):
    def tearDown(self) -> None:
        rope._FREQS_CIS_CACHE.clear()
        compressor_module._SHARED_COS_SIN_CACHE.clear()
        gc.collect()

    def test_rope_cache_reuses_live_tensor_and_evicts_after_unload(self) -> None:
        rope._FREQS_CIS_CACHE.clear()
        args = (8, 16, 0, 10000.0, 1.0, 32, 1)
        first = rope.precompute_freqs_cis(*args, device=torch.device("cpu"))
        second = rope.precompute_freqs_cis(*args, device=torch.device("cpu"))
        self.assertIs(first, second)
        first_ref = weakref.ref(first)

        rope._FREQS_CIS_CACHE.clear()
        self.assertIsNotNone(first_ref())
        self.assertEqual(int(first_ref().numel()), 64)

        del first, second
        gc.collect()
        self.assertIsNone(first_ref())
        self.assertEqual(len(rope._FREQS_CIS_CACHE), 0)

        reloaded = rope.precompute_freqs_cis(
            *args, device=torch.device("cpu")
        )
        self.assertEqual(len(rope._FREQS_CIS_CACHE), 1)
        self.assertEqual(int(reloaded.numel()), 64)

    def test_compressor_cache_reuses_live_tensor_and_evicts_after_unload(self) -> None:
        compressor_module._SHARED_COS_SIN_CACHE.clear()
        freqs = torch.ones(16, 4, dtype=torch.complex64)
        first = CompressorFP8.__new__(CompressorFP8)
        second = CompressorFP8.__new__(CompressorFP8)
        torch.nn.Module.__init__(first)
        torch.nn.Module.__init__(second)

        with mock.patch.object(
            compressor_module,
            "build_cos_sin_cache",
            wraps=compressor_module.build_cos_sin_cache,
        ) as build:
            first.init_rope_cache(freqs)
            second.init_rope_cache(freqs)
            build.assert_called_once_with(freqs)

        self.assertIs(first._cos_sin_cache, second._cos_sin_cache)
        cache_ref = weakref.ref(first._cos_sin_cache)

        compressor_module._SHARED_COS_SIN_CACHE.clear()
        self.assertIsNotNone(cache_ref())
        self.assertEqual(int(second._cos_sin_cache.numel()), 128)

        del first, second, freqs
        gc.collect()
        self.assertIsNone(cache_ref())
        self.assertEqual(len(compressor_module._SHARED_COS_SIN_CACHE), 0)

        reloaded_freqs = torch.ones(16, 4, dtype=torch.complex64)
        reloaded = CompressorFP8.__new__(CompressorFP8)
        torch.nn.Module.__init__(reloaded)
        reloaded.init_rope_cache(reloaded_freqs)
        self.assertEqual(len(compressor_module._SHARED_COS_SIN_CACHE), 1)
        self.assertEqual(int(reloaded._cos_sin_cache.numel()), 128)


if __name__ == "__main__":
    unittest.main()
