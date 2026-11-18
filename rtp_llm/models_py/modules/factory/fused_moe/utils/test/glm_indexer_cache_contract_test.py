"""Paged indexer regression for heterogeneous shared-cache block pitches."""

import sys
import unittest
from unittest import mock

import torch

# The host contracts exercise layout/reference code and mock the native call.
# DeepGEMM's wheel initializes CUDA at import, even on a CPU-only worker.
# Restore just the blocked entry so Triton's native modules stay registered.
_block_deep_gemm = not torch.cuda.is_available() and "deep_gemm" not in sys.modules
if _block_deep_gemm:
    sys.modules["deep_gemm"] = None
try:
    from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score, glm53_indexer_score
finally:
    if _block_deep_gemm:
        del sys.modules["deep_gemm"]

from rtp_llm.models_py.modules.dsv4.fp8._indexer_cache_view import (
    paged_indexer_cache_view,
)


def _inputs(page, device):
    torch.manual_seed(53)
    pages, heads = 6, 32
    pitch = page * 132 + 256
    storage = torch.full((pages, pitch), 173, dtype=torch.uint8, device=device)
    raw = storage[:, : page * 132]
    keys = torch.randn(pages, page, 128, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, page, device=device) * 0.01
    raw[:, : page * 128] = keys.view(torch.uint8).reshape(pages, page * 128)
    raw[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, page * 4)
    pool = raw.view(pages, page, 132)
    q = torch.randn(2, 2, heads, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(4, heads, device=device)
    table = torch.tensor([[5, 1, 3], [0, 4, 2]], dtype=torch.int32, device=device)
    lengths = torch.tensor(
        [[0, page - 1], [page + 3, page * 3]], dtype=torch.int32, device=device
    )
    expected = []
    for batch in range(2):
        k = keys[table[batch].long()].float().reshape(-1, 128)
        sf = scales[table[batch].long()].reshape(-1)
        dots = torch.einsum("nhd,kd->nhk", q[batch].float(), k)
        score = (dots.relu() * weights[batch * 2 : (batch + 1) * 2, :, None]).sum(
            1
        ) * sf
        score.masked_fill_(
            torch.arange(page * 3, device=device) >= lengths[batch, :, None], 0
        )
        expected.append(score)
    return storage, pool, q, weights, table, lengths, torch.cat(expected)


class GlmIndexerCacheContractTest(unittest.TestCase):
    def test_flat_and_pitched_views_alias_the_original_storage(self):
        for page in (16, 32, 64):
            storage, pool, *_ = _inputs(page, "cpu")
            for cache in (pool, pool.unsqueeze(2), pool.contiguous().view(-1, 132)):
                view = paged_indexer_cache_view(cache, page)
                self.assertEqual(view.data_ptr(), cache.data_ptr())
                self.assertEqual(view.shape, (6, page, 1, 132))
                self.assertEqual(
                    view.stride(0), cache.stride(0) if cache.ndim > 2 else page * 132
                )
            self.assertTrue(torch.all(storage[:, page * 132 :] == 173))

    def test_native_dispatch_preserves_block_pitch_without_a_pool_copy(self):
        _, pool, q, weights, table, lengths, _ = _inputs(32, "cpu")
        backend = glm53_indexer_score.fp8_mqa
        with (
            mock.patch.object(glm53_indexer_score, "_HAS_NATIVE_MQA", True),
            mock.patch.object(backend, "get_num_sms", return_value=1),
            mock.patch.object(
                backend, "get_paged_mqa_logits_metadata", return_value=torch.empty(1)
            ),
            mock.patch.object(
                backend, "fp8_paged_mqa_logits", return_value=torch.empty(4, 96)
            ) as kernel,
        ):
            glm53_indexer_score.fp8_paged_indexer_score(
                q, weights, pool, table, lengths, 32, 96
            )
        cache = kernel.call_args.args[1]
        self.assertEqual(cache.data_ptr(), pool.data_ptr())
        self.assertEqual(cache.stride(0), pool.stride(0))
        self.assertFalse(cache.is_contiguous())

    def test_fallback_scores_pitched_block_planar_cache(self):
        storage, pool, q, weights, table, lengths, expected = _inputs(32, "cpu")
        before = storage.clone()
        with mock.patch.object(
            _indexer_score, "supports_paged_mqa_logits", return_value=False
        ):
            actual = _indexer_score.fp8_paged_indexer_score(
                q, weights, pool, table, lengths, 32, 96
            )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_invalid_or_overlapping_pages_are_rejected(self):
        for cache in (
            torch.empty(2, 16, 133, dtype=torch.uint8),
            torch.empty(2, 16, 132, dtype=torch.float32),
            torch.empty(5000, dtype=torch.uint8).as_strided(
                (2, 16, 132), (100, 132, 1)
            ),
        ):
            with self.assertRaises(ValueError):
                paged_indexer_cache_view(cache, 16)


class GlmIndexerCacheGpuTest(unittest.TestCase):
    def test_strided_small_page_and_legacy_kernels_match_reference(self):
        self.assertTrue(torch.cuda.is_available())
        for page in (16, 32):
            storage, pool, q, weights, table, lengths, expected = _inputs(page, "cuda")
            before = storage.clone()
            with (
                mock.patch.object(glm53_indexer_score, "_HAS_NATIVE_MQA", False),
                mock.patch.object(
                    _indexer_score, "supports_paged_mqa_logits", return_value=False
                ),
            ):
                actual = glm53_indexer_score.fp8_paged_indexer_score(
                    q, weights, pool, table, lengths, page, 3 * page
                )
            columns = torch.arange(page * 3, device="cuda")
            valid = columns < lengths.reshape(-1, 1)
            torch.testing.assert_close(
                actual[valid], expected[valid], rtol=3e-5, atol=3e-5
            )
            if page == 16:
                self.assertTrue(torch.isneginf(actual[~valid]).all())
            else:
                self.assertTrue((actual[~valid] == 0).all())
            torch.testing.assert_close(storage, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
