import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    _fp8_paged_indexer_score_sm120,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import _decode_sched_meta
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
    run_chunked_reference,
    validate_sm120_swa_topk_width,
)


class Sm120SparseMlaCanonicalTest(unittest.TestCase):
    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    @patch("rtp_llm.models_py.modules.dsv4.fp8.attention.is_sm120")
    def test_sm120_decode_does_not_build_flash_mla_metadata(
        self, mock_is_sm120, mock_build
    ):
        mock_is_sm120.return_value = True
        metadata = object()

        result = _decode_sched_meta(
            torch.device("cuda", 0),
            metadata,
            batch_size=2,
            q_len=1,
            num_heads=8,
            topk=64,
            extra_attn_type=None,
        )

        self.assertIsNone(result)
        mock_build.assert_not_called()

    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    @patch("rtp_llm.models_py.modules.dsv4.fp8.attention.is_sm120")
    def test_non_sm120_decode_builds_flash_mla_metadata(
        self, mock_is_sm120, mock_build
    ):
        mock_is_sm120.return_value = False
        sentinel = object()
        mock_build.return_value = sentinel
        metadata = object()

        result = _decode_sched_meta(
            torch.device("cuda", 0),
            metadata,
            batch_size=2,
            q_len=1,
            num_heads=8,
            topk=64,
            extra_attn_type="HCA_KV",
        )

        self.assertIs(result, sentinel)
        mock_build.assert_called_once_with(
            metadata,
            batch_size=2,
            q_len=1,
            num_heads=8,
            topk=64,
            extra_attn_type="HCA_KV",
        )

    def test_none_length_compacts_non_negative_slots(self):
        indices = torch.tensor([[11, -1, 13, -1], [-1, 7, 8, 9]], dtype=torch.int64)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.dtype, torch.int32)
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(canonical.tolist(), [[11, 13, -1, -1], [7, 8, 9, -1]])

    def test_short_explicit_length_is_preserved(self):
        indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([2]), (4, 8))
        self.assertEqual(canonical.tolist(), [[1, 2, -1, -1]])
        self.assertEqual(lengths.tolist(), [2])

    def test_unsupported_width_padding_keeps_invalid_slots(self):
        indices = torch.tensor([[4, -1, 9]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.shape, (1, 4))
        self.assertEqual(canonical.tolist(), [[4, 9, -1, -1]])
        self.assertEqual(lengths.tolist(), [2])

    def test_explicit_length_ignores_holes_and_tail(self):
        indices = torch.tensor([[5, -1, 7, 8]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([2]), (4, 8))
        self.assertEqual(canonical.tolist(), [[5, -1, -1, -1]])
        self.assertEqual(lengths.tolist(), [1])

    def test_length_is_clamped_to_kernel_width(self):
        indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([99]), (4, 8))
        self.assertEqual(canonical.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(lengths.tolist(), [4])

    def test_width_overflow_fails_with_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds the largest"):
            canonical_topk(torch.zeros(1, 9, dtype=torch.int32), None, (4, 8))

    def test_normal_decode_width_is_validated_before_first_request(self):
        self.assertEqual(
            validate_sm120_swa_topk_width(513, context="DeepSeek-V4 decode"),
            1024,
        )
        self.assertEqual(
            validate_sm120_swa_topk_width(1025, context="DeepSeek-V4 decode"),
            1025,
        )

    def test_dspark_gamma_is_included_in_startup_width_validation(self):
        window = 1024
        gamma = 3
        dspark_width = ((window + gamma + 127) // 128) * 128
        self.assertEqual(dspark_width, 1152)
        self.assertEqual(
            validate_sm120_swa_topk_width(
                dspark_width,
                context="DeepSeek-V4 DSpark decode",
            ),
            1152,
        )

    def test_chunked_large_width_fallback_matches_dense_softmax(self):
        torch.manual_seed(7)
        rows, heads, dim = 2, 2, 512
        query = torch.randn(rows, heads, dim, dtype=torch.float32) / 16
        cache_rows = torch.randn(8, dim, dtype=torch.float32) / 16
        indices = torch.tensor([[0, 2, 4, -1, -1], [1, 3, 5, 7, -1]])
        lengths = torch.tensor([3, 4], dtype=torch.int32)
        sinks = torch.tensor([0.2, -0.3], dtype=torch.float32)
        actual = torch.empty_like(query)

        def fake_dequantize(_cache, slots):
            safe = slots.clamp_min(0).to(torch.long)
            result = cache_rows.index_select(0, safe)
            return result.masked_fill((slots < 0).unsqueeze(1), 0.0)

        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton."
            "dequantize_slots_to_bf16",
            side_effect=fake_dequantize,
        ):
            run_chunked_reference(
                query=query,
                swa_cache=torch.empty(0),
                swa_indices=indices,
                swa_lens=lengths,
                out=actual,
                scale=0.5,
                sinks=sinks,
                chunk_size=2,
            )

        expected = torch.empty_like(actual)
        for row in range(rows):
            selected = cache_rows.index_select(
                0, indices[row, : lengths[row]].to(torch.long)
            )
            logits = torch.einsum("hd,kd->hk", query[row], selected) * 0.5
            denom_logits = torch.cat((logits, sinks[:, None]), dim=1)
            weights = torch.softmax(denom_logits, dim=1)[:, :-1]
            expected[row] = torch.einsum("hk,kd->hd", weights, selected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_hca_long_context_widths_cover_one_million_tokens(self):
        for source_width, expected_width in (
            (0, 2),
            (1, 2),
            (1562, 2048),
            (2049, 4096),
            (4097, 8192),
            (8192, 8192),
        ):
            with self.subTest(source_width=source_width):
                indices = torch.arange(source_width, dtype=torch.int32).view(1, -1)
                canonical, lengths = canonical_topk(
                    indices, None, SM120_EXTRA_TOPK_WIDTHS
                )
                self.assertEqual(canonical.shape, (1, expected_width))
                self.assertEqual(lengths.tolist(), [source_width])

        with self.assertRaisesRegex(RuntimeError, "exceeds the largest"):
            canonical_topk(
                torch.zeros(1, 8193, dtype=torch.int32),
                None,
                SM120_EXTRA_TOPK_WIDTHS,
            )


class Sm120IndexerFallbackTest(unittest.TestCase):
    @staticmethod
    def _pool(num_blocks: int, block_size: int) -> torch.Tensor:
        # Zero K/scales are sufficient to exercise address/mask logic; the
        # expected score for every valid token is therefore exactly zero.
        pool = torch.zeros(
            num_blocks * block_size,
            132,
            dtype=torch.uint8,
        )
        return pool

    def test_invalid_block_table_slots_are_masked(self):
        q = torch.ones(1, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(1, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=2, block_size=2)
        # The second logical block is padding.  Its positions must not be
        # turned into scores by clamping -1 to physical block zero.
        block_table = torch.tensor([[0, -1]], dtype=torch.int32)
        lengths = torch.tensor([[4]], dtype=torch.int32)

        out = _fp8_paged_indexer_score_sm120(
            q,
            weights,
            pool,
            block_table,
            lengths,
            block_size=2,
            max_ctx_len=4,
        )
        self.assertTrue(torch.isfinite(out[0, :2]).all())
        self.assertTrue(torch.isneginf(out[0, 2:]).all())

    def test_context_length_masks_tail_without_large_intermediate(self):
        q = torch.ones(2, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(2, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=4, block_size=2)
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        lengths = torch.tensor([[1], [3]], dtype=torch.int32)

        out = _fp8_paged_indexer_score_sm120(
            q,
            weights,
            pool,
            block_table,
            lengths,
            block_size=2,
            max_ctx_len=4,
        )
        self.assertTrue(torch.isfinite(out[0, :1]).all())
        self.assertTrue(torch.isneginf(out[0, 1:]).all())
        self.assertTrue(torch.isfinite(out[1, :3]).all())
        self.assertTrue(torch.isneginf(out[1, 3:]).all())

    def test_block_table_capacity_is_validated(self):
        q = torch.ones(1, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(1, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=1, block_size=2)
        with self.assertRaisesRegex(ValueError, "block-table capacity"):
            _fp8_paged_indexer_score_sm120(
                q,
                weights,
                pool,
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[2]], dtype=torch.int32),
                block_size=2,
                max_ctx_len=3,
            )


if __name__ == "__main__":
    unittest.main()
