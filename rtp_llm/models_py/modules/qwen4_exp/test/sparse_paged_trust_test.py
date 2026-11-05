"""Trusted-selection contract of the paged sparse GQA entry.

The SM120 sparse paths trust internal selections and rely on in-kernel
masking; the paged entry mirrors that contract through
``validate_selected=False``. These cases pin the production-mode behaviour
(``RTP_LLM_QSA_STRICT_VALIDATION`` explicitly cleared, the same mode the
sibling ``sparse_paged_fmha_test`` module would otherwise force process-wide
when discovery imports it first): the trust flag skips the per-selection
preflight, strict validation overrides the flag, and untrusted selections
keep the enqueue-only device verdict.
"""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    sparse_paged_gqa_attn,
)
from rtp_llm.models_py.modules.qwen4_exp.test.sparse_paged_fmha_test import (
    _reference,
)

_DEVICE = "cuda"
_PRODUCTION_ENV = {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}

# All-allocated table (no sentinel) for cases whose ids must be valid, and a
# table whose logical block 1 (ids 4..7) maps to the physical-0 allocator
# sentinel for the masking case. page_size=4, cache_blocks=8.
_CLEAN_TABLE = [[1, 4, 6, 5]]
_SENTINEL_TABLE = [[1, 0, 6, 5]]


def _fixture(table, seed=29):
    torch.manual_seed(seed)
    batch, q_heads, kv_heads, query_len, head_dim = 1, 4, 2, 3, 64
    page_size, cache_blocks = 4, 8
    q = torch.randn(
        batch,
        q_heads,
        query_len,
        head_dim,
        device=_DEVICE,
        dtype=torch.bfloat16,
    )
    cache = torch.randn(
        cache_blocks,
        2,
        kv_heads,
        page_size,
        head_dim,
        device=_DEVICE,
        dtype=torch.bfloat16,
    )
    block_table = torch.tensor(table, dtype=torch.int32, device=_DEVICE)
    kv_lens = torch.tensor([[9, 12, 15]], dtype=torch.int32, device=_DEVICE)
    return q, cache, block_table, kv_lens


class SparsePagedTrustedSelectionTest(unittest.TestCase):
    def test_trusted_selection_masks_invalid_ids_without_preflight(self):
        q, cache, block_table, kv_lens = _fixture(_SENTINEL_TABLE)
        # Every invalid class the preflight would reject, one row each:
        # -2 (padding must be -1), kv_len (outside the row's visible length),
        # 1000 (beyond the block-table width), and ids whose logical block 1
        # maps to the unallocated physical-0 sentinel.
        selected = torch.full((1, 3, 8), -1, dtype=torch.int32, device=_DEVICE)
        selected[0, 0, :5] = torch.tensor([0, 2, 7, -2, 9], device=_DEVICE)
        selected[0, 1, :6] = torch.tensor([1, 8, 11, 12, 16, -1], device=_DEVICE)
        selected[0, 2, :4] = torch.tensor([3, 6, 9, 1000], device=_DEVICE)

        with patch.dict(os.environ, _PRODUCTION_ENV), patch.object(
            torch, "_assert_async"
        ) as trap:
            got = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
                validate_selected=False,
            )
        trap.assert_not_called()

        # The kernel drops every invalid id; the reference drops ids outside
        # [0, kv_len) itself, so remove only the two sentinel-block ids
        # (7 in row 0, 6 in row 2) before building the expectation.
        masked = selected.clone()
        masked[0, 0, 2] = -1  # id 7 -> logical block 1 -> physical 0
        masked[0, 2, 1] = -1  # id 6 -> logical block 1 -> physical 0
        expected = _reference(q, cache, block_table, kv_lens, masked)
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    def test_trusted_selection_matches_validated_output_bit_for_bit(self):
        q, cache, block_table, kv_lens = _fixture(_CLEAN_TABLE, seed=31)
        selected = torch.full((1, 3, 8), -1, dtype=torch.int32, device=_DEVICE)
        selected[0, 0, :4] = torch.tensor([0, 2, 5, 8], device=_DEVICE)
        selected[0, 1, :5] = torch.tensor([1, 4, 7, 10, 11], device=_DEVICE)
        selected[0, 2, :3] = torch.tensor([3, 6, 12], device=_DEVICE)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            trusted = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
                validate_selected=False,
            )
            validated = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
            )
        self.assertTrue(torch.equal(trusted, validated))

    def test_trusted_selection_with_count_column_masks_beyond_count(self):
        q, cache, block_table, kv_lens = _fixture(_CLEAN_TABLE, seed=37)
        width = 2051  # the packed count layout's fixed slot width
        selected = torch.full((1, 3, width + 1), -1, dtype=torch.int32, device=_DEVICE)
        valid = torch.tensor([[0, 2, 5], [1, 8, 10], [3, 9, 13]], device=_DEVICE)
        for row in range(3):
            selected[0, row, :3] = valid[row]
        # Inconsistent count (5) with only three packed ids: the trusted path
        # clamps the count and masks the two extra -1 slots.
        selected[..., width] = 5

        with patch.dict(os.environ, _PRODUCTION_ENV), patch.object(
            torch, "_assert_async"
        ) as trap:
            got = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
                validate_selected=False,
                selected_has_count=True,
            )
        trap.assert_not_called()

        packed = selected[..., :width].contiguous()
        expected = _reference(q, cache, block_table, kv_lens, packed)
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    def test_strict_validation_overrides_the_trust_flag(self):
        q, cache, block_table, kv_lens = _fixture(_CLEAN_TABLE)
        selected = torch.full((1, 3, 8), -1, dtype=torch.int32, device=_DEVICE)
        selected[0, 0, 0] = 9  # == kv_lens[0, 0]: outside the row's kv_len

        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
            with self.assertRaisesRegex(ValueError, "outside its row's kv_len"):
                sparse_paged_gqa_attn(
                    q,
                    cache,
                    block_table,
                    kv_lens,
                    selected,
                    page_size=4,
                    kv_head_num=2,
                    validate_selected=False,
                )

    def test_untrusted_selection_keeps_the_enqueue_only_verdict(self):
        q, cache, block_table, kv_lens = _fixture(_CLEAN_TABLE)
        selected = torch.full((1, 3, 8), -1, dtype=torch.int32, device=_DEVICE)
        selected[0, 0, 0] = 9  # == kv_lens[0, 0]: outside the row's kv_len

        with patch.dict(os.environ, _PRODUCTION_ENV), patch.object(
            torch, "_assert_async"
        ) as trap:
            sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
            )
        trap.assert_called_once()
        self.assertIn(
            "selection, kv_lens or block table is invalid",
            trap.call_args.args[1],
        )


if __name__ == "__main__":
    unittest.main()
