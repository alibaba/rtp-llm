"""SM120 cases split out of test_swa_prefill_varlen.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

"""UT: SWA-only prefill varlen (B>1) prepare + scatter math.

Phase-2 of the dsv4 prefill varlen migration. Validates:

  1. ``_get_window_topk_idxs_varlen`` — flat-KV window indices for cold
     prefill across mixed (B, prefix) shapes.
  2. ``_build_swa_prefill_meta_varlen`` leaf builder — per-request meta
     fields with bit-equality to a local B==1 reference.
  3. ``_attn_fp8_swa_via_concat`` step-2 scatter — per-request P_b offset
     placement of the new K BF16 overlay into the [B, M, D] workspace.
  4. Coverage corners: B==2 all-continuation; warmup (kv_cache=None).

Stub pattern mirrors ``test_compressor_prepare_metadata_varlen`` so we
don't pay the FP8 weight + DeepGEMM cost just to validate the SWA
prepare arithmetic. flash_mla_sparse_fwd is NOT exercised here — that
gets coverage from the smoke targets.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_swa_prefill_varlen \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""
import unittest
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_ops
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    SwaPrefillMeta,
    _build_suffix_pool_slot_mapping,
    _get_window_topk_idxs,
    _get_window_topk_idxs_varlen,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

_SWA_FP8_ENTRY_BYTES = 584


class SuffixPoolSlotMappingTest(unittest.TestCase):
    def test_sparse_absolute_block_columns_are_not_compacted(self) -> None:
        slots = _build_suffix_pool_slot_mapping(
            block_table=torch.tensor([[-1, -1, 10, 11]], dtype=torch.int32),
            seq_lens=torch.tensor([256], dtype=torch.int32),
            gather_lens=torch.tensor([128], dtype=torch.int32),
            entries_per_block=64,
            tokens_per_block_for_block_table=64,
            ring_entries=64,
        )

        self.assertEqual(tuple(slots.shape), (1, 128))
        self.assertTrue(torch.equal(slots[0], torch.arange(640, 768)))


if __name__ == "__main__":
    import unittest

    unittest.main()
