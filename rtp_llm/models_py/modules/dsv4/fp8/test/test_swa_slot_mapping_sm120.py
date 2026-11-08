"""SM120 cases split out of test_swa_slot_mapping.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

from rtp_llm.models_py.modules.dsv4.fp8.attention import _swa_paged_slots_fully_covered

"""UT: ``compute_swa_slot_mapping`` Triton kernel.

Validates the per-token paged slot formula used by FP8 SWA
prefill write (``_swa_prefill_ops_triton.compute_swa_slot_mapping``):

    global_pos    = sp[b] + i
    block_in_seq  = global_pos // tokens_per_block_for_block_table
    in_block      = global_pos % ring_entries
    block_id      = block_table[b, block_in_seq]   # sparse table; <=0 = skip
    slot          = -1                       if block_id <= 0
                    block_id * pool_entries_per_block + in_block otherwise

Compared against a Python reference (loop-based torch). Coverage:
  * cold prefill (sp=0) within first block
  * cold prefill spanning multiple blocks
  * continuation prefill (sp>0) — paged-tail block_table with leading -1s
  * SWA-eviction case (seqlen large, only last 2 segments allocated)
  * refactored cache layout with sparse positive block ids across the table
  * multi-request batch (B=2) with different sp / seqlen
  * empty input (num_tokens=0)

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_slot_mapping
"""
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

import rtp_llm.models_py.model_desc.deepseek_v4_dspark_model as dspark_model_module
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    compute_swa_cp_sliced_slot_mapping,
    compute_swa_slot_mapping,
    compute_swa_slot_mapping_from_positions,
)


class SwaSlotMappingTestSm120(unittest.TestCase):
    """Sibling of ``SwaSlotMappingTest`` carrying the SM120 cases."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_paged_prefill_falls_back_when_write_slots_contain_negative(self):
        """Any -1 write slot must abandon the direct-paged prefill route."""
        dense = torch.tensor([0, 5, 129, 511], dtype=torch.long, device=self.device)
        sparse = torch.tensor([0, -1, 129, -1], dtype=torch.long, device=self.device)
        self.assertTrue(_swa_paged_slots_fully_covered(dense))
        self.assertFalse(_swa_paged_slots_fully_covered(sparse))
        # Edge inputs: empty tables have nothing to miss, missing tables are
        # already outside the paged route.
        self.assertTrue(
            _swa_paged_slots_fully_covered(
                torch.empty(0, dtype=torch.long, device=self.device)
            )
        )
        self.assertTrue(_swa_paged_slots_fully_covered(None))


if __name__ == "__main__":
    import unittest

    unittest.main()
