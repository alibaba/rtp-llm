"""UT: ``compute_swa_slot_mapping`` Triton kernel.

Validates the per-token paged-tail slot formula used by FP8 SWA
prefill write (``_swa_prefill_ops_triton.compute_swa_slot_mapping``):

    global_pos    = sp[b] + i
    block_in_seq  = global_pos // block_size
    in_block      = global_pos % block_size
    block_id      = block_table[b, block_in_seq]   # paged tail; -1 = evicted
    slot          = -1                       if block_id <= 0
                    block_id * eb + in_block otherwise

Compared against a Python reference (loop-based torch). Coverage:
  * cold prefill (sp=0) within first block
  * cold prefill spanning multiple blocks
  * continuation prefill (sp>0) — paged-tail block_table with leading -1s
  * SWA-eviction case (seqlen large, only last 2 segments allocated)
  * multi-request batch (B=2) with different sp / seqlen
  * empty input (num_tokens=0)

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_slot_mapping
"""

from __future__ import annotations

import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.dsv4.prefill._swa_ops_triton import (
    compute_swa_slot_mapping,
)


def _ref_compute_swa_slot_mapping(
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len = sp + query_len
    block_size: int,
    num_tokens: int,
) -> torch.Tensor:
    """Pure-torch reference matching the Triton kernel formula."""
    out = torch.full((num_tokens,), -1, dtype=torch.long, device=block_table.device)
    num_reqs = int(seq_lens.shape[0])
    max_blocks = int(block_table.shape[1])
    qsl = query_start_loc.tolist()
    seq_lens_l = seq_lens.tolist()
    bt_cpu = block_table.cpu().tolist()
    for b in range(num_reqs):
        qs, qe = qsl[b], qsl[b + 1]
        query_len = qe - qs
        sp = seq_lens_l[b] - query_len
        for i in range(query_len):
            global_pos = sp + i
            block_in_seq = global_pos // block_size
            in_block = global_pos % block_size
            if block_in_seq < max_blocks:
                block_id = bt_cpu[b][block_in_seq]
            else:
                block_id = -1
            if block_id <= 0:
                slot = -1
            else:
                slot = block_id * block_size + in_block
            out[qs + i] = slot
    return out


class SwaSlotMappingTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_inputs(
        self,
        block_table: List[List[int]],
        query_lens: List[int],
        sp_values: List[int],
    ):
        num_reqs = len(query_lens)
        assert len(sp_values) == num_reqs
        bt = torch.tensor(block_table, dtype=torch.int32, device=self.device)
        cum = [0]
        for q in query_lens:
            cum.append(cum[-1] + q)
        query_start_loc = torch.tensor(cum, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor(
            [sp_values[b] + query_lens[b] for b in range(num_reqs)],
            dtype=torch.int32,
            device=self.device,
        )
        num_tokens = cum[-1]
        return bt, query_start_loc, seq_lens, num_tokens

    def _check(
        self,
        block_table,
        query_lens,
        sp_values,
        block_size,
    ):
        bt, qsl, seq_lens, num_tokens = self._make_inputs(
            block_table, query_lens, sp_values
        )
        got = compute_swa_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            block_size=block_size,
            num_tokens=num_tokens,
        )
        ref = _ref_compute_swa_slot_mapping(bt, qsl, seq_lens, block_size, num_tokens)
        self.assertEqual(got.shape, ref.shape)
        self.assertEqual(got.dtype, ref.dtype)
        diff = (got != ref).nonzero(as_tuple=False)
        self.assertEqual(
            diff.numel(),
            0,
            msg=(
                f"slot_mapping mismatch at indices {diff.flatten().tolist()[:20]}; "
                f"got={got[:20].tolist()} ref={ref[:20].tolist()}"
            ),
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_cold_prefill_single_block(self):
        """sp=0, all tokens fit in block 0."""
        self._check(
            block_table=[[5]],  # one valid block, id=5
            query_lens=[100],
            sp_values=[0],
            block_size=256,
        )

    def test_cold_prefill_spans_two_blocks(self):
        """sp=0, tokens cross the block boundary."""
        self._check(
            block_table=[[3, 7]],
            query_lens=[200],
            sp_values=[0],
            block_size=128,
        )

    def test_continuation_prefill_paged_tail(self):
        """sp>0 with paged-tail bt: leading -1, last 2 entries valid.

        Mirrors DSV4 SWA pool: total_slots=5, fixed_blocks=2 →
        bt = [-1, -1, -1, blk_a, blk_b]. New tokens at sp=900..999
        (query_len=100) span segments 3 and 4 — both valid blocks.
        """
        self._check(
            block_table=[[-1, -1, -1, 11, 12]],
            query_lens=[100],
            sp_values=[900],  # global pos 900..999, eb=256 ⇒ seg 3,4
            block_size=256,
        )

    def test_swa_eviction_some_tokens_dropped(self):
        """Long seq, early tokens land on -1 segments → slot=-1 (skip)."""
        self._check(
            block_table=[[-1, -1, -1, 21, 22]],
            query_lens=[1027],  # cold prefill of 1027 tokens
            sp_values=[0],
            block_size=256,
        )

    def test_continuation_at_segment_boundary(self):
        """sp lands exactly on a block boundary — first new token goes
        to in_block=0 of a fresh segment."""
        self._check(
            block_table=[[-1, -1, 8, 9]],
            query_lens=[256],
            sp_values=[512],  # global 512..767, eb=256 ⇒ seg 2
            block_size=256,
        )

    def test_multi_request_batch(self):
        """B=2 with different sp / different query_len.

        Currently the production builder hardcodes B==1 (see
        ``_build_swa_prefill_meta``); this test exercises the kernel's
        per-request math directly so future B>1 work has a baseline.
        """
        self._check(
            block_table=[
                [-1, -1, 11, 12],
                [-1, -1, 21, 22],
            ],
            query_lens=[200, 100],
            sp_values=[600, 700],
            block_size=256,
        )

    def test_empty_input(self):
        """num_tokens=0 must return an empty int64 tensor without launch."""
        bt = torch.zeros((1, 4), dtype=torch.int32, device=self.device)
        qsl = torch.zeros(2, dtype=torch.int32, device=self.device)  # [0, 0]
        seq_lens = torch.zeros(1, dtype=torch.int32, device=self.device)
        out = compute_swa_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            block_size=256,
            num_tokens=0,
        )
        self.assertEqual(out.shape, (0,))
        self.assertEqual(out.dtype, torch.long)

    def test_block_id_zero_is_invalid(self):
        """``block_id <= 0`` should produce slot=-1 (zero is sentinel for
        unallocated, matching the BF16-path's `(block_id > 0)` guard)."""
        # block_table entry 0 is an invalid block id (treated like -1).
        self._check(
            block_table=[[0, 5]],
            query_lens=[200],
            sp_values=[0],
            block_size=128,
        )


if __name__ == "__main__":
    unittest.main()
