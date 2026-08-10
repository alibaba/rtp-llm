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

from __future__ import annotations

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


def _ref_compute_swa_slot_mapping(
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len = sp + query_len
    num_tokens: int,
    pool_entries_per_block: int,
    tokens_per_block_for_block_table: int,
    ring_entries: int,
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
            block_in_seq = global_pos // tokens_per_block_for_block_table
            in_block = global_pos % ring_entries
            if block_in_seq < max_blocks:
                block_id = bt_cpu[b][block_in_seq]
            else:
                block_id = -1
            block_end = (block_in_seq + 1) * tokens_per_block_for_block_table
            effective_end = min(block_end, seq_lens_l[b])
            tail_write = global_pos + ring_entries >= effective_end
            if block_id <= 0 or not tail_write:
                slot = -1
            else:
                slot = block_id * pool_entries_per_block + in_block
            out[qs + i] = slot
    return out


def _ref_compute_swa_slot_mapping_from_positions(
    block_table: torch.Tensor,
    req_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_entries_per_block: int,
    tokens_per_block_for_block_table: int,
    ring_entries: int,
) -> torch.Tensor:
    """Loop reference for the independent-position DSpARK write mapper."""
    out = torch.full_like(positions, -1, dtype=torch.long)
    block_table_cpu = block_table.cpu().tolist()
    req_ids_cpu = req_id_per_token.cpu().tolist()
    positions_cpu = positions.cpu().tolist()
    seq_lens_cpu = seq_lens.cpu().tolist()
    max_blocks = int(block_table.shape[1])
    for token_idx, (req_id, position) in enumerate(zip(req_ids_cpu, positions_cpu)):
        if req_id < 0 or req_id >= len(seq_lens_cpu):
            continue
        seq_len = seq_lens_cpu[req_id]
        if position < 0 or position >= seq_len:
            continue
        block_in_seq = position // tokens_per_block_for_block_table
        if block_in_seq >= max_blocks:
            continue
        block_id = block_table_cpu[req_id][block_in_seq]
        block_end = (block_in_seq + 1) * tokens_per_block_for_block_table
        effective_end = min(block_end, seq_len)
        if block_id <= 0 or position + ring_entries < effective_end:
            continue
        out[token_idx] = block_id * pool_entries_per_block + position % ring_entries
    return out


def _ref_compute_swa_cp_sliced_slot_mapping(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: int,
    tokens_per_block_for_block_table: int,
    local_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Reference for CP-sliced SWA: block rows and ring entries are separate."""
    out = torch.full((num_tokens,), -1, dtype=torch.long, device=block_table.device)
    full_entries = int(local_entries_per_block) * int(cp_size)
    max_blocks = int(block_table.shape[1])
    qsl = query_start_loc.tolist()
    seq_lens_l = seq_lens.tolist()
    bt_cpu = block_table.cpu().tolist()
    for b in range(int(seq_lens.shape[0])):
        qs, qe = qsl[b], qsl[b + 1]
        query_len = qe - qs
        sp = seq_lens_l[b] - query_len
        for i in range(query_len):
            global_pos = sp + i
            block_in_seq = global_pos // int(tokens_per_block_for_block_table)
            ring_offset = global_pos % full_entries
            owner_rank = ring_offset // int(local_entries_per_block)
            local_offset = ring_offset - owner_rank * int(local_entries_per_block)
            block_id = bt_cpu[b][block_in_seq] if block_in_seq < max_blocks else -1
            block_end = (block_in_seq + 1) * int(tokens_per_block_for_block_table)
            effective_end = min(block_end, seq_lens_l[b])
            tail_write = global_pos + full_entries >= effective_end
            if block_id > 0 and owner_rank == int(cp_rank) and tail_write:
                out[qs + i] = block_id * int(local_entries_per_block) + local_offset
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
        pool_entries_per_block,
        tokens_per_block_for_block_table,
        ring_entries,
    ):
        bt, qsl, seq_lens, num_tokens = self._make_inputs(
            block_table, query_lens, sp_values
        )
        got = compute_swa_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            num_tokens=num_tokens,
            pool_entries_per_block=pool_entries_per_block,
            tokens_per_block_for_block_table=tokens_per_block_for_block_table,
            ring_entries=ring_entries,
        )
        ref = _ref_compute_swa_slot_mapping(
            bt,
            qsl,
            seq_lens,
            num_tokens,
            pool_entries_per_block=pool_entries_per_block,
            tokens_per_block_for_block_table=tokens_per_block_for_block_table,
            ring_entries=ring_entries,
        )
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
        """sp=0, all tokens fit in logical block 0."""
        self._check(
            block_table=[[5]],  # one valid block, id=5
            query_lens=[100],
            sp_values=[0],
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )

    def test_cold_prefill_spans_two_blocks(self):
        """sp=0, tokens cross the block boundary."""
        self._check(
            block_table=[[3, 7]],
            query_lens=[200],
            sp_values=[0],
            pool_entries_per_block=128,
            tokens_per_block_for_block_table=128,
            ring_entries=128,
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
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )

    def test_swa_eviction_some_tokens_dropped(self):
        """Long seq, early tokens land on -1 segments → slot=-1 (skip)."""
        self._check(
            block_table=[[-1, -1, -1, 21, 22]],
            query_lens=[1027],  # cold prefill of 1027 tokens
            sp_values=[0],
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )

    def test_sparse_valid_blocks_are_all_written(self):
        """Refactored cache can keep valid block ids at periodic positions.

        Prefill write must honor every positive entry, not just the final
        tail/reuse block. Tokens in logical blocks 0, 2, and 4 write to their
        physical slots; tokens in logical blocks 1 and 3 are skipped.
        """
        self._check(
            block_table=[[11, -1, 13, -1, 15]],
            query_lens=[5 * 256],
            sp_values=[0],
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )

    def test_large_physical_block_writes_only_ring_tails(self):
        """physical rows can be much larger than SWA ring entries.

        Only the final ring-sized tail before a physical boundary, plus the
        request tail in the next physical row, should write. Earlier tokens
        collide in the ring and must be skipped.
        """
        tpb = 16384
        for ring_entries in (128, 130, 132, 134):
            cases = (
                ("cold_cross_boundary", 0, tpb + 16),
                (
                    "continuation_cross_boundary",
                    tpb - ring_entries - 8,
                    ring_entries + 24,
                ),
                ("short_request_mid_block", 4096, 17),
            )
            for name, sp, query_len in cases:
                with self.subTest(name=name, ring_entries=ring_entries):
                    self._check(
                        block_table=[[7, 11]],
                        query_lens=[query_len],
                        sp_values=[sp],
                        pool_entries_per_block=ring_entries,
                        tokens_per_block_for_block_table=tpb,
                        ring_entries=ring_entries,
                    )

    def test_large_physical_block_multi_request_mixed_tails(self):
        """B>1 with different physical-row/tail shapes.

        Req0 crosses a physical boundary: the first few tokens in the query
        are still before the writable ring tail and must be skipped. Req1 is
        a short mid-block request tail and all tokens are writable.
        """
        tpb = 16384
        for ring_entries in (128, 130, 132, 134):
            with self.subTest(ring_entries=ring_entries):
                self._check(
                    block_table=[
                        [31, 32],
                        [41, 42],
                    ],
                    query_lens=[ring_entries + 20, 33],
                    sp_values=[tpb - ring_entries - 10, 4096],
                    pool_entries_per_block=ring_entries,
                    tokens_per_block_for_block_table=tpb,
                    ring_entries=ring_entries,
                )

    def test_continuation_at_segment_boundary(self):
        """sp lands exactly on a block boundary — first new token goes
        to in_block=0 of a fresh segment."""
        self._check(
            block_table=[[-1, -1, 8, 9]],
            query_lens=[256],
            sp_values=[512],  # global 512..767, eb=256 ⇒ seg 2
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
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
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )

    def test_empty_input(self):
        """num_tokens=0 must return an empty int64 tensor without launch."""
        bt = torch.full((1, 4), -1, dtype=torch.int32, device=self.device)
        qsl = torch.zeros(2, dtype=torch.int32, device=self.device)  # [0, 0]
        seq_lens = torch.zeros(1, dtype=torch.int32, device=self.device)
        out = compute_swa_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            num_tokens=0,
            pool_entries_per_block=256,
            tokens_per_block_for_block_table=256,
            ring_entries=256,
        )
        self.assertEqual(out.shape, (0,))
        self.assertEqual(out.dtype, torch.long)

    def test_dspark_128k_prefill_keeps_unique_physical_block_tails(self):
        """DSpARK's 134-entry ring must not race during long prefill writes."""
        ring_entries = 134

        for prompt_length in (128 * 1024, 128 * 1024 + 17):
            for tokens_per_block in (256, 16384):
                with self.subTest(
                    prompt_length=prompt_length,
                    tokens_per_block=tokens_per_block,
                ):
                    block_count = (
                        prompt_length + tokens_per_block - 1
                    ) // tokens_per_block
                    block_table = torch.arange(
                        1,
                        block_count + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ).view(1, block_count)
                    req_ids = torch.zeros(
                        prompt_length, dtype=torch.int32, device=self.device
                    )
                    positions = torch.arange(
                        prompt_length, dtype=torch.int32, device=self.device
                    )
                    seq_lens = torch.tensor(
                        [prompt_length], dtype=torch.int32, device=self.device
                    )

                    slots = compute_swa_slot_mapping_from_positions(
                        block_table=block_table,
                        req_id_per_token=req_ids,
                        positions=positions,
                        seq_lens=seq_lens,
                        num_tokens=prompt_length,
                        pool_entries_per_block=ring_entries,
                        tokens_per_block_for_block_table=tokens_per_block,
                        ring_entries=ring_entries,
                    )

                    valid_slots = slots[slots >= 0]
                    final_block_length = prompt_length % tokens_per_block
                    if final_block_length == 0:
                        final_block_length = tokens_per_block
                    expected_valid = (block_count - 1) * ring_entries + min(
                        final_block_length, ring_entries
                    )
                    self.assertEqual(int(valid_slots.numel()), expected_valid)
                    self.assertEqual(
                        int(torch.unique(valid_slots).numel()),
                        int(valid_slots.numel()),
                    )
                    first_tail = tokens_per_block - ring_entries
                    self.assertTrue(torch.all(slots[:first_tail] == -1))
                    expected_first_tail = (
                        ring_entries
                        + torch.arange(
                            first_tail,
                            tokens_per_block,
                            dtype=torch.long,
                            device=self.device,
                        )
                        % ring_entries
                    )
                    self.assertTrue(
                        torch.equal(
                            slots[first_tail:tokens_per_block], expected_first_tail
                        )
                    )

    def test_dspark_model_context_write_uses_real_position_mapper(self):
        """Connect the DSpARK context path to the real Triton mapper."""

        class StopAfterContextWrite(Exception):
            pass

        class FakeAttention:
            compress_ratio = 0
            rope_head_dim = 2
            head_dim = 4
            eps = 1e-6

            def __init__(self) -> None:
                self._kv_cache = None
                self._block_tables_by_type = {}
                self._cp_ctx = None
                self.freqs_cis = torch.zeros(
                    (512, 2), dtype=torch.float32, device=self_device
                )
                self.wkv = object()
                self.kv_norm = object()

            def _ensure_freqs_cis_bound(self) -> None:
                pass

            def _swa_entries_per_block(self) -> int:
                return 134

            def _swa_cp_byte_sliced(self) -> bool:
                return False

            def _pool_view_3d_fp8(self, _region: int) -> torch.Tensor:
                return torch.zeros((5, 134, 1), dtype=torch.uint8, device=self_device)

            def _lin(self, _weight: object, x: torch.Tensor) -> torch.Tensor:
                return x

        self_device = self.device
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        model._gen_num_per_cycle = 1
        model._v4_args = SimpleNamespace(window_size=128, dim=4, vocab_size=17)

        attention = FakeAttention()
        model.v4 = SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
        model.kv_cache = object()

        context_rows = 400
        context_req_ids = torch.zeros(
            context_rows, dtype=torch.int32, device=self.device
        )
        context_positions = torch.arange(
            context_rows, dtype=torch.int32, device=self.device
        )
        committed_ends = torch.tensor(
            [context_rows], dtype=torch.int32, device=self.device
        )

        with (
            patch.object(
                dspark_model_module,
                "fused_rmsnorm_rope",
                return_value=torch.zeros(
                    (context_rows, 4), dtype=torch.bfloat16, device=self.device
                ),
            ),
            patch.object(
                dspark_model_module,
                "decode_write_swa_fp8",
                side_effect=StopAfterContextWrite,
            ) as cache_writer,
        ):
            with self.assertRaises(StopAfterContextWrite):
                model._commit_layer_features(
                    layer_idx=0,
                    main_x=torch.zeros(
                        (context_rows, 4), dtype=torch.bfloat16, device=self.device
                    ),
                    context_req_ids=context_req_ids,
                    context_positions=context_positions,
                    committed_ends=committed_ends,
                    block_table=torch.tensor(
                        [[3, 4]], dtype=torch.int32, device=self.device
                    ),
                    tokens_per_block=256,
                    batch_size=1,
                )

        slots = cache_writer.call_args.kwargs["slot_mapping"]
        self.assertTrue(torch.all(slots[:122] == -1))
        self.assertTrue(torch.all(slots[122:256] >= 0))
        self.assertTrue(torch.all(slots[256:266] == -1))
        self.assertTrue(torch.all(slots[266:] >= 0))
        self.assertEqual(int((slots >= 0).sum().item()), 268)

    def test_dspark_position_mapping_masks_dense_decode_padding(self):
        """Graph capacity and rejected rows must not affect live decode writes."""
        graph_batch = 32
        verify_width = 6
        num_tokens = graph_batch * verify_width
        ring_entries = 134
        tokens_per_block = 16384

        block_table = torch.zeros(
            (graph_batch, 1), dtype=torch.int32, device=self.device
        )
        block_table[0, 0] = 11
        block_table[1, 0] = 12
        req_ids = torch.full((num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full_like(req_ids, -1)
        seq_lens = torch.zeros(graph_batch, dtype=torch.int32, device=self.device)

        req_ids[:3] = 0
        positions[:3] = torch.tensor(
            [4096, 4097, 4098], dtype=torch.int32, device=self.device
        )
        seq_lens[0] = 4099
        req_ids[verify_width] = 1
        positions[verify_width] = 1000
        seq_lens[1] = 1001

        slots = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=num_tokens,
            pool_entries_per_block=ring_entries,
            tokens_per_block_for_block_table=tokens_per_block,
            ring_entries=ring_entries,
        )

        expected = _ref_compute_swa_slot_mapping_from_positions(
            block_table,
            req_ids,
            positions,
            seq_lens,
            pool_entries_per_block=ring_entries,
            tokens_per_block_for_block_table=tokens_per_block,
            ring_entries=ring_entries,
        )
        self.assertTrue(torch.equal(slots, expected))
        valid_rows = torch.tensor([0, 1, 2, 6], device=self.device)
        self.assertTrue(torch.all(slots[valid_rows] >= 0))
        padding_mask = torch.ones(num_tokens, dtype=torch.bool, device=self.device)
        padding_mask[valid_rows] = False
        self.assertTrue(torch.all(slots[padding_mask] == -1))

    def test_dspark_position_mapping_matches_mixed_request_tails(self):
        """Physical-block tails and partial request tails are independent."""
        block_table = torch.tensor(
            [[3, 4], [7, 8]], dtype=torch.int32, device=self.device
        )
        req_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, -1],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [120, 121, 122, 255, 300, 365, 366, -1],
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.tensor([400, 367], dtype=torch.int32, device=self.device)

        got = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=int(positions.numel()),
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        expected = _ref_compute_swa_slot_mapping_from_positions(
            block_table,
            req_ids,
            positions,
            seq_lens,
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(got[:2].tolist(), [-1, -1])
        self.assertTrue(torch.all(got[2:7] >= 0))
        self.assertEqual(got[7].item(), -1)

    def test_dspark_position_mapping_normalizes_noncontiguous_inputs(self):
        """The public wrapper must not expose Triton's flat-stride assumption."""
        block_table = torch.tensor(
            [[3, 99, 4, 99], [7, 99, 8, 99]],
            dtype=torch.int32,
            device=self.device,
        )[:, ::2]
        req_ids = torch.tensor(
            [0, 99, 0, 99, 0, 99, 0, 99, 1, 99, 1, 99, 1, 99, -1, 99],
            dtype=torch.int32,
            device=self.device,
        )[::2]
        positions = torch.tensor(
            [120, 99, 121, 99, 122, 99, 255, 99, 300, 99, 365, 99, 366, 99, -1, 99],
            dtype=torch.int32,
            device=self.device,
        )[::2]
        seq_lens = torch.tensor(
            [400, -1, 367, -1], dtype=torch.int32, device=self.device
        )[::2]
        self.assertFalse(block_table.is_contiguous())
        self.assertFalse(req_ids.is_contiguous())
        self.assertFalse(positions.is_contiguous())
        self.assertFalse(seq_lens.is_contiguous())

        got = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=int(positions.numel()),
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        expected = _ref_compute_swa_slot_mapping_from_positions(
            block_table,
            req_ids,
            positions,
            seq_lens,
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        self.assertTrue(torch.equal(got, expected))

    def test_dspark_position_mapping_masks_invalid_rows_and_exact_tail(self):
        """Exercise invalid blocks/ids/positions and both sides of the tail."""
        block_table = torch.tensor(
            [[0, -1, 7, 8]], dtype=torch.int32, device=self.device
        )
        req_ids = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [255, 511, 633, 634, 767, 768, 899, 900, 0, 0],
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.tensor([900], dtype=torch.int32, device=self.device)

        got = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=int(positions.numel()),
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        expected = _ref_compute_swa_slot_mapping_from_positions(
            block_table,
            req_ids,
            positions,
            seq_lens,
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(
            (got >= 0).tolist(),
            [False, False, False, True, True, True, True, False, False, False],
        )

        truncated = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=5,
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        self.assertTrue(torch.equal(truncated, expected[:5]))

        empty = compute_swa_slot_mapping_from_positions(
            block_table=block_table,
            req_id_per_token=req_ids,
            positions=positions,
            seq_lens=seq_lens,
            num_tokens=0,
            pool_entries_per_block=134,
            tokens_per_block_for_block_table=256,
            ring_entries=134,
        )
        self.assertEqual(tuple(empty.shape), (0,))

    def test_dspark_position_mapping_rejects_invalid_geometry(self):
        block_table = torch.tensor([[1]], dtype=torch.int32, device=self.device)
        req_ids = torch.tensor([0], dtype=torch.int32, device=self.device)
        positions = torch.tensor([0], dtype=torch.int32, device=self.device)

        with self.assertRaises(AssertionError):
            compute_swa_slot_mapping_from_positions(
                block_table=block_table,
                req_id_per_token=req_ids,
                positions=positions,
                seq_lens=torch.tensor([1, 1], dtype=torch.int32, device=self.device),
                num_tokens=1,
                pool_entries_per_block=134,
                tokens_per_block_for_block_table=256,
                ring_entries=134,
            )
        with self.assertRaises(AssertionError):
            compute_swa_slot_mapping_from_positions(
                block_table=block_table,
                req_id_per_token=req_ids,
                positions=positions,
                seq_lens=torch.tensor([1], dtype=torch.int32, device=self.device),
                num_tokens=1,
                pool_entries_per_block=133,
                tokens_per_block_for_block_table=256,
                ring_entries=134,
            )

    def test_dspark_position_mapping_cuda_graph_replay(self):
        """Replay must consume new metadata and clear stale graph-bucket rows."""
        graph_batch = 32
        verify_width = 6
        num_tokens = graph_batch * verify_width
        block_table = torch.arange(
            1, graph_batch + 1, dtype=torch.int32, device=self.device
        ).view(graph_batch, 1)
        req_ids = torch.full((num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full_like(req_ids, -1)
        seq_lens = torch.zeros(graph_batch, dtype=torch.int32, device=self.device)

        def launch():
            return compute_swa_slot_mapping_from_positions(
                block_table=block_table,
                req_id_per_token=req_ids,
                positions=positions,
                seq_lens=seq_lens,
                num_tokens=num_tokens,
                pool_entries_per_block=134,
                tokens_per_block_for_block_table=16384,
                ring_entries=134,
            )

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_slots = launch()
        torch.cuda.synchronize()

        req_ids.fill_(-1)
        positions.fill_(-1)
        seq_lens.zero_()
        req_ids[:3] = 0
        positions[:3] = torch.tensor(
            [4096, 4097, 4098], dtype=torch.int32, device=self.device
        )
        seq_lens[0] = 4099
        req_ids[verify_width] = 1
        positions[verify_width] = 1000
        seq_lens[1] = 1001
        graph.replay()
        first = captured_slots.clone()
        first_valid = torch.tensor([0, 1, 2, 6], device=self.device)
        self.assertTrue(torch.all(first[first_valid] >= 0))
        first_padding = torch.ones(num_tokens, dtype=torch.bool, device=self.device)
        first_padding[first_valid] = False
        self.assertTrue(torch.all(first[first_padding] == -1))

        req_ids.fill_(-1)
        positions.fill_(-1)
        seq_lens.zero_()
        req_ids[0] = 0
        positions[0] = 500
        seq_lens[0] = 501
        graph.replay()
        second = captured_slots.clone()
        self.assertGreaterEqual(second[0].item(), 0)
        self.assertTrue(torch.all(second[1:] == -1))

    def test_block_id_zero_is_reserved(self):
        """BlockPool reserves physical block 0; only positive ids are writable."""
        bt, qsl, seq_lens, num_tokens = self._make_inputs(
            [[0, 5]], query_lens=[200], sp_values=[0]
        )
        got = compute_swa_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            num_tokens=num_tokens,
            pool_entries_per_block=128,
            tokens_per_block_for_block_table=128,
            ring_entries=128,
        )
        ref = _ref_compute_swa_slot_mapping(
            bt,
            qsl,
            seq_lens,
            num_tokens,
            pool_entries_per_block=128,
            tokens_per_block_for_block_table=128,
            ring_entries=128,
        )
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(got[0].item(), -1)
        self.assertEqual(got[127].item(), -1)
        self.assertGreater(got[128].item(), 0)

    def test_cp_sliced_uses_ring_for_owner_but_logical_block_for_table(self):
        """CP SWA sharding must not derive block-table rows from ring slices."""
        bt, qsl, seq_lens, num_tokens = self._make_inputs(
            [[11, 12, 13]], query_lens=[160], sp_values=[0]
        )
        tpb = 64
        local_entries = 32
        cp_size = 4
        for cp_rank in range(cp_size):
            with self.subTest(cp_rank=cp_rank):
                got = compute_swa_cp_sliced_slot_mapping(
                    block_table=bt,
                    query_start_loc=qsl,
                    seq_lens=seq_lens,
                    num_tokens=num_tokens,
                    tokens_per_block_for_block_table=tpb,
                    local_entries_per_block=local_entries,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                ref = _ref_compute_swa_cp_sliced_slot_mapping(
                    bt,
                    qsl,
                    seq_lens,
                    num_tokens,
                    tokens_per_block_for_block_table=tpb,
                    local_entries_per_block=local_entries,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                self.assertTrue(
                    torch.equal(got, ref),
                    msg=f"rank={cp_rank} got={got.tolist()} ref={ref.tolist()}",
                )
        rank2 = compute_swa_cp_sliced_slot_mapping(
            block_table=bt,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            num_tokens=num_tokens,
            tokens_per_block_for_block_table=tpb,
            local_entries_per_block=local_entries,
            cp_rank=2,
            cp_size=cp_size,
        )
        self.assertEqual(int(rank2[64].item()), 12 * local_entries)
        self.assertEqual(int(rank2[159].item()), -1)


if __name__ == "__main__":
    unittest.main()
