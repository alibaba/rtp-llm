"""UT: ``CompressorFP8.prepare_metadata`` under batched (positions, b_idx).

The Phase-3a varlen migration replaces the upstream
``_build_prefill_positions(sp_int, 1, seqlen, ...)`` (which writes
``b_idx = zeros(S)`` for the single-request path) with the upper-layer-
derived ``(position_ids, req_id_per_token)`` for B>1 batched prefill.
``prepare_metadata`` was already batched-capable — its slot-mapping
helpers index ``bt[b_idx, ...]`` per token. This UT proves that:

  1. **B==1 bit-equality**: feeding ``(arange(sp,sp+S), zeros)`` produces
     the same slot mappings as the legacy single-request inputs.
  2. **B==2 per-request routing**: every token gets its OWN request's
     block_table row picked, not the single shared row.
  3. **KV slot boundary mask**: ``(pos+1) % ratio == 0`` rule is
     enforced per token (not per request) — same math as B==1, just
     across packed positions.

Stub pattern mirrors ``test_indexer_fp8_prefill_meta`` so we don't pay
the FP8 weight + DeepGEMM cost just to validate slot arithmetic.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_compressor_prepare_metadata_varlen \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    CompressorMeta,
    build_prepare_metadata_args,
)


def _build_prefill_positions(
    sp: int, bsz: int, seqlen: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    assert bsz == 1
    positions = torch.arange(sp, sp + seqlen, device=device, dtype=torch.long)
    b_idx = torch.zeros(seqlen, device=device, dtype=torch.long)
    return positions, b_idx


class _StubCompressor:
    """Stand-in exposing only the attrs ``prepare_metadata`` and the two
    slot-mapping helpers read."""

    def __init__(
        self,
        compress_ratio: int,
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
    ) -> None:
        self.compress_ratio = compress_ratio
        self._state_block_table = state_block_table
        self._state_eb = state_eb
        self._state_tokens_per_block = state_eb
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._kv_cache_sharded = False
        self._cp_ctx = None
        self._kv_pool_view = None  # disable the pool-row overflow guard

    # Bind the unbound methods so ``_StubCompressor`` quacks correctly.
    _compute_state_slot_mapping = CompressorFP8._compute_state_slot_mapping
    _compute_kv_slot_mapping = CompressorFP8._compute_kv_slot_mapping
    prepare_metadata = CompressorFP8.prepare_metadata


def _make_stub(
    device: torch.device,
    *,
    ratio: int = 4,
    state_eb: int = 256,
    kv_eb: int = 64,  # tokens_per_block = kv_eb * ratio = 256
    n_reqs: int = 2,
    blocks_per_req: int = 4,
) -> _StubCompressor:
    """``state`` and ``kv`` block tables: row b → blocks
    ``[b*blocks_per_req+1 .. (b+1)*blocks_per_req]``. We start at 1 so
    each request has distinct non-overlapping physical blocks."""
    state_bt = torch.arange(
        1, n_reqs * blocks_per_req + 1, dtype=torch.int32, device=device
    ).view(n_reqs, blocks_per_req)
    # Use distinct block ids for kv vs state so a per-request routing bug
    # surfaces as a wrong slot id rather than coincidentally matching.
    kv_bt = state_bt + 1000
    return _StubCompressor(ratio, state_bt, state_eb, kv_bt, kv_eb)


def _seq_meta_for_single_req(
    sp: int, S: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (seq_start_per_req, cu_seq_per_req) for a single request."""
    seq_start = torch.tensor([sp], dtype=torch.int64, device=device)
    cu_seq = torch.tensor([0, S], dtype=torch.int64, device=device)
    return seq_start, cu_seq


def _seq_meta_for_batched(
    prefix_lengths: list[int],
    input_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (seq_start_per_req, cu_seq_per_req) for batched requests."""
    seq_start = torch.tensor(prefix_lengths, dtype=torch.int64, device=device)
    cu_seq = torch.zeros(len(input_lengths) + 1, dtype=torch.int64, device=device)
    for i, L in enumerate(input_lengths):
        cu_seq[i + 1] = cu_seq[i] + L
    return seq_start, cu_seq


class CompressorPrepareMetadataVarlenTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")
        self.ratio = 4

    def _prepare(
        self,
        stub: _StubCompressor,
        positions: torch.Tensor,
        b_idx: torch.Tensor,
        prefix_lengths: Optional[list[int]] = None,
        input_lengths: Optional[list[int]] = None,
    ):
        """Call prepare_metadata with auto-derived seq metadata."""
        if prefix_lengths is None:
            n_reqs = int(b_idx.max().item()) + 1
            prefix_lengths = [0] * n_reqs
            input_lengths = [int((b_idx == b).sum().item()) for b in range(n_reqs)]
        seq_start, cu_seq = _seq_meta_for_batched(
            prefix_lengths, input_lengths, self.device  # type: ignore[arg-type]
        )
        return stub.prepare_metadata(
            positions, b_idx, seq_start_per_req=seq_start, cu_seq_per_req=cu_seq
        )

    # ------------------------------------------------------------------
    # B == 1: new (position_ids, req_id_per_token) must equal legacy path.
    # ------------------------------------------------------------------
    def test_b1_batched_path_matches_legacy(self) -> None:
        stub = _make_stub(self.device, n_reqs=1)
        sp, S = 12, 20
        legacy_pos, legacy_b = _build_prefill_positions(sp, 1, S, self.device)
        # Mimic the upper-layer derivation for B==1: position_ids = arange(sp, sp+S),
        # req_id_per_token = zeros (cast to long for prepare_metadata).
        new_pos = torch.arange(
            sp, sp + S, dtype=torch.int64, device=self.device
        ).contiguous()
        new_b = torch.zeros(S, dtype=torch.int64, device=self.device).contiguous()
        legacy_meta = self._prepare(stub, legacy_pos, legacy_b, [sp], [S])
        new_meta = self._prepare(stub, new_pos, new_b, [sp], [S])
        self.assertTrue(torch.equal(legacy_meta.positions, new_meta.positions))
        self.assertTrue(torch.equal(legacy_meta.b_idx, new_meta.b_idx))
        self.assertTrue(torch.equal(legacy_meta.state_slots, new_meta.state_slots))
        self.assertTrue(torch.equal(legacy_meta.kv_slots, new_meta.kv_slots))
        self.assertTrue(torch.equal(legacy_meta.token_to_req, new_meta.token_to_req))

    def test_legacy_positions_are_flat_token_major(self) -> None:
        sp, S = 12, 20
        positions, b_idx = _build_prefill_positions(sp, 1, S, self.device)
        self.assertEqual(tuple(positions.shape), (S,))
        self.assertEqual(tuple(b_idx.shape), (S,))
        self.assertTrue(
            torch.equal(
                positions,
                torch.arange(sp, sp + S, dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(torch.equal(b_idx, torch.zeros_like(b_idx)))

    def test_varlen_prepare_args_flattens_legacy_batch_dim(self) -> None:
        S = 5
        position_ids = torch.arange(
            12, 12 + S, dtype=torch.int32, device=self.device
        ).view(1, S)
        req_id = torch.zeros((1, S), dtype=torch.int32, device=self.device)
        args = build_prepare_metadata_args(
            device=self.device,
            position_ids=position_ids,
            req_id_per_token=req_id,
            seq_start_per_req=torch.tensor([12], dtype=torch.int32, device=self.device),
            cu_seqlens=torch.tensor([0, S], dtype=torch.int32, device=self.device),
        )
        self.assertEqual(tuple(args["positions"].shape), (S,))
        self.assertEqual(tuple(args["b_idx"].shape), (S,))
        self.assertTrue(
            torch.equal(
                args["positions"],
                torch.arange(12, 12 + S, dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                args["b_idx"],
                torch.zeros(S, dtype=torch.int64, device=self.device),
            )
        )

    # ------------------------------------------------------------------
    # B == 2: each token must read its own request's block_table row.
    # ------------------------------------------------------------------
    def _build_batched_positions(
        self,
        prefix_lengths: list[int],
        input_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror prefill/forward.py's derivation — flat ``positions`` +
        per-token ``req_id``."""
        positions = torch.cat(
            [
                torch.arange(p, p + L, dtype=torch.int64, device=self.device)
                for L, p in zip(input_lengths, prefix_lengths)
            ],
            dim=0,
        )
        req_id = torch.cat(
            [
                torch.full((L,), b, dtype=torch.int64, device=self.device)
                for b, L in enumerate(input_lengths)
            ],
            dim=0,
        )
        return positions, req_id

    def test_b2_state_slots_pick_correct_request_row(self) -> None:
        """For each token: state_slot = bt[b_idx[t], pos//eb]
        * eb + pos % eb. With distinct-row block tables and req-local pos
        confined to that row's block range, the slot ids must split
        cleanly per request."""
        stub = _make_stub(self.device, n_reqs=2, blocks_per_req=2)
        # Req0: sp=0,S=8 → positions [0..7] → block_in_seq=0, in_block=pos
        # Req1: sp=300,S=10 → positions [300..309] → block_in_seq=1
        #   (since eb=256, 300//256=1; within the table's two blocks)
        positions, req_id = self._build_batched_positions([0, 300], [8, 10])
        meta = self._prepare(stub, positions, req_id, [0, 300], [8, 10])
        # Req0 first token: bt[0, 0]=1, in_block=0 → slot=1*256+0=256
        self.assertEqual(int(meta.state_slots[0].item()), 256)
        # Req0 last token (pos=7): slot=1*256+7=263
        self.assertEqual(int(meta.state_slots[7].item()), 263)
        # Req1 first token (pos=300): bt[1, 1]=4, in_block=300%256=44 → 4*256+44=1068
        self.assertEqual(int(meta.state_slots[8].item()), 1068)
        # Req1 last token (pos=309): in_block=53 → 4*256+53=1077
        self.assertEqual(int(meta.state_slots[17].item()), 1077)

    def test_b2_kv_slots_boundary_only_per_request(self) -> None:
        """KV slot is non-(-1) only at boundary tokens (pos+1)%ratio==0.
        The rule is per-token, but the routing through bt[b_idx, ...]
        must still be per-request. ratio=4 → boundaries at pos%4==3."""
        stub = _make_stub(self.device, n_reqs=2, blocks_per_req=2)
        # Req0: sp=0,S=8 → boundaries at pos=3,7
        # Req1: sp=4,S=8 → boundaries at pos=7,11
        positions, req_id = self._build_batched_positions([0, 4], [8, 8])
        meta = self._prepare(stub, positions, req_id, [0, 4], [8, 8])
        kv = meta.kv_slots
        # tokens_per_block = kv_eb * ratio = 64 * 4 = 256
        # For req0 boundary at pos=3:
        #   bt[0, 0]=1001 (kv_bt = state_bt + 1000), in_block = (3%256)//4 = 0
        #   slot = 1001*64 + 0 = 64064
        # Indices in flat positions: req0 covers t=0..7, req1 covers t=8..15
        self.assertEqual(int(kv[3].item()), 1001 * 64 + 0)  # req0 boundary 1
        self.assertEqual(int(kv[7].item()), 1001 * 64 + 1)  # req0 boundary 2
        # Req1 boundary at pos=7: bt[1,0]=1003, in_block=(7%256)//4=1 → 1003*64+1
        self.assertEqual(int(kv[8 + 3].item()), 1003 * 64 + 1)
        # Req1 boundary at pos=11: in_block=(11%256)//4=2 → 1003*64+2
        self.assertEqual(int(kv[8 + 7].item()), 1003 * 64 + 2)
        # Non-boundary tokens (e.g. t=0, pos=0) are -1
        self.assertEqual(int(kv[0].item()), -1)
        self.assertEqual(int(kv[8].item()), -1)
        # Cross-check: non-boundary req1 (t=8+4, pos=8): pos+1=9, 9%4!=0 → -1
        self.assertEqual(int(kv[8 + 4].item()), -1)

    def test_b2_token_to_req_passes_through_b_idx(self) -> None:
        """token_to_req is the int32 alias of b_idx — used by
        run_fused_compress_kv_write to route per-token writes."""
        stub = _make_stub(self.device, n_reqs=2)
        positions, req_id = self._build_batched_positions([0, 16], [4, 6])
        meta = self._prepare(stub, positions, req_id, [0, 16], [4, 6])
        self.assertTrue(torch.equal(meta.token_to_req, req_id.to(torch.int32)))

    def test_b2_unallocated_block_yields_minus_one_state_slot(self) -> None:
        """When bt[b, k] == -1 (sentinel), state_slot must be -1 — the
        per-request mask must respect each request's own block table."""
        # Patch req1's first block to -1 (unallocated); req0 keeps block_id=1.
        stub = _make_stub(self.device, n_reqs=2, blocks_per_req=2)
        stub._state_block_table = stub._state_block_table.clone()
        stub._state_block_table[1, 0] = -1  # req1's first block unallocated
        # Req0: sp=0,S=4 → all 4 tokens land in req0's first block (id=1)
        # Req1: sp=0,S=4 → all 4 tokens want req1's first block (now id=-1 → -1)
        positions, req_id = self._build_batched_positions([0, 0], [4, 4])
        meta = self._prepare(stub, positions, req_id, [0, 0], [4, 4])
        # Req0 tokens (t=0..3): valid slots
        for t in range(4):
            self.assertGreater(int(meta.state_slots[t].item()), 0)
        # Req1 tokens (t=4..7): masked to -1
        for t in range(4, 8):
            self.assertEqual(int(meta.state_slots[t].item()), -1)

    def test_zero_block_id_is_invalid_sentinel(self) -> None:
        """block_id == 0 is the unallocated sentinel (unified with reader)."""
        stub = _make_stub(self.device, n_reqs=1, blocks_per_req=2)
        stub._state_block_table = stub._state_block_table.clone()
        stub._state_block_table[0, 1] = 0
        positions = torch.tensor([256, 257], dtype=torch.long, device=self.device)
        req_id = torch.zeros_like(positions)
        meta = self._prepare(stub, positions, req_id, [256], [2])
        self.assertEqual(meta.state_slots.tolist(), [-1, -1])

    def test_state_sparse_valid_blocks_are_all_written(self) -> None:
        """HCA_STATE / CSA_STATE / INDEXER_STATE share this state mapping.

        The refactored cache may expose periodic valid logical blocks rather
        than a tail-only table. Every positive state block id must produce a
        writable state slot; gaps must stay ``-1``.

        Note: ``positions`` deliberately stops at 1279 (= 5*256 - 1). A
        position 1280 would map to ``(1280//256) % 5 == 0`` and collide
        with the pos=0 slot in block 11 — that wrap-aliasing is the ring
        contract of the writer, not a bug, and is masked by the
        ``seq_end_per_req`` write guard at the caller (see
        ``_fused_compressor_meta_triton`` HAS_SEQ_END path). This UT
        targets writer slot arithmetic only, so wrap cases are out of scope.
        """
        stub = _make_stub(self.device, n_reqs=1, blocks_per_req=5)
        stub._state_block_table = torch.tensor(
            [[11, -1, 13, -1, 15]], dtype=torch.int64, device=self.device
        )
        positions = torch.tensor(
            [
                0,
                255,
                256,
                511,
                512,
                767,
                768,
                1023,
                1024,
                1279,
            ],
            dtype=torch.int64,
            device=self.device,
        )
        req_id = torch.zeros_like(positions)
        meta = self._prepare(stub, positions, req_id, [0], [int(positions.numel())])

        expected = torch.tensor(
            [
                11 * 256 + 0,
                11 * 256 + 255,
                -1,
                -1,
                13 * 256 + 0,
                13 * 256 + 255,
                -1,
                -1,
                15 * 256 + 0,
                15 * 256 + 255,
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.assertTrue(torch.equal(meta.state_slots, expected))

        head_size = 8
        kv = torch.arange(
            positions.numel() * head_size,
            dtype=torch.float32,
            device=self.device,
        ).view(positions.numel(), head_size)
        score = torch.zeros_like(kv)
        ape = torch.zeros(
            (self.ratio, head_size), dtype=torch.float32, device=self.device
        )
        sentinel = -123.0
        state_cache = torch.full(
            (16, 256, 2 * head_size),
            sentinel,
            dtype=torch.float32,
            device=self.device,
        )
        run_save_partial_states(
            kv,
            score,
            ape,
            positions,
            state_cache,
            meta.state_slots,
            compress_ratio=self.ratio,
        )
        torch.cuda.synchronize()

        # pos=0→kv[0], pos=255→kv[1] in block 11
        self.assertTrue(torch.equal(state_cache[11, 0, :head_size], kv[0]))
        self.assertTrue(torch.equal(state_cache[11, 255, :head_size], kv[1]))
        # pos=512→kv[4], pos=767→kv[5] in block 13
        self.assertTrue(torch.equal(state_cache[13, 0, :head_size], kv[4]))
        self.assertTrue(torch.equal(state_cache[13, 255, :head_size], kv[5]))
        # pos=1024→kv[8], pos=1279→kv[9] in block 15
        self.assertTrue(torch.equal(state_cache[15, 0, :head_size], kv[8]))
        self.assertTrue(torch.equal(state_cache[15, 255, :head_size], kv[9]))
        # Blocks 12, 14 had negative block_ids → untouched
        self.assertTrue(torch.all(state_cache[12] == sentinel))
        self.assertTrue(torch.all(state_cache[14] == sentinel))


if __name__ == "__main__":
    unittest.main()
