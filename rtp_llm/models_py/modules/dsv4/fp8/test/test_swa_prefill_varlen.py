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

from __future__ import annotations

import unittest
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_ops
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    SwaPrefillMeta,
    _build_suffix_pool_slot_mapping,
    _get_window_topk_idxs,
    _get_window_topk_idxs_varlen,
)


class _StubKvCache:
    def __init__(self, tokens_per_block: int) -> None:
        self.group_region_names = [SWA_KV]
        self.seq_size_per_block = int(tokens_per_block)
        self.kernel_seq_size_per_block = int(tokens_per_block)


class _StubAttention:
    """Minimal stand-in for ``Attention`` exposing only what the SWA
    prefill meta leaf builders actually read:

      * ``self.window_size`` / ``self.compress_ratio``
      * ``self._kv_cache`` (just truthy / None)
      * ``self._block_tables_by_type[SWA_KV]``
      * ``self._pool_entries_per_block(SWA_KV)`` → ``self._eb``
    """

    def __init__(
        self,
        *,
        window_size: int,
        compress_ratio: int,
        block_table_swa: Optional[torch.Tensor],
        eb: int,
        kv_cache_present: bool = True,
        kv_cache: Optional[object] = None,
    ) -> None:
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self._eb = eb
        self._kv_cache = (
            (kv_cache if kv_cache is not None else _StubKvCache(eb))
            if kv_cache_present
            else None
        )
        self._block_tables_by_type = (
            {SWA_KV: block_table_swa} if block_table_swa is not None else None
        )

    def _pool_entries_per_block(self, attn_type: int) -> int:
        if attn_type != SWA_KV:
            return 0
        return self._eb

    # Bind the unbound leaf builder so the stub quacks correctly.
    _build_swa_prefill_meta_varlen = Attention._build_swa_prefill_meta_varlen


def _make_block_table(n_reqs: int, blocks_per_req: int, device) -> torch.Tensor:
    """Per-request block ids: row b → ``[b*blocks_per_req+1 ..
    (b+1)*blocks_per_req]``. block_id 0 is the unallocated sentinel so
    we start at 1."""
    return torch.arange(
        1, n_reqs * blocks_per_req + 1, dtype=torch.int64, device=device
    ).view(n_reqs, blocks_per_req)


class _FakeLargeBlockKvCache:
    """Expose scalar C++ KVCache fields used by require_pool_tokens_per_block."""

    group_region_names = [SWA_KV]
    seq_size_per_block = 16384
    kernel_seq_size_per_block = 128


def _flat_positions(
    prefix_lengths: list[int], input_lengths: list[int], device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror ``prefill/forward.py`` — flat ``positions`` + ``cu_seqlens``
    + ``req_id_per_token`` from per-request prefix/input lengths."""
    positions = torch.cat(
        [
            torch.arange(p, p + L, dtype=torch.int64, device=device)
            for L, p in zip(input_lengths, prefix_lengths)
        ],
        dim=0,
    )
    req_id = torch.cat(
        [
            torch.full((L,), b, dtype=torch.int32, device=device)
            for b, L in enumerate(input_lengths)
        ],
        dim=0,
    )
    cu_seqlens = torch.zeros(len(input_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(input_lengths, dtype=torch.int32, device=device), dim=0
    )
    return positions, req_id, cu_seqlens


# -------------------------------------------------------------------------
# 1. _get_window_topk_idxs_varlen — pure-tensor helper
# -------------------------------------------------------------------------
class GetWindowTopkIdxsVarlenTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def test_b1_cold_matches_legacy(self) -> None:
        """B==1, sp==0: varlen output (squeezed) == legacy ``[1, T, win]``."""
        win, S = 8, 12
        positions, req_id, cu_seqlens = _flat_positions([0], [S], self.device)
        prefix_lengths = torch.tensor([0], dtype=torch.int32, device=self.device)
        varlen = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )  # [T, win]
        legacy = _get_window_topk_idxs(win, 1, S, 0, self.device).squeeze(0)  # [T, win]
        self.assertEqual(varlen.shape, legacy.shape)
        self.assertTrue(torch.equal(varlen, legacy))

    def test_b1_wrapped_token_dim_matches_flat(self) -> None:
        win, S = 8, 12
        positions, req_id, cu_seqlens = _flat_positions([0], [S], self.device)
        prefix_lengths = torch.tensor([0], dtype=torch.int32, device=self.device)
        flat = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        wrapped = _get_window_topk_idxs_varlen(
            win,
            cu_seqlens.view(1, -1),
            positions.view(1, -1),
            prefix_lengths.view(1, -1),
            req_id.view(1, -1),
        )
        self.assertEqual(tuple(wrapped.shape), (S, win))
        self.assertTrue(torch.equal(wrapped, flat))

    def test_b1_continuation_indices(self) -> None:
        """B==1, sp>0: index window in flat-KV layout = local pos within
        the request (cu_seqlens[0]==0 ⇒ flat idx == local pos)."""
        win, sp, S = 4, 100, 6
        positions, req_id, cu_seqlens = _flat_positions([sp], [S], self.device)
        prefix_lengths = torch.tensor([sp], dtype=torch.int32, device=self.device)
        out = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        # Token 0 (local pos 0): window = [0]; rest -1.
        self.assertEqual(int(out[0, 0].item()), 0)
        for k in range(1, win):
            self.assertEqual(int(out[0, k].item()), -1)
        # Token at local pos 5: window = [2,3,4,5] (full win=4, no padding).
        self.assertEqual(out[5, 0].item(), 2)
        self.assertEqual(out[5, 1].item(), 3)
        self.assertEqual(out[5, 2].item(), 4)
        self.assertEqual(out[5, 3].item(), 5)

    def test_b2_mixed_sp_indices_per_request(self) -> None:
        """B==2 mixed (cold + continuation). Each request's window indices
        must reference the request's own slice of the flat-KV layout."""
        win = 4
        positions, req_id, cu_seqlens = _flat_positions([0, 50], [3, 4], self.device)
        prefix_lengths = torch.tensor([0, 50], dtype=torch.int32, device=self.device)
        out = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        # Req 0 (cold, S=3): tokens t=0,1,2 with local pos 0,1,2.
        # Flat indices = local pos (cu_seqlens[0]==0).
        # t=0: [0, -1, -1, -1]
        self.assertEqual(int(out[0, 0].item()), 0)
        self.assertEqual(int(out[0, 1].item()), -1)
        # t=2: [0, 1, 2, -1]
        self.assertEqual(int(out[2, 0].item()), 0)
        self.assertEqual(int(out[2, 1].item()), 1)
        self.assertEqual(int(out[2, 2].item()), 2)
        self.assertEqual(int(out[2, 3].item()), -1)
        # Req 1 (cont, sp=50, S=4): tokens t=3,4,5,6 at flat positions 3..6.
        # Local pos 0,1,2,3 → flat idx = cu_seqlens[1] + local = 3 + local.
        # t=3 local pos 0: [3, -1, -1, -1]
        self.assertEqual(int(out[3, 0].item()), 3)
        self.assertEqual(int(out[3, 1].item()), -1)
        # t=6 local pos 3: full win=4 → [3, 4, 5, 6]
        self.assertEqual(int(out[6, 0].item()), 3)
        self.assertEqual(int(out[6, 1].item()), 4)
        self.assertEqual(int(out[6, 2].item()), 5)
        self.assertEqual(int(out[6, 3].item()), 6)

    def test_win_larger_than_seqlen_pads_tail(self) -> None:
        """``window_size`` > ``S_b`` for every request: each row has only
        ``S_b`` valid slots at the start, the rest are -1 (right-pad)."""
        win, S = 8, 3
        positions, req_id, cu_seqlens = _flat_positions([0], [S], self.device)
        prefix_lengths = torch.tensor([0], dtype=torch.int32, device=self.device)
        out = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        # t=0 local pos 0: [0, -1, -1, -1, -1, -1, -1, -1]
        self.assertEqual(int(out[0, 0].item()), 0)
        for k in range(1, win):
            self.assertEqual(int(out[0, k].item()), -1)
        # t=2 local pos 2: [0, 1, 2, -1, -1, -1, -1, -1]
        self.assertEqual(int(out[2, 0].item()), 0)
        self.assertEqual(int(out[2, 1].item()), 1)
        self.assertEqual(int(out[2, 2].item()), 2)
        for k in range(3, win):
            self.assertEqual(int(out[2, k].item()), -1)

    def test_single_token_request(self) -> None:
        """B==1, S==1: only the first slot is valid; tail is -1."""
        win = 4
        positions, req_id, cu_seqlens = _flat_positions([42], [1], self.device)
        prefix_lengths = torch.tensor([42], dtype=torch.int32, device=self.device)
        out = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        self.assertEqual(out.shape, (1, win))
        # local_pos = 0 → window = [0]; flat idx = cu_seqlens[0] + 0 = 0.
        self.assertEqual(int(out[0, 0].item()), 0)
        for k in range(1, win):
            self.assertEqual(int(out[0, k].item()), -1)

    def test_b2_all_continuation_indices(self) -> None:
        """Both requests have prefix > 0 — varlen helper still ignores
        prefix size, only uses local pos within each request."""
        win = 4
        # req0 sp=8 S=3, req1 sp=20 S=2. Local pos req0: [0,1,2]; req1: [0,1].
        positions, req_id, cu_seqlens = _flat_positions([8, 20], [3, 2], self.device)
        prefix_lengths = torch.tensor([8, 20], dtype=torch.int32, device=self.device)
        out = _get_window_topk_idxs_varlen(
            win, cu_seqlens, positions, prefix_lengths, req_id
        )
        # Req 0 starts at flat idx cu_seqlens[0]=0; req 1 at cu_seqlens[1]=3.
        # t=0 (req0 lp=0): [0, -1, -1, -1]
        self.assertEqual(int(out[0, 0].item()), 0)
        self.assertEqual(int(out[0, 1].item()), -1)
        # t=2 (req0 lp=2): [0, 1, 2, -1]
        self.assertEqual(int(out[2, 0].item()), 0)
        self.assertEqual(int(out[2, 1].item()), 1)
        self.assertEqual(int(out[2, 2].item()), 2)
        self.assertEqual(int(out[2, 3].item()), -1)
        # t=3 (req1 lp=0): [3, -1, -1, -1] (cu_seqlens[1] = 3)
        self.assertEqual(int(out[3, 0].item()), 3)
        self.assertEqual(int(out[3, 1].item()), -1)
        # t=4 (req1 lp=1): [3, 4, -1, -1]
        self.assertEqual(int(out[4, 0].item()), 3)
        self.assertEqual(int(out[4, 1].item()), 4)
        self.assertEqual(int(out[4, 2].item()), -1)
        self.assertEqual(int(out[4, 3].item()), -1)


# -------------------------------------------------------------------------
# 2. _build_swa_prefill_meta varlen branch — stub-driven
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for triton kernels")
class BuildSwaPrefillMetaVarlenTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def _build_stub(
        self,
        *,
        win: int = 8,
        compress_ratio: int = 0,
        n_reqs: int = 1,
        blocks_per_req: int = 4,
        eb: int = 256,
        kv_cache_present: bool = True,
        kv_cache: Optional[object] = None,
    ) -> _StubAttention:
        bt = _make_block_table(n_reqs, blocks_per_req, self.device)
        return _StubAttention(
            window_size=win,
            compress_ratio=compress_ratio,
            block_table_swa=bt,
            eb=eb,
            kv_cache_present=kv_cache_present,
            kv_cache=kv_cache,
        )

    def _build_meta_legacy(
        self, stub: _StubAttention, sp: int, S: int
    ) -> SwaPrefillMeta:
        """Local B==1 scalar reference for the varlen path's B==1 collapse."""
        win = stub.window_size
        is_swa_only = stub.compress_ratio == 0
        topk_length_kv_full: Optional[torch.Tensor] = None
        if is_swa_only or stub._kv_cache is None:
            positions = torch.arange(S, device=self.device, dtype=torch.int32)
            topk_length_kv_full = torch.clamp(positions + 1, max=win)

        bt = (
            stub._block_tables_by_type.get(SWA_KV)
            if stub._block_tables_by_type is not None
            else None
        )
        eb = stub._pool_entries_per_block(SWA_KV)
        if stub._kv_cache is None or bt is None or bt.numel() == 0 or eb <= 0:
            return SwaPrefillMeta(
                slot_mapping=None,
                query_start_loc=None,
                combined_seq_lens=None,
                topk_length_kv_full=topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                cache_slot_mapping=None,
            )

        seq_total = sp + S
        query_start_loc = torch.tensor([0, S], device=self.device, dtype=torch.int32)
        combined_seq_lens = torch.tensor(
            [seq_total], device=self.device, dtype=torch.int32
        )
        bt_swa = bt[:1].to(device=self.device, dtype=torch.int32).contiguous()
        slot_mapping = _swa_ops.compute_swa_slot_mapping(
            block_table=bt_swa,
            query_start_loc=query_start_loc,
            seq_lens=combined_seq_lens,
            num_tokens=S,
            pool_entries_per_block=eb,
            tokens_per_block_for_block_table=stub._kv_cache.seq_size_per_block,
            ring_entries=eb,
        )

        if not is_swa_only:
            return SwaPrefillMeta(
                slot_mapping=slot_mapping,
                query_start_loc=query_start_loc,
                combined_seq_lens=combined_seq_lens,
                topk_length_kv_full=None,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                cache_slot_mapping=None,
            )

        combined_gather_lens = _swa_ops.compute_prefill_gather_lens(
            seq_lens=combined_seq_lens,
            query_start_loc=query_start_loc,
            num_prefills=1,
            num_decodes=0,
            window_size=win,
        )
        combined_gather_len_max = S + min(sp, win - 1)
        M = max(combined_gather_len_max, 1)

        if sp > 0:
            prefix_len = min(sp, win - 1)
            cache_seq_lens = torch.tensor([sp], device=self.device, dtype=torch.int32)
            cache_gather_lens = torch.tensor(
                [prefix_len], device=self.device, dtype=torch.int32
            )
            cache_slot_mapping = _build_suffix_pool_slot_mapping(
                block_table=bt_swa,
                seq_lens=cache_seq_lens,
                gather_lens=cache_gather_lens,
                entries_per_block=eb,
                tokens_per_block_for_block_table=stub._kv_cache.seq_size_per_block,
                ring_entries=eb,
            )
            topk_indices_empty = torch.empty(
                (S, 0), dtype=torch.int32, device=self.device
            )
            combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices(
                topk_indices=topk_indices_empty,
                query_start_loc=query_start_loc,
                seq_lens=combined_seq_lens,
                gather_lens=combined_gather_lens,
                window_size=win,
                compress_ratio=1,
                topk=0,
                M=M,
                N=0,
            )
            prefix_len_max = prefix_len
        else:
            cache_seq_lens = None
            cache_gather_lens = None
            cache_slot_mapping = None
            combined_indices = None
            combined_lens = None
            prefix_len_max = 0

        return SwaPrefillMeta(
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            combined_seq_lens=combined_seq_lens,
            topk_length_kv_full=topk_length_kv_full,
            combined_gather_lens=combined_gather_lens,
            combined_gather_len_max=combined_gather_len_max,
            M=M,
            cache_seq_lens=cache_seq_lens,
            cache_gather_lens=cache_gather_lens,
            prefix_len_max=prefix_len_max,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            slot_in_flat=None,
            cache_slot_mapping=cache_slot_mapping,
        )

    def _build_meta_varlen(
        self,
        stub: _StubAttention,
        prefix_lengths: list[int],
        input_lengths: list[int],
    ) -> SwaPrefillMeta:
        positions, req_id, cu_seqlens = _flat_positions(
            prefix_lengths, input_lengths, self.device
        )
        pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=self.device)
        il = torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
        any_cont = bool((pl > 0).any().item())
        return stub._build_swa_prefill_meta_varlen(
            seqlen=int(positions.numel()),
            device=self.device,
            any_cont=any_cont,
            batch_size=len(input_lengths),
            cu_seqlens=cu_seqlens,
            input_lengths=il,
            prefix_lengths=pl,
            position_ids=positions,
            req_id_per_token=req_id,
        )

    # ----- B==1 bit-equality vs legacy scalar path -------------------------
    def test_b1_cold_swa_only_matches_legacy(self) -> None:
        stub = self._build_stub(compress_ratio=0)
        legacy = self._build_meta_legacy(stub, sp=0, S=12)
        varlen = self._build_meta_varlen(stub, [0], [12])
        self.assertTrue(torch.equal(varlen.slot_mapping, legacy.slot_mapping))
        self.assertTrue(torch.equal(varlen.query_start_loc, legacy.query_start_loc))
        self.assertTrue(torch.equal(varlen.combined_seq_lens, legacy.combined_seq_lens))
        self.assertTrue(
            torch.equal(varlen.topk_length_kv_full, legacy.topk_length_kv_full)
        )
        self.assertTrue(
            torch.equal(varlen.combined_gather_lens, legacy.combined_gather_lens)
        )
        self.assertEqual(varlen.combined_gather_len_max, legacy.combined_gather_len_max)
        self.assertEqual(varlen.M, legacy.M)
        # Cold has no cache_* / combined_*.
        self.assertIsNone(varlen.cache_seq_lens)
        self.assertIsNone(legacy.cache_seq_lens)
        self.assertIsNone(varlen.combined_indices)
        self.assertIsNone(legacy.combined_indices)
        self.assertEqual(varlen.prefix_len_max, 0)
        self.assertEqual(legacy.prefix_len_max, 0)

    def test_b1_continuation_swa_only_matches_legacy(self) -> None:
        win, sp, S = 8, 32, 6
        stub = self._build_stub(win=win, compress_ratio=0)
        legacy = self._build_meta_legacy(stub, sp=sp, S=S)
        varlen = self._build_meta_varlen(stub, [sp], [S])
        # Group-1 + Group-2 + cache_* + combined_* all populated.
        self.assertTrue(torch.equal(varlen.slot_mapping, legacy.slot_mapping))
        self.assertTrue(torch.equal(varlen.combined_seq_lens, legacy.combined_seq_lens))
        self.assertTrue(
            torch.equal(varlen.topk_length_kv_full, legacy.topk_length_kv_full)
        )
        self.assertTrue(torch.equal(varlen.cache_seq_lens, legacy.cache_seq_lens))
        self.assertTrue(torch.equal(varlen.cache_gather_lens, legacy.cache_gather_lens))
        self.assertEqual(varlen.M, legacy.M)
        self.assertTrue(torch.equal(varlen.combined_indices, legacy.combined_indices))
        self.assertTrue(torch.equal(varlen.combined_lens, legacy.combined_lens))
        # ``prefix_len_max`` intentionally diverges: legacy carries the actual
        # ``min(sp, win-1)`` (used for the B==1 single-slice copy in
        # ``_attn_fp8_swa_via_concat``); varlen sets it to a sentinel ``1``
        # because the per-token P_b math runs through ``cache_gather_lens``
        # instead. Both are >0 so the dequant guard fires identically.
        self.assertGreater(legacy.prefix_len_max, 0)
        self.assertEqual(varlen.prefix_len_max, 1)

    # ----- B==2 — per-request semantics -----------------------------------
    def test_b2_cold_plus_cold_no_continuation_fields(self) -> None:
        """All-cold batch: any_cont == False → no cache_* / combined_*.
        topk_length_kv_full[t] = min(local_pos[t]+1, win) per request."""
        win = 8
        stub = self._build_stub(win=win, compress_ratio=0, n_reqs=2)
        meta = self._build_meta_varlen(stub, [0, 0], [3, 5])
        self.assertIsNone(meta.cache_seq_lens)
        self.assertIsNone(meta.cache_gather_lens)
        self.assertIsNone(meta.combined_indices)
        self.assertIsNone(meta.combined_lens)
        self.assertEqual(meta.prefix_len_max, 0)
        # combined_seq_lens = [3, 5] (sp + S per req).
        self.assertTrue(
            torch.equal(
                meta.combined_seq_lens,
                torch.tensor([3, 5], dtype=torch.int32, device=self.device),
            )
        )
        # topk_length_kv_full per token:
        # req0 (S=3): [1, 2, 3]; req1 (S=5): [1, 2, 3, 4, 5]; all clamp ≤ win=8.
        expected = torch.tensor(
            [1, 2, 3, 1, 2, 3, 4, 5], dtype=torch.int32, device=self.device
        )
        self.assertTrue(torch.equal(meta.topk_length_kv_full, expected))
        # combined_gather_len_max = max(input + min(prefix, win-1)) = 5 + 0 = 5
        self.assertEqual(meta.combined_gather_len_max, 5)
        self.assertEqual(meta.M, 5)

    def test_b2_cold_plus_continuation_full_meta(self) -> None:
        """Mixed batch: any_cont == True → full meta. Per-request
        cache_seq_lens / cache_gather_lens reflect each request's prefix."""
        win = 8
        stub = self._build_stub(win=win, compress_ratio=0, n_reqs=2)
        meta = self._build_meta_varlen(stub, [0, 32], [4, 3])
        self.assertIsNotNone(meta.cache_seq_lens)
        self.assertIsNotNone(meta.cache_gather_lens)
        self.assertIsNotNone(meta.combined_indices)
        # Per-req cache_seq_lens = prefix_lengths.
        self.assertTrue(
            torch.equal(
                meta.cache_seq_lens,
                torch.tensor([0, 32], dtype=torch.int32, device=self.device),
            )
        )
        # Per-req cache_gather_lens = clamp(prefix, win-1) = [0, min(32, 7)] = [0, 7].
        self.assertTrue(
            torch.equal(
                meta.cache_gather_lens,
                torch.tensor([0, 7], dtype=torch.int32, device=self.device),
            )
        )
        # prefix_len_max sentinel under varlen — see the bit-equality test above.
        self.assertEqual(meta.prefix_len_max, 1)
        # combined_gather_len_max = max(input + min(prefix, win-1)).
        # req0: 4 + 0 = 4; req1: 3 + 7 = 10. max = 10.
        self.assertEqual(meta.combined_gather_len_max, 10)
        self.assertEqual(meta.M, 10)
        # combined_lens per token: cold req topk + swa = 0 + min(local+1, win),
        # cont req similarly. For req0 (S=4): [1, 2, 3, 4]; for req1 (sp=32, S=3):
        # local pos 0,1,2 → swa_len = min(33, 8), min(34, 8), min(35, 8) = 8.
        expected_cmb = torch.tensor(
            [1, 2, 3, 4, 8, 8, 8], dtype=torch.int32, device=self.device
        )
        self.assertTrue(torch.equal(meta.combined_lens, expected_cmb))

    def test_b2_all_continuation_full_meta(self) -> None:
        """Both requests have prefix > 0. ``cache_seq_lens`` should be
        all-non-zero; combined_lens reflects each request's own swa cap."""
        win = 8
        stub = self._build_stub(win=win, compress_ratio=0, n_reqs=2)
        # Req 0: sp=4 (< win-1=7), prefix tail clamped to 4.
        # Req 1: sp=20 (> win-1=7), prefix tail clamped to 7.
        meta = self._build_meta_varlen(stub, [4, 20], [3, 5])
        self.assertIsNotNone(meta.cache_seq_lens)
        self.assertIsNotNone(meta.cache_gather_lens)
        # cache_seq_lens = raw prefix_lengths.
        self.assertTrue(
            torch.equal(
                meta.cache_seq_lens,
                torch.tensor([4, 20], dtype=torch.int32, device=self.device),
            )
        )
        # cache_gather_lens = clamp(prefix, win-1).
        self.assertTrue(
            torch.equal(
                meta.cache_gather_lens,
                torch.tensor([4, 7], dtype=torch.int32, device=self.device),
            )
        )
        # combined_gather_len_max = max(3 + 4, 5 + 7) = 12.
        self.assertEqual(meta.combined_gather_len_max, 12)
        self.assertEqual(meta.M, 12)
        # combined_lens (swa-only ⇒ topk_len=0):
        # req0 sp=4 S=3: positions abs=[4,5,6] → swa_len = min(p+1, win) = [5, 6, 7]
        # req1 sp=20 S=5: positions abs=[20..24] → swa_len capped at win=8 = [8]*5
        expected_cmb = torch.tensor(
            [5, 6, 7, 8, 8, 8, 8, 8], dtype=torch.int32, device=self.device
        )
        self.assertTrue(torch.equal(meta.combined_lens, expected_cmb))

    def test_large_physical_swa_write_slot_mapping_tail_mask(self) -> None:
        """SWA write meta must use physical row tokens and small ring entries.

        With physical=16384 and ring in {128,130,132,134}, prefill writes only
        physical block/request tails. If the builder used ``eb`` as the
        block-table stride, this would either address out-of-range rows or
        write every ring collision.
        """
        tpb = 16384
        for ring in (128, 130, 132, 134):
            with self.subTest(ring=ring, shape="cold_cross_boundary"):
                stub = self._build_stub(
                    win=128,
                    compress_ratio=0,
                    n_reqs=1,
                    blocks_per_req=2,
                    eb=ring,
                    kv_cache=_FakeLargeBlockKvCache(),
                )
                meta = self._build_meta_varlen(stub, [0], [tpb + 16])
                self.assertIsNotNone(meta.slot_mapping)
                sm = meta.slot_mapping
                self.assertEqual(int(sm.shape[0]), tpb + 16)
                self.assertEqual(int(sm[0].item()), -1)
                self.assertEqual(int(sm[tpb - ring - 1].item()), -1)
                self.assertEqual(
                    int(sm[tpb - ring].item()), 1 * ring + ((tpb - ring) % ring)
                )
                self.assertEqual(int(sm[tpb - 1].item()), 1 * ring + ((tpb - 1) % ring))
                self.assertEqual(int(sm[tpb].item()), 2 * ring + (tpb % ring))
                self.assertEqual(int(sm[-1].item()), 2 * ring + ((tpb + 15) % ring))

            with self.subTest(ring=ring, shape="continuation_cross_boundary"):
                stub = self._build_stub(
                    win=128,
                    compress_ratio=0,
                    n_reqs=1,
                    blocks_per_req=2,
                    eb=ring,
                    kv_cache=_FakeLargeBlockKvCache(),
                )
                sp = tpb - ring - 8
                meta = self._build_meta_varlen(stub, [sp], [ring + 24])
                self.assertIsNotNone(meta.slot_mapping)
                sm = meta.slot_mapping
                self.assertEqual(int(sm[0].item()), -1)
                self.assertEqual(int(sm[7].item()), -1)
                self.assertEqual(int(sm[8].item()), 1 * ring + ((tpb - ring) % ring))
                self.assertEqual(
                    int(sm[ring + 7].item()), 1 * ring + ((tpb - 1) % ring)
                )
                self.assertEqual(int(sm[ring + 8].item()), 2 * ring + (tpb % ring))
                self.assertEqual(int(sm[-1].item()), 2 * ring + ((tpb + 15) % ring))

            with self.subTest(ring=ring, shape="short_request_mid_block"):
                stub = self._build_stub(
                    win=128,
                    compress_ratio=0,
                    n_reqs=1,
                    blocks_per_req=2,
                    eb=ring,
                    kv_cache=_FakeLargeBlockKvCache(),
                )
                sp = 4096
                meta = self._build_meta_varlen(stub, [sp], [17])
                self.assertIsNotNone(meta.slot_mapping)
                sm = meta.slot_mapping
                self.assertEqual(int((sm >= 0).sum().item()), 17)
                self.assertEqual(int(sm[0].item()), 1 * ring + (sp % ring))
                self.assertEqual(int(sm[-1].item()), 1 * ring + ((sp + 16) % ring))

    # ----- Warmup (no kv_cache) -------------------------------------------
    def test_warmup_no_kv_cache_topk_length_only(self) -> None:
        """When ``self._kv_cache is None`` (warmup forward), Group-1 fields
        stay None; ``topk_length_kv_full`` is cache-independent so it's
        still set so the warmup attn path can consume it."""
        win = 8
        stub = self._build_stub(
            win=win, compress_ratio=0, n_reqs=2, kv_cache_present=False
        )
        meta = self._build_meta_varlen(stub, [0, 16], [3, 4])
        # Group-1 unset.
        self.assertIsNone(meta.slot_mapping)
        self.assertIsNone(meta.query_start_loc)
        self.assertIsNone(meta.combined_seq_lens)
        # Group-2 attention meta (cache-dependent ones) also unset.
        self.assertIsNone(meta.combined_gather_lens)
        self.assertIsNone(meta.combined_indices)
        self.assertIsNone(meta.combined_lens)
        self.assertIsNone(meta.cache_seq_lens)
        self.assertIsNone(meta.cache_gather_lens)
        self.assertEqual(meta.M, 0)
        # ``topk_length_kv_full`` is cache-independent — built from
        # position_ids - prefix_lengths.gather(req_id) regardless.
        self.assertIsNotNone(meta.topk_length_kv_full)
        # req0 sp=0 S=3: [1, 2, 3]; req1 sp=16 S=4: [1, 2, 3, 4] (all < win).
        expected = torch.tensor(
            [1, 2, 3, 1, 2, 3, 4], dtype=torch.int32, device=self.device
        )
        self.assertTrue(torch.equal(meta.topk_length_kv_full, expected))


# -------------------------------------------------------------------------
# 3. _attn_fp8_swa_via_concat new-K scatter math
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for index_copy_")
class ViaConcatScatterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_b1_continuation_scatter_matches_slice(self) -> None:
        """B==1 scatter must place new K at workspace[0, P:P+S, :] —
        bit-equal to the legacy single-slice copy."""
        win, sp, S, D = 8, 32, 6, 4
        P = min(sp, win - 1)  # 7
        M = S + P  # 13
        prefix_lengths = torch.tensor([sp], dtype=torch.int32, device=self.device)
        positions, req_id, _ = _flat_positions([sp], [S], self.device)
        kv_full = torch.randn(S, D, dtype=torch.bfloat16, device=self.device)

        # Vectorized scatter (mirror _attn_fp8_swa_via_concat step 2).
        ws_vec = torch.zeros((1, M, D), dtype=torch.bfloat16, device=self.device)
        prefix_l = prefix_lengths.to(torch.long)
        req_l = req_id.to(torch.long)
        pos_l = positions.to(torch.long)
        P_b = torch.clamp_max(prefix_l, win - 1)
        P_per_token = P_b.gather(0, req_l)
        local_pos = pos_l - prefix_l.gather(0, req_l)
        slot = req_l * M + P_per_token + local_pos
        ws_vec.view(1 * M, D).index_copy_(0, slot, kv_full)

        # Legacy single-slice (bit-equality oracle).
        ws_legacy = torch.zeros((1, M, D), dtype=torch.bfloat16, device=self.device)
        ws_legacy[:, P : P + S, :].copy_(kv_full.unsqueeze(0))

        self.assertTrue(torch.equal(ws_vec, ws_legacy))

    def test_b2_scatter_per_request_offsets(self) -> None:
        """B==2: each request's new K must land at workspace[b, P_b:P_b+S_b, :]
        and not bleed into the other request's slice."""
        win, D = 8, 4
        prefix_lengths_list = [0, 16]
        input_lengths_list = [3, 4]
        prefix_lengths = torch.tensor(
            prefix_lengths_list, dtype=torch.int32, device=self.device
        )
        positions, req_id, _ = _flat_positions(
            prefix_lengths_list, input_lengths_list, self.device
        )
        T = sum(input_lengths_list)
        P_b_list = [min(p, win - 1) for p in prefix_lengths_list]
        S_max_plus_P = max(
            P_b_list[b] + input_lengths_list[b] for b in range(len(input_lengths_list))
        )
        M = max(S_max_plus_P, 1)
        kv_full = torch.randn(T, D, dtype=torch.bfloat16, device=self.device)

        ws = torch.zeros((2, M, D), dtype=torch.bfloat16, device=self.device)
        prefix_l = prefix_lengths.to(torch.long)
        req_l = req_id.to(torch.long)
        pos_l = positions.to(torch.long)
        P_b_t = torch.clamp_max(prefix_l, win - 1)
        P_per_token = P_b_t.gather(0, req_l)
        local_pos = pos_l - prefix_l.gather(0, req_l)
        slot = req_l * M + P_per_token + local_pos
        ws.view(2 * M, D).index_copy_(0, slot, kv_full)

        # Req 0 (P=0, S=3): new K at ws[0, 0:3, :].
        self.assertTrue(torch.equal(ws[0, 0:3, :], kv_full[0:3]))
        # Req 0 tail (rows 3..M-1) untouched.
        self.assertTrue(torch.all(ws[0, 3:, :] == 0))
        # Req 1 (P=7, S=4): new K at ws[1, 7:11, :].
        self.assertTrue(torch.equal(ws[1, 7:11, :], kv_full[3:7]))
        # Req 1 prefix slot (rows 0..6) untouched (zero-init).
        self.assertTrue(torch.all(ws[1, 0:7, :] == 0))
        # Req 1 tail (rows 11..M-1) untouched.
        self.assertTrue(torch.all(ws[1, 11:, :] == 0))


if __name__ == "__main__":
    unittest.main()
