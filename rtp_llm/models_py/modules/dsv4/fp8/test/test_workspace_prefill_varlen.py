"""UT: CSA / HCA workspace prefill metadata varlen (B>=1) builder.

Phase-3b of the dsv4 prefill varlen migration. Validates that
``Attention._build_workspace_meta`` correctly produces:

  1. **Legacy B==1 / DSV4_VARLEN_PREFILL=0** — bit-equal to the
     pre-Phase-3 scalar implementation (sp_int + seqlen scalars).
  2. **Varlen B>=2** — N_max-padded layout with per-request
     ``swa_seq_lens`` / ``cmp_seq_lens`` / ``swa_gather_lens`` /
     ``new_k_slot_in_flat`` derived from ``prefix_lengths`` +
     ``input_lengths`` + ``cu_seqlens`` + ``position_ids`` +
     ``req_id_per_token``.
  3. **Warmup short-circuits** — None when pool unbound / SWA-only
     layer / unallocated block tables.
  4. **HCA dense_cmp_topk** — ``[T_total, N_max]`` arange grid;
     CSA dense_cmp_topk == None.
  5. **Workspace overlay scatter math** — feeding ``new_k_slot_in_flat``
     to ``index_copy_`` over a synthetic ``[B, M, D]`` workspace places
     each request's new K at exactly ``[b, N_max + P_b : N_max + P_b + S_b, :]``
     and leaves all other slots zero.

Stub-driven: no FP8 weights, no kernel launches. Mirrors the pattern
from ``test_swa_prefill_varlen.py`` and ``test_compressor_prepare_metadata_varlen``.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_workspace_prefill_varlen \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import os
import unittest
from typing import Optional
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import WorkspaceMeta


# -------------------------------------------------------------------------
# Stub helpers
# -------------------------------------------------------------------------
class _StubAttention:
    """Minimal stand-in for ``Attention`` that exposes only what
    ``_build_workspace_meta`` actually reads:

      * ``self.window_size`` / ``self.compress_ratio``
      * ``self._kv_cache`` (truthy / None)
      * ``self._block_tables_by_type[SWA_KV/CSA_KV/HCA_KV]``
      * ``self._pool_entries_per_block(attn_type)`` → ``self._eb_by_type``
    """

    def __init__(
        self,
        *,
        window_size: int,
        compress_ratio: int,
        block_tables: Optional[dict] = None,
        eb_by_type: Optional[dict] = None,
        kv_cache_present: bool = True,
    ) -> None:
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self._kv_cache = object() if kv_cache_present else None
        self._block_tables_by_type = block_tables
        self._eb_by_type = eb_by_type or {}

    def _pool_entries_per_block(self, attn_type: int) -> int:
        return int(self._eb_by_type.get(attn_type, 0))

    # Bind the unbound methods so the stub quacks correctly.
    _build_workspace_meta = Attention._build_workspace_meta


def _make_block_table(n_reqs: int, blocks_per_req: int, device) -> torch.Tensor:
    """Per-request block ids: row b → ``[b*blocks_per_req+1 ..
    (b+1)*blocks_per_req]``. block_id 0 is the unallocated sentinel so
    we start at 1."""
    return torch.arange(
        1, n_reqs * blocks_per_req + 1, dtype=torch.int64, device=device
    ).view(n_reqs, blocks_per_req)


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


def _make_stub(
    *,
    win: int,
    compress_ratio: int,
    n_reqs: int = 1,
    blocks_per_req: int = 4,
    swa_eb: int = 256,
    cmp_eb: int = 64,
    kv_cache_present: bool = True,
    bt_present: bool = True,
    cmp_bt_present: bool = True,
    swa_bt_present: bool = True,
    swa_eb_override: Optional[int] = None,
    cmp_eb_override: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> _StubAttention:
    """Build a stub attention with paired SWA + CSA/HCA pools.

    ``swa_bt_present`` / ``cmp_bt_present`` / ``bt_present`` (= overall)
    let warmup-style tests detach individual pools.
    ``swa_eb_override`` / ``cmp_eb_override`` simulate eb<=0 (pool not yet
    sized) without removing the block tables themselves.
    """
    dev = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    bt_swa = (
        _make_block_table(n_reqs, blocks_per_req, dev)
        if (bt_present and swa_bt_present)
        else None
    )
    cmp_at = (
        CSA_KV if compress_ratio == 4 else HCA_KV if compress_ratio == 128 else None
    )
    bt_cmp = (
        _make_block_table(n_reqs, blocks_per_req, dev)
        if (bt_present and cmp_bt_present and cmp_at is not None)
        else None
    )

    block_tables: Optional[dict] = None
    if bt_present:
        block_tables = {}
        if bt_swa is not None:
            block_tables[SWA_KV] = bt_swa
        if cmp_at is not None and bt_cmp is not None:
            block_tables[cmp_at] = bt_cmp

    eb_by_type: dict = {}
    eb_by_type[SWA_KV] = swa_eb_override if swa_eb_override is not None else swa_eb
    if cmp_at is not None:
        eb_by_type[cmp_at] = cmp_eb_override if cmp_eb_override is not None else cmp_eb

    return _StubAttention(
        window_size=win,
        compress_ratio=compress_ratio,
        block_tables=block_tables,
        eb_by_type=eb_by_type,
        kv_cache_present=kv_cache_present,
    )


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_meta_legacy(
    stub: _StubAttention, sp: int, S: int, with_dense_cmp_topk: bool
) -> Optional[WorkspaceMeta]:
    """Legacy B==1 / DSV4_VARLEN_PREFILL=0 — explicit ``use_varlen=False``
    so the dispatch falls into the scalar branch regardless of env. The
    contract guard in ``_build_shared_prefill_meta`` is the only place the
    env is read in production; sub-builders take ``use_varlen`` as a kwarg."""
    return stub._build_workspace_meta(
        seqlen=S,
        sp_int=sp,
        device=_device(),
        with_dense_cmp_topk=with_dense_cmp_topk,
        use_varlen=False,
    )


def _build_meta_varlen(
    stub: _StubAttention,
    prefix_lengths: list[int],
    input_lengths: list[int],
    with_dense_cmp_topk: bool,
) -> Optional[WorkspaceMeta]:
    dev = _device()
    positions, req_id, cu_seqlens = _flat_positions(prefix_lengths, input_lengths, dev)
    pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=dev)
    il = torch.tensor(input_lengths, dtype=torch.int32, device=dev)
    return stub._build_workspace_meta(
        seqlen=int(positions.numel()),
        sp_int=prefix_lengths[0],
        device=dev,
        with_dense_cmp_topk=with_dense_cmp_topk,
        use_varlen=True,
        batch_size=len(input_lengths),
        cu_seqlens=cu_seqlens,
        input_lengths=il,
        prefix_lengths=pl,
        sp_per_req=pl.to(torch.int64),
        position_ids=positions,
        req_id_per_token=req_id,
        max_seqlen_q=int(il.max().item()),
    )


# -------------------------------------------------------------------------
# 1. Legacy B==1 — scalar values, slot_in_flat = arange + N + P
# -------------------------------------------------------------------------
class BuildWorkspaceMetaLegacyTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = _device()

    def test_csa_b1_cold_scalar_fields(self) -> None:
        """ratio=4, sp=0, S=20: N=5, P=0, gather=20, M=25.
        slot_in_flat = arange(20) + 5 + 0 = [5, 6, ..., 24]."""
        stub = _make_stub(win=8, compress_ratio=4)
        m = _build_meta_legacy(stub, sp=0, S=20, with_dense_cmp_topk=False)
        self.assertIsNotNone(m)
        self.assertEqual(m.M, 25)
        self.assertEqual(m.N, 5)
        self.assertEqual(int(m.swa_seq_lens.item()), 20)
        self.assertEqual(int(m.cmp_seq_lens.item()), 5)
        self.assertEqual(int(m.swa_gather_lens.item()), 20)
        self.assertTrue(
            torch.equal(
                m.qsl,
                torch.tensor([0, 20], dtype=torch.int32, device=self.device),
            )
        )
        expected_slot = torch.arange(20, dtype=torch.long, device=self.device) + 5
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected_slot))
        self.assertIsNone(m.dense_cmp_topk)

    def test_csa_b1_continuation_scalar_fields(self) -> None:
        """ratio=4, sp=10, S=20, win=8: N=(10+20)//4=7, P=min(10,7)=7,
        gather=27, M=34. slot_in_flat = arange(20) + (7+7)."""
        stub = _make_stub(win=8, compress_ratio=4)
        m = _build_meta_legacy(stub, sp=10, S=20, with_dense_cmp_topk=False)
        self.assertEqual(m.N, 7)
        self.assertEqual(m.M, 34)
        self.assertEqual(int(m.swa_seq_lens.item()), 30)
        self.assertEqual(int(m.cmp_seq_lens.item()), 7)
        self.assertEqual(int(m.swa_gather_lens.item()), 27)
        expected_slot = torch.arange(20, dtype=torch.long, device=self.device) + 14
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected_slot))

    def test_hca_b1_continuation_dense_cmp_topk(self) -> None:
        """ratio=128, sp=256, S=128: N=(256+128)//128=3.
        dense_cmp_topk shape [128, 3], values arange(3) per row."""
        stub = _make_stub(win=512, compress_ratio=128)
        m = _build_meta_legacy(stub, sp=256, S=128, with_dense_cmp_topk=True)
        self.assertEqual(m.N, 3)
        self.assertIsNotNone(m.dense_cmp_topk)
        self.assertEqual(m.dense_cmp_topk.shape, (128, 3))
        self.assertEqual(m.dense_cmp_topk.dtype, torch.int32)
        # Every row == arange(3).
        expected_row = torch.arange(3, dtype=torch.int32, device=self.device)
        for t in (0, 50, 127):
            self.assertTrue(torch.equal(m.dense_cmp_topk[t], expected_row))

    def test_hca_b1_dense_cmp_topk_zero_when_N_zero(self) -> None:
        """sp=0, S=64, ratio=128 ⇒ N=0. dense_cmp_topk must be ``[T, 0]``
        (not None) so the kernel still gets a valid empty grid."""
        stub = _make_stub(win=512, compress_ratio=128)
        m = _build_meta_legacy(stub, sp=0, S=64, with_dense_cmp_topk=True)
        self.assertEqual(m.N, 0)
        self.assertEqual(m.dense_cmp_topk.shape, (64, 0))

    def test_legacy_block_table_sliced_to_b1(self) -> None:
        """Block tables are pre-allocated for ``max_batch_size`` rows;
        legacy must slice ``[:1]`` so the meta only carries the active
        request's row."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=4, blocks_per_req=3)
        m = _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        self.assertEqual(m.swa_bt_int32.shape, (1, 3))
        self.assertEqual(m.cmp_bt_int32.shape, (1, 3))
        self.assertEqual(m.swa_bt_int32.dtype, torch.int32)

    def test_b1_varlen_tensors_passed_but_batch_eq_1_takes_legacy(self) -> None:
        """Even with DSV4_VARLEN_PREFILL=1 + all varlen tensors populated,
        ``batch_size == 1`` MUST take the legacy branch (== bit-equal).
        Bisect-channel guarantee."""
        stub = _make_stub(win=8, compress_ratio=4)
        legacy = _build_meta_legacy(stub, sp=4, S=10, with_dense_cmp_topk=False)
        varlen_b1 = _build_meta_varlen(stub, [4], [10], with_dense_cmp_topk=False)
        self.assertEqual(legacy.M, varlen_b1.M)
        self.assertEqual(legacy.N, varlen_b1.N)
        self.assertTrue(torch.equal(legacy.swa_seq_lens, varlen_b1.swa_seq_lens))
        self.assertTrue(torch.equal(legacy.cmp_seq_lens, varlen_b1.cmp_seq_lens))
        self.assertTrue(torch.equal(legacy.swa_gather_lens, varlen_b1.swa_gather_lens))
        self.assertTrue(torch.equal(legacy.qsl, varlen_b1.qsl))
        self.assertTrue(
            torch.equal(legacy.new_k_slot_in_flat, varlen_b1.new_k_slot_in_flat)
        )


# -------------------------------------------------------------------------
# 2. Warmup / unbound short-circuits
# -------------------------------------------------------------------------
class BuildWorkspaceMetaWarmupTest(unittest.TestCase):

    def test_no_kv_cache_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, kv_cache_present=False)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_no_block_tables_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, bt_present=False)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_swa_bt_missing_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, swa_bt_present=False)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_cmp_bt_missing_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, cmp_bt_present=False)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_swa_eb_zero_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, swa_eb_override=0)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_cmp_eb_zero_returns_none(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, cmp_eb_override=0)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )

    def test_swa_only_layer_returns_none(self) -> None:
        """compress_ratio==0 layers don't go through the workspace path."""
        stub = _make_stub(win=8, compress_ratio=0)
        self.assertIsNone(
            _build_meta_legacy(stub, sp=0, S=12, with_dense_cmp_topk=False)
        )


# -------------------------------------------------------------------------
# 3. Varlen B>=2 — per-request fields + N_max-padded scatter math
# -------------------------------------------------------------------------
@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA required for stack().tolist() sync"
)
class BuildWorkspaceMetaVarlenTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = _device()

    # ----- B==2 cold + cold (any_cont == False) ----------------------------
    def test_b2_cold_plus_cold_fields(self) -> None:
        """sp=[0,0], S=[16,12], ratio=4, win=8:
        seq_total=[16,12], N=[4,3], P=[0,0], gather=[16,12], M_b=[20,15]
        N_max=4, gather_max=16, M=20."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [0, 0], [16, 12], with_dense_cmp_topk=False)
        self.assertEqual(m.N, 4)
        self.assertEqual(m.M, 20)
        self.assertTrue(
            torch.equal(
                m.swa_seq_lens,
                torch.tensor([16, 12], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cmp_seq_lens,
                torch.tensor([4, 3], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.swa_gather_lens,
                torch.tensor([16, 12], dtype=torch.int32, device=self.device),
            )
        )
        # qsl == cu_seqlens [0, 16, 28]
        self.assertTrue(
            torch.equal(
                m.qsl,
                torch.tensor([0, 16, 28], dtype=torch.int32, device=self.device),
            )
        )
        # Block tables sliced to actual B==2.
        self.assertEqual(m.swa_bt_int32.shape[0], 2)
        self.assertEqual(m.cmp_bt_int32.shape[0], 2)

        # new_k_slot_in_flat per token:
        # Req 0 (T=0..15): slot = 0*M + N_max + 0 + local_pos = 4 + 0..15
        # Req 1 (T=16..27): slot = 1*M + N_max + 0 + local_pos = 20+4 + 0..11
        expected = torch.cat(
            [
                torch.arange(16, dtype=torch.long, device=self.device) + 4,
                torch.arange(12, dtype=torch.long, device=self.device) + 24,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected))

    # ----- B==2 cold + continuation ---------------------------------------
    def test_b2_cold_plus_continuation_fields(self) -> None:
        """sp=[0, 32], S=[8, 6], ratio=4, win=8:
        seq_total=[8, 38], N=[2, 9], P=[0, 7], gather=[8, 13]
        N_max=9, gather_max=13, M=22."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [0, 32], [8, 6], with_dense_cmp_topk=False)
        self.assertEqual(m.N, 9)
        self.assertEqual(m.M, 22)
        self.assertTrue(
            torch.equal(
                m.swa_seq_lens,
                torch.tensor([8, 38], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cmp_seq_lens,
                torch.tensor([2, 9], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.swa_gather_lens,
                torch.tensor([8, 13], dtype=torch.int32, device=self.device),
            )
        )
        # Slot calc:
        # Req 0 (T=0..7, P=0): slot = 0*22 + 9 + 0 + 0..7 = [9..16]
        # Req 1 (T=8..13, P=7): slot = 1*22 + 9 + 7 + 0..5 = [38..43]
        expected = torch.cat(
            [
                torch.arange(8, dtype=torch.long, device=self.device) + 9,
                torch.arange(6, dtype=torch.long, device=self.device) + 38,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected))

    # ----- B==2 all continuation, mixed sp --------------------------------
    def test_b2_all_continuation_mixed_sp(self) -> None:
        """sp=[20, 50], S=[5, 8], ratio=4, win=8:
        seq_total=[25, 58], N=[6, 14], P=[7, 7], gather=[12, 15]
        N_max=14, gather_max=15, M=29."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [20, 50], [5, 8], with_dense_cmp_topk=False)
        self.assertEqual(m.N, 14)
        self.assertEqual(m.M, 29)
        self.assertTrue(
            torch.equal(
                m.cmp_seq_lens,
                torch.tensor([6, 14], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.swa_gather_lens,
                torch.tensor([12, 15], dtype=torch.int32, device=self.device),
            )
        )
        # Both requests have prefix > win-1, so P_b is clamped at 7 for both.
        # Req 0 (T=0..4): slot = 0*29 + 14 + 7 + 0..4 = [21..25]
        # Req 1 (T=5..12): slot = 1*29 + 14 + 7 + 0..7 = [50..57]
        expected = torch.cat(
            [
                torch.arange(5, dtype=torch.long, device=self.device) + 21,
                torch.arange(8, dtype=torch.long, device=self.device) + 50,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected))

    # ----- B==2 with N_b=0 corner -----------------------------------------
    def test_b2_one_req_with_zero_N(self) -> None:
        """sp=[0, 0], S=[2, 8], ratio=4, win=8: req0 N_b=0 (S<ratio),
        req1 N_b=2. N_max=2."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [0, 0], [2, 8], with_dense_cmp_topk=False)
        self.assertEqual(m.N, 2)
        self.assertTrue(
            torch.equal(
                m.cmp_seq_lens,
                torch.tensor([0, 2], dtype=torch.int32, device=self.device),
            )
        )
        # gather_max = max(2, 8) = 8. M = 2 + 8 = 10.
        self.assertEqual(m.M, 10)
        # Req 0 (T=0..1): slot = 2 + 0..1 = [2, 3]
        # Req 1 (T=2..9): slot = 10 + 2 + 0..7 = [12..19]
        expected = torch.cat(
            [
                torch.arange(2, dtype=torch.long, device=self.device) + 2,
                torch.arange(8, dtype=torch.long, device=self.device) + 12,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected))

    # ----- HCA dense_cmp_topk under varlen --------------------------------
    def test_b2_hca_dense_cmp_topk(self) -> None:
        """HCA varlen with mixed sp: dense_cmp_topk shape [T_total, N_max],
        each row == arange(N_max). Per-token validity is masked downstream
        by the combine_topk kernel via COMPRESS_RATIO."""
        stub = _make_stub(win=512, compress_ratio=128, n_reqs=2)
        # sp=[0, 256], S=[64, 128]: N=[0, 3], N_max=3, T_total=192.
        m = _build_meta_varlen(stub, [0, 256], [64, 128], with_dense_cmp_topk=True)
        self.assertIsNotNone(m.dense_cmp_topk)
        self.assertEqual(m.dense_cmp_topk.shape, (192, 3))
        self.assertEqual(m.dense_cmp_topk.dtype, torch.int32)
        expected_row = torch.arange(3, dtype=torch.int32, device=self.device)
        # Spot-check tokens from each request — both should see the same
        # arange(N_max) row. The kernel handles per-token compressed_lens
        # via COMPRESS_RATIO + per-request seq_lens, not via row content.
        for t in (0, 63, 64, 191):
            self.assertTrue(torch.equal(m.dense_cmp_topk[t], expected_row))

    def test_b2_csa_dense_cmp_topk_is_none(self) -> None:
        """CSA path always passes ``with_dense_cmp_topk=False`` because the
        runtime indexer output replaces dense_cmp_topk."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [0, 32], [8, 6], with_dense_cmp_topk=False)
        self.assertIsNone(m.dense_cmp_topk)

    # ----- Block table slicing under B>1 ----------------------------------
    def test_block_table_sliced_to_active_b(self) -> None:
        """``swa_bt`` / ``cmp_bt`` are pre-allocated for ``max_batch_size``
        rows; varlen branch must slice ``[:batch_size]``. Verify this
        works when the stub has more rows than the active batch."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=4, blocks_per_req=3)
        m = _build_meta_varlen(stub, [0, 0], [10, 8], with_dense_cmp_topk=False)
        self.assertEqual(m.swa_bt_int32.shape, (2, 3))
        self.assertEqual(m.cmp_bt_int32.shape, (2, 3))
        self.assertEqual(m.swa_bt_int32.dtype, torch.int32)
        self.assertTrue(m.swa_bt_int32.is_contiguous())
        self.assertTrue(m.cmp_bt_int32.is_contiguous())


# -------------------------------------------------------------------------
# 4. Workspace overlay scatter — feed slot_in_flat to index_copy_ and
#    verify per-request placement matches expected per-row slices.
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for index_copy_")
class WorkspaceOverlayScatterTest(unittest.TestCase):
    """End-to-end check that the meta's ``new_k_slot_in_flat`` is a valid
    scatter index for the ``_attn_via_workspace`` step-4 BF16 overlay."""

    def setUp(self) -> None:
        self.device = _device()

    def _scatter(self, m: WorkspaceMeta, B: int, T: int, D: int) -> torch.Tensor:
        """Mirror the ``_attn_via_workspace`` step-4 sequence using the
        meta's pre-baked ``new_k_slot_in_flat``."""
        kv_full = (
            torch.arange(1, T + 1, dtype=torch.bfloat16, device=self.device)
            .view(T, 1)
            .expand(T, D)
            .contiguous()
        )
        workspace = torch.zeros((B, m.M, D), dtype=torch.bfloat16, device=self.device)
        workspace.view(B * m.M, D).index_copy_(0, m.new_k_slot_in_flat, kv_full)
        return workspace, kv_full

    def test_b1_scatter_matches_legacy_slice(self) -> None:
        """B==1 cold prefill: scatter via slot_in_flat == legacy
        ``workspace[:, N+P:N+P+S, :].copy_(kv_full)``."""
        stub = _make_stub(win=8, compress_ratio=4)
        sp, S = 4, 10
        m = _build_meta_legacy(stub, sp=sp, S=S, with_dense_cmp_topk=False)
        D = 8
        ws_vec, kv_full = self._scatter(m, B=1, T=S, D=D)
        # Legacy slice oracle.
        N = m.N
        P = min(sp, m.M - N - S)  # gather_len = S + P → P = gather_len - S
        # Sanity: P matches min(sp, win-1) == 4 for sp=4, win=8.
        self.assertEqual(P, min(sp, 8 - 1))
        ws_oracle = torch.zeros_like(ws_vec)
        ws_oracle[0, N + P : N + P + S, :].copy_(kv_full)
        self.assertTrue(torch.equal(ws_vec, ws_oracle))

    def test_b2_cold_plus_cont_scatter_per_request(self) -> None:
        """B==2 cold+cont: each request's new K must land at
        ``ws[b, N_max + P_b : N_max + P_b + S_b, :]`` and bleed nowhere else."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        prefix = [0, 32]
        S_list = [8, 6]
        m = _build_meta_varlen(stub, prefix, S_list, with_dense_cmp_topk=False)
        D = 8
        T = sum(S_list)
        ws, kv_full = self._scatter(m, B=2, T=T, D=D)

        N_max = m.N
        # Req 0 (P=0, S=8): kv_full[0:8] → ws[0, 4:12, :]. Wait — N_max=9 here
        # (recompute: sp=[0,32] ⇒ N=[0+8//4, 32+6//4]=[2, 9], N_max=9).
        # Req 0: ws[0, 9 + 0 : 9 + 0 + 8, :] = kv_full[0:8].
        self.assertEqual(N_max, 9)
        self.assertTrue(torch.equal(ws[0, 9:17, :], kv_full[0:8]))
        # Outside that slice in req 0 must remain zero (compressed [0:9] +
        # the SWA prefix tail [9:9] is 0 wide for sp=0, then tail).
        self.assertTrue(torch.all(ws[0, :9, :] == 0))
        self.assertTrue(torch.all(ws[0, 17:, :] == 0))
        # Req 1 (P=7, S=6): ws[1, 9+7 : 9+7+6, :] = ws[1, 16:22, :] = kv_full[8:14].
        self.assertTrue(torch.equal(ws[1, 16:22, :], kv_full[8:14]))
        # Compressed [0:9] + prefix tail [9:16] (7 rows of zeros) untouched
        # (zero-init); tail slot beyond M=22 doesn't exist.
        self.assertTrue(torch.all(ws[1, :16, :] == 0))

    def test_b2_all_continuation_clamped_p(self) -> None:
        """Both requests' sp > win-1 ⇒ both get ``P_b = win-1`` clamp.
        Verify each request's new K lands at the clamped offset."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        prefix = [20, 50]
        S_list = [5, 8]
        m = _build_meta_varlen(stub, prefix, S_list, with_dense_cmp_topk=False)
        D = 4
        T = sum(S_list)
        ws, kv_full = self._scatter(m, B=2, T=T, D=D)
        N_max = m.N
        self.assertEqual(N_max, 14)
        # Req 0: P = min(20, 7) = 7. Slot = N_max + 7 = 21. S=5.
        self.assertTrue(torch.equal(ws[0, 21:26, :], kv_full[0:5]))
        self.assertTrue(torch.all(ws[0, :21, :] == 0))
        # Tail from 26 to M=29.
        self.assertTrue(torch.all(ws[0, 26:, :] == 0))
        # Req 1: P = min(50, 7) = 7. Slot = N_max + 7 = 21. S=8.
        self.assertTrue(torch.equal(ws[1, 21:29, :], kv_full[5:13]))
        self.assertTrue(torch.all(ws[1, :21, :] == 0))


from rtp_llm.models_py.modules.dsv4.attn_type import (  # noqa: E402
    CSA_STATE,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
)

# =========================================================================
# Per-builder coverage: _build_compressor_meta, _build_csa_prefill_meta,
# _build_hca_prefill_meta. These three sit on top of
# _build_workspace_meta + delegate to compressor.prepare_metadata /
# indexer.prepare. We stub the compressor + indexer to capture call args
# and assert each builder threads varlen kwargs correctly.
# =========================================================================
from rtp_llm.models_py.modules.dsv4.fp8.attention import (  # noqa: E402
    AttentionFP8 as Attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    CsaPrefillMeta,
    HcaPrefillMeta,
    _use_varlen_prefill,
)


class _StubCompressor:
    """Capture every ``prepare_metadata`` call so the UT can assert the
    builder fed ``positions`` / ``b_idx`` / ``is_batched`` /
    ``seq_start_per_req`` / ``cu_seq_per_req`` correctly."""

    def __init__(self) -> None:
        self.calls: list = []

    def prepare_metadata(
        self,
        positions,
        b_idx,
        is_batched=False,
        seq_start_per_req=None,
        cu_seq_per_req=None,
    ):
        self.calls.append(
            dict(
                positions=positions,
                b_idx=b_idx,
                is_batched=is_batched,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )
        )
        return f"compressor_meta_call_{len(self.calls)}"


class _StubIndexer:
    """Capture every ``prepare`` call so the UT can assert the CSA
    builder threads varlen tensors through to the indexer."""

    def __init__(self, freqs_cis: Optional[torch.Tensor] = None) -> None:
        # Required by ``_ensure_freqs_cis_bound`` (called transitively).
        self.freqs_cis = freqs_cis if freqs_cis is not None else torch.empty(0)
        self.compressor = _StubCompressor()  # nested compressor (CSA-only)
        self.compressor.freqs_cis = self.freqs_cis  # type: ignore[attr-defined]
        self.calls: list = []

    def prepare(
        self,
        bsz,
        seqlen,
        sp_int,
        device,
        kv_block_table=None,
        kv_eb=0,
        *,
        use_varlen=None,
        batch_size=1,
        cu_seqlens=None,
        input_lengths=None,
        prefix_lengths=None,
        position_ids=None,
        req_id_per_token=None,
        max_seqlen_q=0,
    ):
        self.calls.append(
            dict(
                bsz=bsz,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=kv_block_table,
                kv_eb=kv_eb,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        )
        return f"indexer_meta_call_{len(self.calls)}"


def _make_meta_stub(
    *,
    win: int,
    compress_ratio: int,
    n_reqs: int = 2,
    blocks_per_req: int = 4,
    swa_eb: int = 256,
    cmp_eb: int = 64,
    indexer_eb: int = 32,
    state_eb: int = 32,
    bind_indexer: bool = False,
    device: Optional[torch.device] = None,
) -> _StubAttention:
    """Extend ``_make_stub`` with INDEXER_KV / INDEXER_STATE / CSA_STATE /
    HCA_STATE pools + a stub Compressor / Indexer attached to the stub
    Attention. The compressor is required by every builder; the indexer
    is required only by CSA (``compress_ratio == 4``)."""
    dev = device or _device()
    base = _make_stub(
        win=win,
        compress_ratio=compress_ratio,
        n_reqs=n_reqs,
        blocks_per_req=blocks_per_req,
        swa_eb=swa_eb,
        cmp_eb=cmp_eb,
        device=dev,
    )
    # Attach state + indexer pools so _set_compressor_pool_context's
    # `set_pool_context` lookups don't AttributeError. The pool tables
    # themselves are unused — the stubs for compressor/indexer don't
    # touch them, but the bind path requires the dict entries to exist.
    if base._block_tables_by_type is not None:
        for at in (INDEXER_KV, INDEXER_STATE, CSA_STATE, HCA_STATE):
            base._block_tables_by_type.setdefault(
                at, _make_block_table(n_reqs, blocks_per_req, dev)
            )
        base._eb_by_type[INDEXER_KV] = indexer_eb
        base._eb_by_type[INDEXER_STATE] = state_eb
        base._eb_by_type[CSA_STATE] = state_eb
        base._eb_by_type[HCA_STATE] = state_eb
    base.compressor = _StubCompressor()
    if bind_indexer:
        base.indexer = _StubIndexer()
    else:
        base.indexer = None
    base.layer_id = 0
    base.tp_size = 1
    base.tp_rank = 0
    return base


class _NoBindStubAttention(_StubAttention):
    """Variant that no-ops ``_set_compressor_pool_context`` /
    ``_clear_compressor_pool_context``. The real method walks the full
    pool view machinery (KVCache.get_layer_cache, _pool_view_3d_fp8,
    Compressor.set_pool_context); for prepare_metadata stub tests we just
    want to verify the dispatch logic calls compressor.prepare_metadata
    with the right args. Override here so the stub doesn't need to fake
    out KVCache + framework."""

    def _set_compressor_pool_context(self) -> None:
        pass

    def _clear_compressor_pool_context(self) -> None:
        pass

    _build_workspace_meta = Attention._build_workspace_meta
    _build_compressor_meta = Attention._build_compressor_meta
    _build_csa_prefill_meta = Attention._build_csa_prefill_meta
    _build_hca_prefill_meta = Attention._build_hca_prefill_meta


def _make_no_bind_stub(**kwargs) -> _NoBindStubAttention:
    """Promote a ``_StubAttention`` to ``_NoBindStubAttention`` (sub-class
    that no-ops the pool bind/unbind so the prepare_metadata UT doesn't
    have to fake KVCache.get_layer_cache + pool view tensors)."""
    base = _make_meta_stub(**kwargs)
    promoted = _NoBindStubAttention(
        window_size=base.window_size,
        compress_ratio=base.compress_ratio,
        block_tables=base._block_tables_by_type,
        eb_by_type=base._eb_by_type,
        kv_cache_present=base._kv_cache is not None,
    )
    promoted.compressor = base.compressor
    promoted.indexer = base.indexer
    promoted.layer_id = 0
    promoted.tp_size = 1
    promoted.tp_rank = 0
    return promoted


# -------------------------------------------------------------------------
# 5. _build_compressor_meta — varlen vs legacy dispatch (shared by HCA
#    + the inline CSA path). Verifies positions/b_idx/is_batched/
#    seq_start_per_req/cu_seq_per_req plumbing.
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for tensor scaffolding")
class BuildCompressorMetaTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = _device()

    def _call(
        self,
        stub: _NoBindStubAttention,
        sp_int: int,
        seqlen: int,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        prefix_lengths: Optional[list] = None,
        input_lengths: Optional[list] = None,
    ):
        """``use_varlen`` is mandatory — the new dispatch contract is
        ``_build_shared_prefill_meta`` reads the env once and threads the
        decision down. Sub-builder UTs must pass it explicitly so they
        don't re-read the env."""
        if use_varlen:
            assert prefix_lengths is not None and input_lengths is not None
            positions, req_id, cu_seqlens = _flat_positions(
                prefix_lengths, input_lengths, self.device
            )
            pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=self.device)
            il = torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
            return stub._build_compressor_meta(
                seqlen=int(positions.numel()),
                sp_int=prefix_lengths[0],
                device=self.device,
                use_varlen=True,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=il,
                prefix_lengths=pl,
                sp_per_req=pl.to(torch.int64),
                position_ids=positions,
                req_id_per_token=req_id,
                max_seqlen_q=int(il.max().item()),
            )
        return stub._build_compressor_meta(
            seqlen=seqlen, sp_int=sp_int, device=self.device, use_varlen=False
        )

    def test_legacy_b1_uses_build_prefill_positions(self) -> None:
        """``use_varlen=False`` (legacy / DSV4_VARLEN_PREFILL=0 bisect path):
        positions = arange(sp, sp+S), b_idx = zeros(S), is_batched=False,
        seq_start/cu_seq = None."""
        stub = _make_no_bind_stub(win=8, compress_ratio=128)  # HCA host compressor
        ret = self._call(stub, sp_int=10, seqlen=20, use_varlen=False)
        self.assertEqual(ret, "compressor_meta_call_1")
        call = stub.compressor.calls[0]
        self.assertEqual(call["is_batched"], False)
        self.assertIsNone(call["seq_start_per_req"])
        self.assertIsNone(call["cu_seq_per_req"])
        # positions == arange(sp, sp+S) per _build_prefill_positions contract.
        expected_pos = torch.arange(10, 30, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(call["positions"], expected_pos))
        # b_idx == zeros(S) (B==1 flat batch).
        self.assertTrue(
            torch.equal(
                call["b_idx"], torch.zeros(20, dtype=torch.long, device=self.device)
            )
        )

    def test_varlen_b2_threads_position_ids_and_req_id(self) -> None:
        """B==2 cold+cont: positions = position_ids.long(), b_idx =
        req_id_per_token.long(), is_batched=True, seq_start_per_req =
        prefix_lengths.int32, cu_seq_per_req = cu_seqlens.int32."""
        stub = _make_no_bind_stub(win=8, compress_ratio=128, n_reqs=2)
        ret = self._call(
            stub,
            sp_int=0,
            seqlen=0,
            use_varlen=True,
            batch_size=2,
            prefix_lengths=[0, 32],
            input_lengths=[8, 6],
        )
        self.assertEqual(ret, "compressor_meta_call_1")
        call = stub.compressor.calls[0]
        self.assertEqual(call["is_batched"], True)
        # positions == flat per-token absolute positions (req0 [0..7] +
        # req1 [32..37]).
        expected_pos = torch.cat(
            [
                torch.arange(0, 8, dtype=torch.long, device=self.device),
                torch.arange(32, 38, dtype=torch.long, device=self.device),
            ]
        )
        self.assertTrue(torch.equal(call["positions"], expected_pos))
        self.assertEqual(call["positions"].dtype, torch.long)
        self.assertTrue(call["positions"].is_contiguous())
        # b_idx == req_id_per_token (req0 zeros, req1 ones).
        expected_bidx = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            dtype=torch.long,
            device=self.device,
        )
        self.assertTrue(torch.equal(call["b_idx"], expected_bidx))
        # seq_start_per_req == prefix_lengths int32.
        self.assertEqual(call["seq_start_per_req"].dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                call["seq_start_per_req"],
                torch.tensor([0, 32], dtype=torch.int32, device=self.device),
            )
        )
        # cu_seq_per_req == cu_seqlens int32 = [0, 8, 14].
        self.assertEqual(call["cu_seq_per_req"].dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                call["cu_seq_per_req"],
                torch.tensor([0, 8, 14], dtype=torch.int32, device=self.device),
            )
        )

    def test_b1_under_varlen_collapses_to_legacy_positions(self) -> None:
        """``use_varlen=True`` + B==1 — under the new contract (env-driven,
        B-agnostic) the varlen branch runs but its positions / b_idx
        collapse to the same values ``_build_prefill_positions(sp, 1, S)``
        would have produced. Bit-equal collapse is the bisect channel
        between varlen and legacy."""
        stub = _make_no_bind_stub(win=8, compress_ratio=128)
        positions, req_id, cu_seqlens = _flat_positions([4], [10], self.device)
        pl = torch.tensor([4], dtype=torch.int32, device=self.device)
        il = torch.tensor([10], dtype=torch.int32, device=self.device)
        stub._build_compressor_meta(
            seqlen=10,
            sp_int=4,
            device=self.device,
            use_varlen=True,
            batch_size=1,
            cu_seqlens=cu_seqlens,
            input_lengths=il,
            prefix_lengths=pl,
            sp_per_req=pl.to(torch.int64),
            position_ids=positions,
            req_id_per_token=req_id,
            max_seqlen_q=10,
        )
        call = stub.compressor.calls[0]
        # Varlen branch — kernel-native batched dispatch even at B==1.
        self.assertEqual(call["is_batched"], True)
        self.assertIsNotNone(call["seq_start_per_req"])
        self.assertIsNotNone(call["cu_seq_per_req"])
        self.assertEqual(call["seq_start_per_req"].dtype, torch.int32)
        self.assertEqual(call["cu_seq_per_req"].dtype, torch.int32)
        # B==1 collapse: positions == arange(sp, sp+S), b_idx == zeros(S)
        # — bit-equal to _build_prefill_positions(sp=4, bsz=1, seqlen=10).
        expected_pos = torch.arange(4, 14, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(call["positions"], expected_pos))
        self.assertTrue(
            torch.equal(
                call["b_idx"],
                torch.zeros(10, dtype=torch.long, device=self.device),
            )
        )


# -------------------------------------------------------------------------
# 6. _build_csa_prefill_meta — wraps indexer.prepare + compressor +
#    workspace_meta. Verifies all three sub-calls receive the right
#    varlen kwargs.
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for tensor scaffolding")
class BuildCsaPrefillMetaTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = _device()

    def _call(
        self,
        stub: _NoBindStubAttention,
        prefix_lengths: list,
        input_lengths: list,
        *,
        use_varlen: bool = True,
    ) -> CsaPrefillMeta:
        positions, req_id, cu_seqlens = _flat_positions(
            prefix_lengths, input_lengths, self.device
        )
        pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=self.device)
        il = torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
        # ``_build_csa_prefill_meta`` asserts ``isinstance(self.indexer,
        # IndexerFP8)``. Patch the symbol the builder imports to our stub
        # base so the type check passes without dragging in the real
        # IndexerFP8 weights / DeepGEMM kernels.
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.indexer.IndexerFP8",
            _StubIndexer,
        ):
            return stub._build_csa_prefill_meta(
                seqlen=int(positions.numel()),
                sp_int=prefix_lengths[0],
                device=self.device,
                use_varlen=use_varlen,
                batch_size=len(input_lengths),
                cu_seqlens=cu_seqlens,
                input_lengths=il,
                prefix_lengths=pl,
                sp_per_req=pl.to(torch.int64),
                position_ids=positions,
                req_id_per_token=req_id,
                max_seqlen_q=int(il.max().item()),
            )

    def test_b1_varlen_indexer_and_compressor_args(self) -> None:
        """``use_varlen=True`` + B==1 — under the new contract, B==1 collapses
        to bit-equal varlen output. Indexer + nested compressor both run
        through the batched dispatch."""
        stub = _make_no_bind_stub(win=8, compress_ratio=4, n_reqs=1, bind_indexer=True)
        ret = self._call(stub, [4], [10], use_varlen=True)
        self.assertIsInstance(ret, CsaPrefillMeta)
        idx_call = stub.indexer.calls[0]
        self.assertEqual(idx_call["bsz"], 1)
        self.assertEqual(idx_call["seqlen"], 10)
        self.assertEqual(idx_call["sp_int"], 4)
        self.assertEqual(idx_call["batch_size"], 1)
        self.assertEqual(idx_call["max_seqlen_q"], 10)
        self.assertIs(
            idx_call["kv_block_table"],
            stub._block_tables_by_type[INDEXER_KV],
        )
        self.assertEqual(idx_call["kv_eb"], 32)
        # CSA inline compressor.prepare_metadata: varlen path even at B==1.
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], True)
        self.assertEqual(cmp_call["positions"].dtype, torch.long)
        # B==1 collapse: positions == arange(sp, sp+S) == position_ids.
        expected_pos = torch.arange(4, 14, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))
        # b_idx == zeros(S) for B==1.
        self.assertTrue(
            torch.equal(
                cmp_call["b_idx"],
                torch.zeros(10, dtype=torch.long, device=self.device),
            )
        )
        # seq_start_per_req == [4], cu_seq_per_req == [0, 10] (both int32).
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([4], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 10], dtype=torch.int32, device=self.device),
            )
        )

    def test_b1_legacy_indexer_and_compressor_args(self) -> None:
        """``use_varlen=False`` (DSV4_VARLEN_PREFILL=0 bisect channel):
        indexer takes legacy scalar sp/seqlen, inline compressor takes
        the legacy ``_build_prefill_positions`` path."""
        stub = _make_no_bind_stub(win=8, compress_ratio=4, n_reqs=1, bind_indexer=True)
        ret = self._call(stub, [4], [10], use_varlen=False)
        self.assertIsInstance(ret, CsaPrefillMeta)
        idx_call = stub.indexer.calls[0]
        self.assertEqual(idx_call["seqlen"], 10)
        self.assertEqual(idx_call["sp_int"], 4)
        # Inline compressor takes legacy branch → is_batched=False.
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], False)
        self.assertIsNone(cmp_call["seq_start_per_req"])
        self.assertIsNone(cmp_call["cu_seq_per_req"])
        expected_pos = torch.arange(4, 14, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))

    def test_b2_varlen_threads_all_kwargs(self) -> None:
        """B==2: indexer.prepare receives every per-request tensor,
        compressor.prepare_metadata uses position_ids/req_id_per_token
        AND is_batched + seq_start_per_req + cu_seq_per_req."""
        stub = _make_no_bind_stub(win=8, compress_ratio=4, n_reqs=2, bind_indexer=True)
        ret = self._call(stub, [0, 32], [8, 6], use_varlen=True)
        self.assertIsInstance(ret, CsaPrefillMeta)

        # ---- Indexer call ----
        idx_call = stub.indexer.calls[0]
        self.assertEqual(idx_call["bsz"], 1)  # legacy positional kept
        self.assertEqual(idx_call["batch_size"], 2)  # canonical varlen B
        self.assertEqual(idx_call["seqlen"], 14)  # T_total = 8+6
        self.assertEqual(idx_call["max_seqlen_q"], 8)  # max(input_lengths)
        # Per-request tensors threaded through.
        self.assertIsNotNone(idx_call["cu_seqlens"])
        self.assertTrue(
            torch.equal(
                idx_call["cu_seqlens"],
                torch.tensor([0, 8, 14], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                idx_call["prefix_lengths"],
                torch.tensor([0, 32], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                idx_call["input_lengths"],
                torch.tensor([8, 6], dtype=torch.int32, device=self.device),
            )
        )
        # position_ids per token — same as _flat_positions builds.
        expected_pos = torch.cat(
            [
                torch.arange(0, 8, dtype=torch.int64, device=self.device),
                torch.arange(32, 38, dtype=torch.int64, device=self.device),
            ]
        )
        self.assertTrue(torch.equal(idx_call["position_ids"], expected_pos))

        # ---- Compressor call (inline) ----
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], True)
        self.assertEqual(cmp_call["positions"].dtype, torch.long)
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))
        # b_idx == req_id_per_token long.
        expected_bidx = torch.tensor(
            [0] * 8 + [1] * 6,
            dtype=torch.long,
            device=self.device,
        )
        self.assertTrue(torch.equal(cmp_call["b_idx"], expected_bidx))
        # seq_start_per_req == prefix_lengths int32.
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([0, 32], dtype=torch.int32, device=self.device),
            )
        )
        # cu_seq_per_req == cu_seqlens int32.
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 8, 14], dtype=torch.int32, device=self.device),
            )
        )

        # ---- Workspace meta ----
        wm = ret.workspace_meta
        self.assertIsNotNone(wm)
        # CSA layer ⇒ dense_cmp_topk = None (indexer fills runtime topk).
        self.assertIsNone(wm.dense_cmp_topk)
        # Per-request fields baked correctly (delegated to _build_workspace_meta
        # which has its own dedicated test class above; just spot-check).
        self.assertEqual(int(wm.swa_seq_lens.shape[0]), 2)

    def test_b1_csa_returns_workspace_meta_with_no_dense_topk(self) -> None:
        """CSA layer always builds workspace_meta with dense_cmp_topk=None
        regardless of B — runtime indexer output replaces it."""
        stub = _make_no_bind_stub(win=8, compress_ratio=4, n_reqs=1, bind_indexer=True)
        ret = self._call(stub, [4], [10], use_varlen=True)
        self.assertIsNotNone(ret.workspace_meta)
        self.assertIsNone(ret.workspace_meta.dense_cmp_topk)
        # And indexer + compressor returns are bundled into the meta.
        self.assertEqual(ret.indexer_meta, "indexer_meta_call_1")
        self.assertEqual(ret.compressor_meta, "compressor_meta_call_1")


# -------------------------------------------------------------------------
# 7. _build_hca_prefill_meta — wraps _build_compressor_meta +
#    _build_workspace_meta(with_dense_cmp_topk=True). Verifies dense_cmp_topk
#    is built + compressor_meta delegated correctly.
# -------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for tensor scaffolding")
class BuildHcaPrefillMetaTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = _device()

    def _call(
        self,
        stub: _NoBindStubAttention,
        prefix_lengths: list,
        input_lengths: list,
        *,
        use_varlen: bool = True,
    ) -> HcaPrefillMeta:
        positions, req_id, cu_seqlens = _flat_positions(
            prefix_lengths, input_lengths, self.device
        )
        pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=self.device)
        il = torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
        return stub._build_hca_prefill_meta(
            seqlen=int(positions.numel()),
            sp_int=prefix_lengths[0],
            device=self.device,
            use_varlen=use_varlen,
            batch_size=len(input_lengths),
            cu_seqlens=cu_seqlens,
            input_lengths=il,
            prefix_lengths=pl,
            sp_per_req=pl.to(torch.int64),
            position_ids=positions,
            req_id_per_token=req_id,
            max_seqlen_q=int(il.max().item()),
        )

    def test_b1_varlen_compressor_call(self) -> None:
        """``use_varlen=True`` + B==1 — varlen branch under the new contract.
        positions/b_idx collapse to bit-equal of the legacy arange/zeros."""
        stub = _make_no_bind_stub(win=512, compress_ratio=128, n_reqs=1)
        ret = self._call(stub, [256], [128], use_varlen=True)
        self.assertIsInstance(ret, HcaPrefillMeta)
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], True)
        # positions = arange(256, 256+128) = position_ids.
        expected_pos = torch.arange(256, 384, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))
        self.assertTrue(
            torch.equal(
                cmp_call["b_idx"],
                torch.zeros(128, dtype=torch.long, device=self.device),
            )
        )
        # seq_start / cu_seq populated under varlen.
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([256], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 128], dtype=torch.int32, device=self.device),
            )
        )
        # workspace_meta has dense_cmp_topk (HCA path).
        wm = ret.workspace_meta
        self.assertIsNotNone(wm)
        self.assertIsNotNone(wm.dense_cmp_topk)
        # N=(256+128)//128=3, T=128.
        self.assertEqual(wm.dense_cmp_topk.shape, (128, 3))

    def test_b1_legacy_compressor_call(self) -> None:
        """``use_varlen=False`` (DSV4_VARLEN_PREFILL=0 bisect channel):
        compressor takes the legacy ``_build_prefill_positions`` path."""
        stub = _make_no_bind_stub(win=512, compress_ratio=128, n_reqs=1)
        ret = self._call(stub, [256], [128], use_varlen=False)
        self.assertIsInstance(ret, HcaPrefillMeta)
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], False)
        self.assertIsNone(cmp_call["seq_start_per_req"])
        self.assertIsNone(cmp_call["cu_seq_per_req"])
        expected_pos = torch.arange(256, 384, dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))
        # Workspace meta still constructed (legacy branch in _build_workspace_meta).
        wm = ret.workspace_meta
        self.assertIsNotNone(wm)
        self.assertIsNotNone(wm.dense_cmp_topk)
        self.assertEqual(wm.dense_cmp_topk.shape, (128, 3))

    def test_b2_varlen_compressor_and_workspace_dense_topk(self) -> None:
        """B==2: compressor gets is_batched=True + per-request tensors;
        workspace gets dense_cmp_topk shape [T_total, N_max]."""
        stub = _make_no_bind_stub(win=512, compress_ratio=128, n_reqs=2)
        # sp=[0, 256], S=[64, 128]: N=[0, 3], N_max=3, T_total=192.
        ret = self._call(stub, [0, 256], [64, 128], use_varlen=True)
        cmp_call = stub.compressor.calls[0]
        self.assertEqual(cmp_call["is_batched"], True)
        # positions = req0 [0..63] + req1 [256..383]
        expected_pos = torch.cat(
            [
                torch.arange(0, 64, dtype=torch.long, device=self.device),
                torch.arange(256, 384, dtype=torch.long, device=self.device),
            ]
        )
        self.assertTrue(torch.equal(cmp_call["positions"], expected_pos))
        # b_idx = req_id long.
        expected_bidx = torch.tensor(
            [0] * 64 + [1] * 128,
            dtype=torch.long,
            device=self.device,
        )
        self.assertTrue(torch.equal(cmp_call["b_idx"], expected_bidx))
        # seq_start_per_req = prefix_lengths int32.
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([0, 256], dtype=torch.int32, device=self.device),
            )
        )
        # workspace dense_cmp_topk: [192, 3] arange.
        wm = ret.workspace_meta
        self.assertIsNotNone(wm)
        self.assertEqual(wm.dense_cmp_topk.shape, (192, 3))
        self.assertEqual(wm.N, 3)

    def test_hca_has_no_indexer_call(self) -> None:
        """HCA path doesn't call indexer.prepare (HCA has no indexer).
        Stub indexer left attached as None — verify nothing tries to use it."""
        stub = _make_no_bind_stub(
            win=512, compress_ratio=128, n_reqs=1, bind_indexer=False
        )
        # Should not raise. self.indexer is None on HCA layers in the prod
        # construction path; the builder must never touch it.
        self._call(stub, [256], [128])
        self.assertIsNone(stub.indexer)


# -------------------------------------------------------------------------
# 8. _use_varlen_prefill — env-driven dispatch gate threaded down from
#    Attention._build_shared_prefill_meta to every sub-builder. Pin the
#    semantics so a future env-var rename doesn't silently change the
#    default dispatch.
# -------------------------------------------------------------------------
class UseVarlenPrefillGateTest(unittest.TestCase):

    def test_env_default_is_on(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_VARLEN_PREFILL", None)
            self.assertTrue(_use_varlen_prefill())

    def test_env_one_enables(self) -> None:
        with mock.patch.dict(os.environ, {"DSV4_VARLEN_PREFILL": "1"}):
            self.assertTrue(_use_varlen_prefill())

    def test_env_zero_disables(self) -> None:
        with mock.patch.dict(os.environ, {"DSV4_VARLEN_PREFILL": "0"}):
            self.assertFalse(_use_varlen_prefill())


if __name__ == "__main__":
    unittest.main()
