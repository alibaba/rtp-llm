"""UT: CSA / HCA workspace prefill metadata varlen (B>=1) builder.

Phase-3b of the dsv4 prefill varlen migration. Validates that
``Attention._build_workspace_meta`` correctly produces:

  1. **B==1 varlen** — bit-equal to the scalar reference
     (sp_int + seqlen scalars).
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

import unittest
from typing import Optional
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import row_scatter
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    CsaPrefillMeta,
    HcaPrefillMeta,
    WorkspaceMeta,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV, SWA_KV

_SWA_FP8_ENTRY_BYTES = 584


# -------------------------------------------------------------------------
# Stub helpers
# -------------------------------------------------------------------------
class _StubAttention:
    """Minimal stand-in for ``Attention`` that exposes only what
    ``_build_workspace_meta`` actually reads:

      * ``self.window_size`` / ``self.compress_ratio``
      * ``self._kv_cache`` (truthy / None)
      * ``self._block_tables_by_type[SWA_KV/CSA_KV/HCA_KV]``
      * ``self._pool_entries_per_block(tag)`` → ``self._eb_by_type``
    """

    def __init__(
        self,
        *,
        window_size: int,
        compress_ratio: int,
        block_tables: Optional[dict] = None,
        eb_by_type: Optional[dict] = None,
        kv_cache_present: bool = True,
        cp_byte_sliced: bool = False,
        cp_size: int = 2,
    ) -> None:
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self._cp_byte_sliced = bool(cp_byte_sliced)
        self._cp_size = int(cp_size)

        class _StubKvCache:
            """Minimal stand-in for the tag-driven C++ ``KVCache`` binding."""

            group_tags = [SWA_KV, CSA_KV, HCA_KV]
            seq_size_per_block = 256
            kernel_seq_size_per_block = 256

            def get_seq_size_per_block(self, tag: str) -> int:
                if tag not in self.group_tags:
                    raise RuntimeError(f"unknown cache tag {tag!r}")
                return self.seq_size_per_block

            def get_kernel_seq_size_per_block(self, tag: str) -> int:
                if tag not in self.group_tags:
                    raise RuntimeError(f"unknown cache tag {tag!r}")
                return self.kernel_seq_size_per_block

        self._kv_cache = _StubKvCache() if kv_cache_present else None
        self._block_tables_by_type = block_tables
        self._eb_by_type = eb_by_type or {}
        swa_bt = block_tables.get(SWA_KV) if block_tables is not None else None
        if self._cp_byte_sliced and swa_bt is not None:
            swa_eb = int(self._eb_by_type.get(SWA_KV, 0))
            num_blocks = int(swa_bt.max().item()) + 1
            local_slice_bytes = max(1, swa_eb * _SWA_FP8_ENTRY_BYTES // self._cp_size)
            self._raw_u8 = torch.empty(
                num_blocks,
                local_slice_bytes,
                dtype=torch.uint8,
                device=swa_bt.device,
            )
        else:
            self._raw_u8 = None

    def _pool_entries_per_block(self, attn_type: str) -> int:
        return int(self._eb_by_type.get(attn_type, 0))

    def _swa_entries_per_block(self) -> int:
        return self._pool_entries_per_block(SWA_KV)

    def _swa_cp_byte_sliced(self) -> bool:
        return self._cp_byte_sliced

    def _pool_raw_u8(self, attn_type: str) -> Optional[torch.Tensor]:
        if attn_type != SWA_KV:
            return None
        return self._raw_u8

    def _build_swa_cp_byte_compaction(
        self,
        slot_mapping: torch.Tensor,
        full_entries_per_block: int,
        validation_site: str,
        negative_mode: str,
        gather_lens: Optional[torch.Tensor] = None,
    ):
        if not self._swa_cp_byte_sliced():
            return None
        raw = self._pool_raw_u8(SWA_KV)
        assert raw is not None, "byte-sliced FP8 SWA pool unavailable"
        return build_cp_byte_sliced_slot_compaction(
            slot_mapping,
            full_entries_per_block=full_entries_per_block,
            num_blocks=int(raw.shape[0]),
            validation_site=validation_site,
            negative_mode=negative_mode,
            gather_lens=gather_lens,
        )

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
    cp_byte_sliced: bool = False,
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
        cp_byte_sliced=cp_byte_sliced,
    )


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_meta_legacy(
    stub: _StubAttention, sp: int, S: int, with_dense_cmp_topk: bool
) -> Optional[WorkspaceMeta]:
    """B==1 scalar reference through the production varlen path."""
    return _build_meta_varlen(stub, [sp], [S], with_dense_cmp_topk)


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

    def test_b1_varlen_tensors_match_scalar_reference(self) -> None:
        """With all varlen tensors populated, ``batch_size == 1`` must
        collapse to the scalar reference."""
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

        # new_k_slot_in_flat per token (ragged layout):
        # req0: cu_cmp=0, N=4, gather=16 => cu_swa=4, ws=20 => slot = 4 + 0..15 = [4..19]
        # req1: cu_cmp=20, N=3, gather=12 => cu_swa=23, ws=15 => slot = 23 + 0..11 = [23..34]
        self.assertEqual(m.total_ws_rows, 35)
        self.assertTrue(
            torch.equal(
                m.cu_cmp_offsets,
                torch.tensor([0, 20], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cu_swa_offsets,
                torch.tensor([4, 23], dtype=torch.int32, device=self.device),
            )
        )
        expected = torch.cat(
            [
                torch.arange(16, dtype=torch.long, device=self.device) + 4,
                torch.arange(12, dtype=torch.long, device=self.device) + 23,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(m.new_k_slot_in_flat, expected))

    def test_wrapped_token_dim_matches_flat(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        dev = self.device
        positions, req_id, cu_seqlens = _flat_positions([0, 32], [8, 6], dev)
        pl = torch.tensor([0, 32], dtype=torch.int32, device=dev)
        il = torch.tensor([8, 6], dtype=torch.int32, device=dev)
        flat = stub._build_workspace_meta(
            seqlen=int(positions.numel()),
            sp_int=0,
            device=dev,
            with_dense_cmp_topk=False,
            use_varlen=True,
            batch_size=2,
            cu_seqlens=cu_seqlens,
            input_lengths=il,
            prefix_lengths=pl,
            sp_per_req=pl.to(torch.int64),
            position_ids=positions,
            req_id_per_token=req_id,
            max_seqlen_q=int(il.max().item()),
        )
        wrapped = stub._build_workspace_meta(
            seqlen=int(positions.numel()),
            sp_int=0,
            device=dev,
            with_dense_cmp_topk=False,
            use_varlen=True,
            batch_size=2,
            cu_seqlens=cu_seqlens.view(1, -1),
            input_lengths=il.view(1, -1),
            prefix_lengths=pl.view(1, -1),
            sp_per_req=pl.to(torch.int64).view(1, -1),
            position_ids=positions.view(1, -1),
            req_id_per_token=req_id.view(1, -1),
            max_seqlen_q=int(il.max().item()),
        )
        self.assertTrue(
            torch.equal(wrapped.new_k_slot_in_flat, flat.new_k_slot_in_flat)
        )
        self.assertTrue(torch.equal(wrapped.swa_seq_lens, flat.swa_seq_lens))
        self.assertTrue(torch.equal(wrapped.qsl, flat.qsl))

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
        # Ragged slot calc:
        # Req 0 (N=2, P=0, S=8, ws=10): cu_cmp=0, cu_swa=2, slot = [2..9]
        # Req 1 (N=9, P=7, S=6, ws=22): cu_cmp=10, cu_swa=19, slot = 19+7+0..5 = [26..31]
        self.assertEqual(m.total_ws_rows, 32)
        self.assertTrue(
            torch.equal(
                m.cu_cmp_offsets,
                torch.tensor([0, 10], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cu_swa_offsets,
                torch.tensor([2, 19], dtype=torch.int32, device=self.device),
            )
        )
        expected = torch.cat(
            [
                torch.arange(8, dtype=torch.long, device=self.device) + 2,
                torch.arange(6, dtype=torch.long, device=self.device) + 26,
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
        # Ragged offsets:
        # Req 0 (N=6, P=7, S=5, ws=18): cu_cmp=0, cu_swa=6, slot = 6+7+0..4 = [13..17]
        # Req 1 (N=14, P=7, S=8, ws=29): cu_cmp=18, cu_swa=32, slot = 32+7+0..7 = [39..46]
        self.assertEqual(m.total_ws_rows, 47)
        self.assertTrue(
            torch.equal(
                m.cu_cmp_offsets,
                torch.tensor([0, 18], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cu_swa_offsets,
                torch.tensor([6, 32], dtype=torch.int32, device=self.device),
            )
        )
        expected = torch.cat(
            [
                torch.arange(5, dtype=torch.long, device=self.device) + 13,
                torch.arange(8, dtype=torch.long, device=self.device) + 39,
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
        # Ragged offsets:
        # Req 0 (N=0, P=0, S=2, ws=2): cu_cmp=0, cu_swa=0, slot = [0, 1]
        # Req 1 (N=2, P=0, S=8, ws=10): cu_cmp=2, cu_swa=4, slot = [4..11]
        self.assertEqual(m.total_ws_rows, 12)
        self.assertTrue(
            torch.equal(
                m.cu_cmp_offsets,
                torch.tensor([0, 2], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                m.cu_swa_offsets,
                torch.tensor([0, 4], dtype=torch.int32, device=self.device),
            )
        )
        expected = torch.cat(
            [
                torch.arange(2, dtype=torch.long, device=self.device),
                torch.arange(8, dtype=torch.long, device=self.device) + 4,
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

    def test_cp_byte_sliced_precomputes_swa_cache_compaction(self) -> None:
        stub = _make_stub(
            win=8,
            compress_ratio=4,
            n_reqs=2,
            blocks_per_req=4,
            swa_eb=16,
            cmp_eb=8,
            cp_byte_sliced=True,
        )
        m = _build_meta_varlen(stub, [0, 32], [8, 6], with_dense_cmp_topk=False)
        self.assertIsNotNone(m.swa_cache_slot_mapping)
        self.assertIsNotNone(m.swa_cache_compaction)
        self.assertEqual(
            tuple(m.swa_cache_compaction.compact_slots.shape),
            tuple(m.swa_cache_slot_mapping.shape),
        )
        self.assertEqual(m.swa_cache_compaction.gather_lens_cpu, (0, 7))
        self.assertGreater(int(m.swa_cache_compaction.unique_blocks.numel()), 0)

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
        del B
        kv_full = (
            torch.arange(1, T + 1, dtype=torch.bfloat16, device=self.device)
            .view(T, 1)
            .expand(T, D)
            .contiguous()
        )
        workspace = torch.zeros(
            (max(m.total_ws_rows, 1), D), dtype=torch.bfloat16, device=self.device
        )
        row_scatter(workspace, m.new_k_slot_in_flat, kv_full)
        return workspace, kv_full

    def test_b1_scatter_matches_legacy_slice(self) -> None:
        """B==1 cold prefill: scatter via slot_in_flat == legacy
        ``workspace[N+P:N+P+S, :].copy_(kv_full)``."""
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
        ws_oracle[N + P : N + P + S, :].copy_(kv_full)
        self.assertTrue(torch.equal(ws_vec, ws_oracle))

    def test_b2_cold_plus_cont_scatter_per_request(self) -> None:
        """B==2 cold+cont: each request's new K must land at
        ``ws[cu_swa[b] + P_b : cu_swa[b] + P_b + S_b, :]`` and bleed nowhere else."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        prefix = [0, 32]
        S_list = [8, 6]
        m = _build_meta_varlen(stub, prefix, S_list, with_dense_cmp_topk=False)
        D = 8
        T = sum(S_list)
        ws, kv_full = self._scatter(m, B=2, T=T, D=D)

        # Req 0: cu_cmp=0, N=2, P=0, S=8 -> cu_swa=2, slice=[2:10]
        # Req 1: cu_cmp=10, N=9, P=7, S=6 -> cu_swa=19, slice=[26:32]
        self.assertEqual(m.total_ws_rows, 32)
        self.assertTrue(torch.equal(ws[2:10, :], kv_full[0:8]))
        self.assertTrue(torch.all(ws[:2, :] == 0))
        self.assertTrue(torch.all(ws[10:26, :] == 0))
        self.assertTrue(torch.equal(ws[26:32, :], kv_full[8:14]))

    def test_b2_all_continuation_clamped_p(self) -> None:
        """Both requests' sp > win-1 ⇒ both get ``P_b = win-1`` clamp.
        Verify each request's new K lands at the clamped ragged offset."""
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        prefix = [20, 50]
        S_list = [5, 8]
        m = _build_meta_varlen(stub, prefix, S_list, with_dense_cmp_topk=False)
        D = 4
        T = sum(S_list)
        ws, kv_full = self._scatter(m, B=2, T=T, D=D)
        # Req 0: N=6, P=7, S=5 -> cu_swa=6, slice=[13:18]
        # Req 1: cu_cmp=18, N=14, P=7, S=8 -> cu_swa=32, slice=[39:47]
        self.assertEqual(m.total_ws_rows, 47)
        self.assertTrue(torch.equal(ws[13:18, :], kv_full[0:5]))
        self.assertTrue(torch.all(ws[:13, :] == 0))
        self.assertTrue(torch.all(ws[18:39, :] == 0))
        self.assertTrue(torch.equal(ws[39:47, :], kv_full[5:13]))

    def test_row_scatter_overlay_matches_index_copy_bf16_512(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=2)
        m = _build_meta_varlen(stub, [0, 32], [8, 6], with_dense_cmp_topk=False)
        T, D = 14, 512
        kv_full = torch.randn((T, D), dtype=torch.bfloat16, device=self.device)
        for slot_dtype in (torch.int64, torch.int32):
            with self.subTest(slot_dtype=slot_dtype):
                ws_triton = torch.zeros(
                    (max(m.total_ws_rows, 1), D),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                ws_ref = torch.zeros_like(ws_triton)
                slots = m.new_k_slot_in_flat.to(dtype=slot_dtype)
                row_scatter(ws_triton, slots, kv_full)
                ws_ref.index_copy_(0, slots.to(torch.long), kv_full)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(ws_triton.view(torch.int16), ws_ref.view(torch.int16))
                )


# =========================================================================
# Per-builder coverage: _build_compressor_meta, _build_csa_prefill_meta,
# _build_hca_prefill_meta. These three sit on top of
# _build_workspace_meta + delegate to compressor.prepare_metadata /
# indexer.prepare. We stub the compressor + indexer to capture call args
# and assert each builder threads varlen kwargs correctly.
# =========================================================================
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (  # noqa: E402
    CSA_STATE,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
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
        has_prefix,
        is_batched=False,
        seq_start_per_req=None,
        cu_seq_per_req=None,
    ):
        self.calls.append(
            dict(
                positions=positions,
                b_idx=b_idx,
                has_prefix=has_prefix,
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
        has_prefix=False,
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
                has_prefix=has_prefix,
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
                has_prefix=any(v > 0 for v in prefix_lengths),
            )
        return stub._build_compressor_meta(
            seqlen=seqlen,
            sp_int=sp_int,
            device=self.device,
            use_varlen=False,
            has_prefix=sp_int > 0,
        )

    def test_use_varlen_false_uses_single_request_positions(self) -> None:
        """``use_varlen=False`` direct helper path:
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
        prefix_lengths.long, cu_seq_per_req = cu_seqlens.long."""
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
        # seq_start_per_req == prefix_lengths int64.
        self.assertEqual(call["seq_start_per_req"].dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                call["seq_start_per_req"],
                torch.tensor([0, 32], dtype=torch.int64, device=self.device),
            )
        )
        # cu_seq_per_req == cu_seqlens int64 = [0, 8, 14].
        self.assertEqual(call["cu_seq_per_req"].dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                call["cu_seq_per_req"],
                torch.tensor([0, 8, 14], dtype=torch.int64, device=self.device),
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
            has_prefix=True,
        )
        call = stub.compressor.calls[0]
        # Varlen branch — kernel-native batched dispatch even at B==1.
        self.assertEqual(call["is_batched"], True)
        self.assertIsNotNone(call["seq_start_per_req"])
        self.assertIsNotNone(call["cu_seq_per_req"])
        self.assertEqual(call["seq_start_per_req"].dtype, torch.int64)
        self.assertEqual(call["cu_seq_per_req"].dtype, torch.int64)
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
                has_prefix=any(v > 0 for v in prefix_lengths),
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
        # seq_start_per_req == [4], cu_seq_per_req == [0, 10] (both int64).
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([4], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 10], dtype=torch.int64, device=self.device),
            )
        )

    def test_b1_legacy_indexer_and_compressor_args(self) -> None:
        """``use_varlen=False`` is rejected by the FP8 prefill path."""
        stub = _make_no_bind_stub(win=8, compress_ratio=4, n_reqs=1, bind_indexer=True)
        with self.assertRaisesRegex(RuntimeError, "requires varlen metadata"):
            self._call(stub, [4], [10], use_varlen=False)

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
        # seq_start_per_req == prefix_lengths int64.
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([0, 32], dtype=torch.int64, device=self.device),
            )
        )
        # cu_seq_per_req == cu_seqlens int64.
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 8, 14], dtype=torch.int64, device=self.device),
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
            has_prefix=any(v > 0 for v in prefix_lengths),
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
                torch.tensor([256], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                cmp_call["cu_seq_per_req"],
                torch.tensor([0, 128], dtype=torch.int64, device=self.device),
            )
        )
        # workspace_meta has dense_cmp_topk (HCA path).
        wm = ret.workspace_meta
        self.assertIsNotNone(wm)
        self.assertIsNotNone(wm.dense_cmp_topk)
        # N=(256+128)//128=3, T=128.
        self.assertEqual(wm.dense_cmp_topk.shape, (128, 3))

    def test_b1_legacy_compressor_call(self) -> None:
        """``use_varlen=False`` is rejected by the FP8 prefill path."""
        stub = _make_no_bind_stub(win=512, compress_ratio=128, n_reqs=1)
        with self.assertRaisesRegex(RuntimeError, "requires varlen metadata"):
            self._call(stub, [256], [128], use_varlen=False)

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
        # seq_start_per_req = prefix_lengths int64.
        self.assertTrue(
            torch.equal(
                cmp_call["seq_start_per_req"],
                torch.tensor([0, 256], dtype=torch.int64, device=self.device),
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RaggedWorkspaceEndToEndTest(unittest.TestCase):
    """End-to-end mixed-reuse CSA / HCA / CP>1 / Raw-Q merge workspace tests."""

    def setUp(self) -> None:
        self.device = _device()
        torch.manual_seed(2026)

    def test_cp_host_lengths_zero_sync_mode2_offsets(self) -> None:
        stub = _make_stub(win=8, compress_ratio=4, n_reqs=3, blocks_per_req=4)
        # Asymmetric mixed requests: req0=(sp=0, S=4), req1=(sp=32, S=2), req2=(sp=4, S=6)
        prefix_list = [0, 32, 4]
        input_list = [4, 2, 6]
        # N = [1, 8, 2]; P = [0, 7, 4]; gather = [4, 9, 10] => ws = [5, 17, 12] -> total = 34
        # Compare vs padded: B * M = 3 * (8 + 10) = 54.
        pl = torch.tensor(prefix_list, dtype=torch.int32, device=self.device)
        il_full = torch.tensor(input_list, dtype=torch.int32, device=self.device)
        cu_full = torch.tensor([0, 4, 6, 12], dtype=torch.int32, device=self.device)
        local_il = torch.tensor([2, 1, 3], dtype=torch.int32, device=self.device)
        local_cu = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device=self.device)
        stub._cp_ctx = type(
            "_Cp",
            (),
            dict(
                cp_size=2,
                cp_rank=0,
                kv_cache_sharded=False,
                input_lengths_global=il_full,
                input_lengths_global_host=tuple(input_list),
                prefix_lengths=pl,
                prefix_lengths_host=tuple(prefix_list),
                cu_seqlens_global=cu_full,
                seq_len_full=12,
                prefix_length=0,
            ),
        )()
        positions = torch.arange(6, dtype=torch.int64, device=self.device)
        req_ids = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64, device=self.device)
        with mock.patch.object(
            torch.Tensor,
            "tolist",
            side_effect=AssertionError("Mode 2 CP host-lengths path must not D2H sync"),
        ):
            wm = stub._build_workspace_meta(
                seqlen=6,
                sp_int=0,
                device=self.device,
                with_dense_cmp_topk=True,
                use_varlen=True,
                batch_size=3,
                cu_seqlens=local_cu,
                input_lengths=local_il,
                prefix_lengths=pl,
                sp_per_req=pl.long(),
                position_ids=positions,
                req_id_per_token=req_ids,
                max_seqlen_q=3,
            )
        self.assertIsNotNone(wm)
        self.assertEqual(wm.total_ws_rows, 34)
        self.assertLess(wm.total_ws_rows, 3 * wm.M)
        self.assertEqual(wm.cu_swa_offsets_cpu, (1, 13, 24))
        self.assertTrue(
            torch.equal(
                wm.cu_cmp_offsets,
                torch.tensor([0, 5, 22], dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                wm.cu_swa_offsets,
                torch.tensor([1, 13, 24], dtype=torch.int32, device=self.device),
            )
        )

    def _make_exec_attention(
        self,
        *,
        compress_ratio: int,
        win: int,
        eb_cmp: int,
        eb_swa: int,
        bt_cmp: torch.Tensor,
        bt_swa: torch.Tensor,
        cmp_pool: torch.Tensor,
        swa_pool: Optional[torch.Tensor],
        cp_ctx=None,
        swa_raw_u8: Optional[torch.Tensor] = None,
    ) -> Attention:
        layer = Attention.__new__(Attention)
        torch.nn.Module.__init__(layer)
        layer.layer_id = 1
        layer.compress_ratio = compress_ratio
        layer.window_size = win
        layer.head_dim = 512
        layer.dim = 16
        layer.n_heads = 1
        layer.softmax_scale = 1.0
        layer.attn_sink = None
        layer._cp_ctx = cp_ctx

        class _Kv:
            group_tags = [SWA_KV, CSA_KV, HCA_KV]

            def get_seq_size_per_block(self, tag: str) -> int:
                return eb_swa if tag == SWA_KV else eb_cmp * compress_ratio

            def get_kernel_seq_size_per_block(self, tag: str) -> int:
                return self.get_seq_size_per_block(tag)

        cmp_at = CSA_KV if compress_ratio == 4 else HCA_KV
        layer._kv_cache = _Kv()
        layer._block_tables_by_type = {SWA_KV: bt_swa, cmp_at: bt_cmp}
        layer._pool_entries_per_block = (
            lambda tag: eb_swa if tag == SWA_KV else eb_cmp
        )  # type: ignore[assignment]
        layer._swa_entries_per_block = lambda: eb_swa  # type: ignore[assignment]
        layer._swa_cp_byte_sliced = lambda: swa_raw_u8 is not None  # type: ignore[assignment]
        layer._pool_raw_u8 = (
            lambda tag: swa_raw_u8 if tag == SWA_KV else None
        )  # type: ignore[assignment]
        layer._pool_view_3d_fp8 = (
            lambda tag: swa_pool if tag == SWA_KV else cmp_pool
        )  # type: ignore[assignment]
        layer._should_async_workspace_reads_for_prefill = lambda c: False  # type: ignore[assignment]
        return layer

    def test_mixed_reuse_csa_and_hca_attn_via_workspace(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta, PrefillQKV

        win = 8
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                eb_cmp = 8
                eb_swa = 8
                prefix_list = [0, 32, 8] if ratio == 4 else [0, 256, 128]
                input_list = [6, 4, 12] if ratio == 4 else [64, 128, 32]
                B = 3
                max_blocks = max((p + s + eb_swa - 1) // eb_swa for p, s in zip(prefix_list, input_list)) + 1
                bt_swa = _make_block_table(B, max_blocks, self.device).to(torch.int32)
                bt_cmp = _make_block_table(B, max_blocks, self.device).to(torch.int32)
                num_blocks = int(bt_swa.max().item()) + 1
                cmp_pool = torch.zeros(
                    (num_blocks, eb_cmp, _SWA_FP8_ENTRY_BYTES),
                    dtype=torch.uint8,
                    device=self.device,
                )
                swa_pool = torch.zeros(
                    (num_blocks, eb_swa, _SWA_FP8_ENTRY_BYTES),
                    dtype=torch.uint8,
                    device=self.device,
                )

                # Fill pools with exact integer/power-of-two markers (0-loss under UE8M0).
                cmp_markers: list[torch.Tensor] = []
                swa_markers: list[torch.Tensor] = []
                for b, (sp, S) in enumerate(zip(prefix_list, input_list)):
                    nb = (sp + S) // ratio
                    cmp_vals = (
                        (torch.arange(1, nb + 1, dtype=torch.bfloat16, device=self.device) + b * 100.0)
                        .unsqueeze(1)
                        .expand(nb, 512)
                        .contiguous()
                    )
                    cmp_markers.append(cmp_vals)
                    if nb > 0:
                        cmp_slots = torch.tensor(
                            [int(bt_cmp[b, i // eb_cmp].item()) * eb_cmp + (i % eb_cmp) for i in range(nb)],
                            dtype=torch.int64,
                            device=self.device,
                        )
                        quantize_and_insert_k_cache(cmp_vals, cmp_pool, cmp_slots)

                    swa_vals = (
                        (torch.arange(1, sp + 1, dtype=torch.bfloat16, device=self.device) + b * 1000.0 + 500.0)
                        .unsqueeze(1)
                        .expand(sp, 512)
                        .contiguous()
                    )
                    swa_markers.append(swa_vals)
                    if sp > 0:
                        swa_slots = torch.tensor(
                            [int(bt_swa[b, i // eb_swa].item()) * eb_swa + (i % eb_swa) for i in range(sp)],
                            dtype=torch.int64,
                            device=self.device,
                        )
                        quantize_and_insert_k_cache(swa_vals, swa_pool, swa_slots)

                layer = self._make_exec_attention(
                    compress_ratio=ratio,
                    win=win,
                    eb_cmp=eb_cmp,
                    eb_swa=eb_swa,
                    bt_cmp=bt_cmp,
                    bt_swa=bt_swa,
                    cmp_pool=cmp_pool,
                    swa_pool=swa_pool,
                )
                positions, req_ids, cu_seqlens = _flat_positions(
                    prefix_list, input_list, self.device
                )
                pl = torch.tensor(prefix_list, dtype=torch.int32, device=self.device)
                il = torch.tensor(input_list, dtype=torch.int32, device=self.device)
                wm = layer._build_workspace_meta(
                    seqlen=sum(input_list),
                    sp_int=0,
                    device=self.device,
                    with_dense_cmp_topk=(ratio == 128),
                    use_varlen=True,
                    batch_size=B,
                    cu_seqlens=cu_seqlens,
                    input_lengths=il,
                    prefix_lengths=pl,
                    sp_per_req=pl.long(),
                    position_ids=positions,
                    req_id_per_token=req_ids,
                    max_seqlen_q=max(input_list),
                )
                self.assertIsNotNone(wm)
                self.assertLess(wm.total_ws_rows, B * wm.M)

                T = sum(input_list)
                kv_full = (
                    (torch.arange(1, T + 1, dtype=torch.bfloat16, device=self.device) + 2000.0)
                    .unsqueeze(1)
                    .expand(T, 512)
                    .contiguous()
                )
                captured: dict[str, torch.Tensor] = {}

                def _fake_projected(
                    *,
                    q,
                    kv,
                    indices,
                    topk_length,
                    freqs_cis,
                    prefill_workspace,
                    profile_name,
                    out,
                ):
                    del q, freqs_cis, prefill_workspace, profile_name
                    captured["kv"] = kv.clone()
                    captured["indices"] = indices.clone()
                    captured["topk_length"] = topk_length.clone()
                    return out

                layer._flash_mla_sparse_fwd_chunked_projected = _fake_projected  # type: ignore[assignment]
                common = PrefillMeta(
                    seqlen=T,
                    seqlen_full=T,
                    rd=0,
                    device=self.device,
                    cp_ctx=None,
                    cp_on=False,
                    freqs_cis=torch.zeros(T, 2, device=self.device),
                    topk_idxs=torch.zeros(T, 0, dtype=torch.int32, device=self.device),
                    sp_int=0,
                    any_cont=True,
                    row_seqlens_full=il.long(),
                    use_varlen=True,
                    sp_per_req=pl.long(),
                    cu_seqlens=cu_seqlens,
                    batch_size=B,
                    input_lengths=il,
                    prefix_lengths=pl,
                    position_ids=positions,
                    req_id_per_token=req_ids,
                    max_seqlen_q=max(input_list),
                )
                qkv = PrefillQKV(
                    qr=torch.zeros(T, 512, dtype=torch.bfloat16, device=self.device),
                    q=torch.zeros(T, 1, 512, dtype=torch.bfloat16, device=self.device),
                    kv_full=kv_full.clone(),
                )
                if ratio == 4:
                    cmp_topk_runtime = torch.zeros((T, 2), dtype=torch.int32, device=self.device)
                else:
                    cmp_topk_runtime = None

                with mock.patch(
                    "rtp_llm.models_py.modules.dsv4.fp8.attention.dispose_tensor",
                    lambda t: None,
                ):
                    layer._attn_via_workspace(qkv, common, wm, cmp_topk_runtime)

                kv = captured["kv"]
                self.assertEqual(tuple(kv.shape), (wm.total_ws_rows, 1, 512))
                cu_cmp = wm.cu_cmp_offsets.tolist()
                cu_swa = wm.cu_swa_offsets.tolist()
                tok_off = 0
                for b, (sp, S) in enumerate(zip(prefix_list, input_list)):
                    nb = (sp + S) // ratio
                    pb = min(sp, win - 1)
                    if nb > 0:
                        self.assertTrue(
                            torch.allclose(
                                kv[cu_cmp[b] : cu_cmp[b] + nb, 0, :].float(),
                                cmp_markers[b].float(),
                                rtol=0.1,
                                atol=1.0,
                            )
                        )
                    if pb > 0:
                        self.assertTrue(
                            torch.allclose(
                                kv[cu_swa[b] : cu_swa[b] + pb, 0, :].float(),
                                swa_markers[b][-pb:].float(),
                                rtol=0.1,
                                atol=1.0,
                            )
                        )
                    self.assertTrue(
                        torch.equal(
                            kv[cu_swa[b] + pb : cu_swa[b] + pb + S, 0, :],
                            kv_full[tok_off : tok_off + S],
                        )
                    )
                    tok_off += S

                # Spot-check HCA combine cache key includes total_ws_rows.
                if ratio == 128:
                    layer._attn_via_workspace(
                        PrefillQKV(
                            qr=torch.zeros(T, 512, dtype=torch.bfloat16, device=self.device),
                            q=torch.zeros(T, 1, 512, dtype=torch.bfloat16, device=self.device),
                            kv_full=kv_full.clone(),
                        ),
                        common,
                        wm,
                        None,
                    )

    def test_cp_gt1_sharded_reader_bytesliced_and_raw_q_merge_ragged(self) -> None:
        from types import SimpleNamespace

        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.modules.dsv4 import cp as cp_mod
        from rtp_llm.models_py.modules.dsv4.fp8._pool_reader import (
            CPShardConfig,
            CPShardedPoolReader,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
            quantize_and_insert_k_cache_cp_byte_sliced,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta, PrefillQKV

        win = 8
        ratio = 4
        eb = 8
        cp_size = 2
        B = 2
        prefix_list = [4, 20]
        input_list = [4, 4]  # full seq_len_full=8, asymmetric N=[2, 6], P=[4, 7]
        pl = torch.tensor(prefix_list, dtype=torch.int32, device=self.device)
        il_full = torch.tensor(input_list, dtype=torch.int32, device=self.device)
        cu_full = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        positions_full, req_full, _ = _flat_positions(
            prefix_list, input_list, self.device
        )
        # Rank-0 local chunk owns half the tokens (2 + 2 = 4 tokens).
        local_il = torch.tensor([2, 2], dtype=torch.int32, device=self.device)
        local_cu = torch.tensor([0, 2, 4], dtype=torch.int32, device=self.device)
        local_pos = torch.tensor([4, 5, 20, 21], dtype=torch.int64, device=self.device)
        local_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)

        bt = _make_block_table(B, 4, self.device).to(torch.int32)
        num_blocks = int(bt.max().item()) + 1
        cmp_pool = torch.zeros(
            (num_blocks, eb, _SWA_FP8_ENTRY_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        swa_pool_3d = torch.zeros(
            (num_blocks, eb, _SWA_FP8_ENTRY_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        full_stride_bytes = eb * _SWA_FP8_ENTRY_BYTES + 256
        local_slice_bytes = full_stride_bytes // cp_size
        swa_raw_ranks = [
            torch.zeros((num_blocks, local_slice_bytes), dtype=torch.uint8, device=self.device)
            for _ in range(cp_size)
        ]

        # Populate pools with known values.
        swa_all_slots = []
        swa_all_vals = []
        cmp_vals_list = []
        for b, (sp, S) in enumerate(zip(prefix_list, input_list)):
            nb = (sp + S) // ratio
            cv = (
                (torch.arange(1, nb + 1, dtype=torch.bfloat16, device=self.device) + b * 50.0)
                .unsqueeze(1)
                .expand(nb, 512)
                .contiguous()
            )
            cmp_vals_list.append(cv)
            cs = torch.tensor(
                [int(bt[b, i // eb].item()) * eb + (i % eb) for i in range(nb)],
                dtype=torch.int64,
                device=self.device,
            )
            quantize_and_insert_k_cache(cv, cmp_pool, cs)
            sv = (
                (torch.arange(1, sp + 1, dtype=torch.bfloat16, device=self.device) + b * 200.0 + 100.0)
                .unsqueeze(1)
                .expand(sp, 512)
                .contiguous()
            )
            ss = torch.tensor(
                [int(bt[b, i // eb].item()) * eb + (i % eb) for i in range(sp)],
                dtype=torch.int64,
                device=self.device,
            )
            quantize_and_insert_k_cache(sv, swa_pool_3d, ss)
            swa_all_slots.append(ss)
            swa_all_vals.append(sv)
        all_ss = torch.cat(swa_all_slots)
        all_sv = torch.cat(swa_all_vals)
        w_comp = build_cp_byte_sliced_slot_compaction(
            all_ss,
            full_entries_per_block=eb,
            num_blocks=num_blocks,
            validation_site="test.cp_e2e.write",
            negative_mode="skip_minus_one",
        )
        for r in range(cp_size):
            quantize_and_insert_k_cache_cp_byte_sliced(
                all_sv,
                swa_raw_ranks[r],
                all_ss,
                full_entries_per_block=eb,
                cp_rank=r,
                cp_size=cp_size,
                compaction=w_comp,
            )

        cp_ctx = SimpleNamespace(
            cp_size=cp_size,
            cp_rank=0,
            kv_cache_sharded=False,
            input_lengths_global=il_full,
            input_lengths_global_host=tuple(input_list),
            prefix_lengths=pl,
            prefix_lengths_host=tuple(prefix_list),
            cu_seqlens_global=cu_full,
            seq_len_full=sum(input_list),
            prefix_length=0,
            global_positions=local_pos,
            req_id_per_token=local_req,
            local_token_indices=torch.tensor([0, 1, 4, 5], dtype=torch.long, device=self.device),
            gather_restore_indices=torch.arange(sum(input_list), dtype=torch.int64, device=self.device),
        )

        # Case A: CP>1 + CP byte-sliced SWA + CPShardedPoolReader in _attn_via_workspace.
        layer_cp = self._make_exec_attention(
            compress_ratio=ratio,
            win=win,
            eb_cmp=eb,
            eb_swa=eb,
            bt_cmp=bt,
            bt_swa=bt,
            cmp_pool=cmp_pool,
            swa_pool=None,
            cp_ctx=cp_ctx,
            swa_raw_u8=swa_raw_ranks[0],
        )
        wm_cp = layer_cp._build_workspace_meta(
            seqlen=4,
            sp_int=0,
            device=self.device,
            with_dense_cmp_topk=True,
            use_varlen=True,
            batch_size=B,
            cu_seqlens=local_cu,
            input_lengths=local_il,
            prefix_lengths=pl,
            sp_per_req=pl.long(),
            position_ids=local_pos,
            req_id_per_token=local_req,
            max_seqlen_q=2,
        )
        self.assertIsNotNone(wm_cp)
        # Swap in a mocked CPShardedPoolReader on wm_cp to exercise CPShardedPoolReader 2D path end-to-end.
        per_req_cmp = wm_cp.cmp_seq_lens.long()
        total_local_cmp = int(
            cp_mod.cp_padded_local_kv_lens(per_req_cmp, cp_size, eb).sum().item()
        )
        restore_idx = cp_mod.build_kv_allgather_restore_indices(
            per_req_cmp,
            cp_size,
            eb,
            self.device,
            total_local_kv=total_local_cmp,
        )
        sharded_reader = CPShardedPoolReader(
            CPShardConfig(
                cp_ctx=cp_ctx,
                per_req_total_kv_lens=per_req_cmp,
                restore_indices=restore_idx,
                block_size=eb,
                total_local_kv=total_local_cmp,
                seq_lens_total=int(per_req_cmp.sum().item()),
            )
        )
        wm_cp = wm_cp._replace(cmp_reader=sharded_reader)
        kv_full = (
            (torch.arange(1, 9, dtype=torch.bfloat16, device=self.device) + 900.0)
            .unsqueeze(1)
            .expand(8, 512)
            .contiguous()
        )
        captured_cp: dict[str, torch.Tensor] = {}

        def _fake_proj_cp(*, q, kv, indices, topk_length, freqs_cis, prefill_workspace, profile_name, out):
            del q, freqs_cis, prefill_workspace, profile_name
            captured_cp["kv"] = kv.clone()
            captured_cp["indices"] = indices.clone()
            captured_cp["topk_length"] = topk_length.clone()
            return out

        layer_cp._flash_mla_sparse_fwd_chunked_projected = _fake_proj_cp  # type: ignore[assignment]
        common_cp = PrefillMeta(
            seqlen=4,
            seqlen_full=8,
            rd=0,
            device=self.device,
            cp_ctx=cp_ctx,
            cp_on=True,
            freqs_cis=torch.zeros(4, 2, device=self.device),
            topk_idxs=torch.zeros(4, 0, dtype=torch.int32, device=self.device),
            sp_int=0,
            any_cont=True,
            row_seqlens_full=il_full.long(),
            use_varlen=True,
            sp_per_req=pl.long(),
            cu_seqlens=local_cu,
            batch_size=B,
            input_lengths=local_il,
            prefix_lengths=pl,
            position_ids=local_pos,
            req_id_per_token=local_req,
            max_seqlen_q=2,
        )

        # Extract flat [total_cmp, 584] packed rows ([576 data | 8 scale]) from
        # block-major cmp_pool ([num_blocks, eb*576 + eb*8]) matching restore_idx.
        raw_cmp_blk = cmp_pool.view(num_blocks, -1)
        packed_rows = []
        for b, (sp, S) in enumerate(zip(prefix_list, input_list)):
            for i in range((sp + S) // ratio):
                blk = int(bt[b, i // eb].item())
                pos = i % eb
                packed_rows.append(
                    torch.cat(
                        [
                            raw_cmp_blk[blk, pos * 576 : (pos + 1) * 576],
                            raw_cmp_blk[
                                blk, eb * 576 + pos * 8 : eb * 576 + (pos + 1) * 8
                            ],
                        ],
                        dim=0,
                    )
                )
        expected_packed_cmp = torch.stack(packed_rows, dim=0)
        rank_major_buf = torch.zeros(
            (cp_size * total_local_cmp, _SWA_FP8_ENTRY_BYTES),
            dtype=expected_packed_cmp.dtype,
            device=self.device,
        )
        rank_major_buf.index_copy_(0, restore_idx, expected_packed_cmp)

        def fake_reader_ag(tensor, group=None):
            del tensor, group
            return rank_major_buf

        def fake_swa_ag(tensor, group=None):
            del tensor, group
            comp_u = wm_cp.swa_cache_compaction.unique_blocks
            return torch.cat([raw.index_select(0, comp_u) for raw in swa_raw_ranks], dim=0)

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8._pool_reader.all_gather",
            side_effect=fake_reader_ag,
        ), mock.patch.object(
            sharded_reader,
            "_build_local_flat",
            return_value=torch.empty(0, dtype=torch.uint8, device=self.device),
        ), mock.patch(
            "rtp_llm.models_py.distributed.collective_torch.all_gather",
            side_effect=fake_swa_ag,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.attention.dispose_tensor",
            lambda t: None,
        ):
            layer_cp._attn_via_workspace(
                PrefillQKV(
                    qr=torch.zeros(4, 512, dtype=torch.bfloat16, device=self.device),
                    q=torch.zeros(4, 1, 512, dtype=torch.bfloat16, device=self.device),
                    kv_full=kv_full.clone(),
                ),
                common_cp,
                wm_cp,
                None,
            )

        self.assertEqual(tuple(captured_cp["kv"].shape), (wm_cp.total_ws_rows, 1, 512))
        cu_cmp = wm_cp.cu_cmp_offsets.tolist()
        cu_swa = wm_cp.cu_swa_offsets.tolist()
        # Req 0: N=2, P=4, S=4
        self.assertTrue(
            torch.equal(captured_cp["kv"][cu_swa[0] + 4 : cu_swa[0] + 8, 0, :], kv_full[:4])
        )
        # Req 1: N=6, P=7, S=4
        self.assertTrue(
            torch.equal(captured_cp["kv"][cu_swa[1] + 7 : cu_swa[1] + 11, 0, :], kv_full[4:])
        )
        torch.testing.assert_close(
            captured_cp["kv"][cu_cmp[0] : cu_cmp[0] + 2, 0, :].float(),
            cmp_vals_list[0].float(),
            rtol=0.1,
            atol=1.0,
        )
        torch.testing.assert_close(
            captured_cp["kv"][cu_cmp[1] : cu_cmp[1] + 6, 0, :].float(),
            cmp_vals_list[1].float(),
            rtol=0.1,
            atol=1.0,
        )

        # Case B: CP>1 raw-Q merge ragged 2D local workspace with B=2.
        import sys
        from types import SimpleNamespace as _NS
        from contextlib import ExitStack

        raw_captured: dict[str, torch.Tensor] = {}

        def _fake_flash_raw(*, q, kv, indices, sm_scale, attn_sink, topk_length):
            del sm_scale, attn_sink
            raw_captured["kv"] = kv.clone()
            raw_captured["indices"] = indices.clone()
            raw_captured["topk_length"] = topk_length.clone()
            return (
                torch.zeros((q.shape[0], 1, 512), dtype=torch.bfloat16, device=self.device),
                None,
                torch.zeros((q.shape[0], 1), dtype=torch.float32, device=self.device),
            )

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    sys.modules,
                    {
                        "flash_mla": _NS(flash_mla_sparse_fwd=_fake_flash_raw),
                        "flash_mla_legacy": _NS(flash_mla_sparse_fwd=_fake_flash_raw),
                    },
                )
            )
            stack.enter_context(
                mock.patch(
                    "rtp_llm.models_py.modules.dsv4.fp8.attention.cp_all_gather_full_varlen",
                    side_effect=lambda t, c: torch.zeros(
                        (8, t.shape[1], t.shape[2]), dtype=t.dtype, device=t.device
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    collective_torch,
                    "all_gather",
                    side_effect=lambda t, group=None: t.repeat(cp_size, *([1] * (t.dim() - 1))),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    layer_cp,
                    "_cp_local_full_row_indices",
                    return_value=torch.tensor([0, 1, 4, 5], dtype=torch.long, device=self.device),
                )
            )
            out_raw = layer_cp._attn_via_workspace_cp_raw_q_merge(
                qkv=PrefillQKV(
                    qr=torch.zeros(4, 512, dtype=torch.bfloat16, device=self.device),
                    q=torch.zeros(4, 1, 512, dtype=torch.bfloat16, device=self.device),
                    kv_full=kv_full.clone(),
                ),
                common=common_cp,
                workspace_meta=wm_cp,
                cmp_topk_runtime=None,
                cmp_pool_3d=cmp_pool,
                swa_pool_3d=swa_pool_3d,
            )
        self.assertEqual(tuple(out_raw.shape), (1, 4, 1, 512))
        # local_cmp_lens for cp_rank=0 with eb=8, cmp_seq=[2, 6]: both blocks 0 are owned by rank0 => local_cmp=[2, 6]
        # swa_gather=[8, 11] => expected ragged local rows = 2+8 + 6+11 = 27 (< 2*(6+11)=34).
        self.assertEqual(tuple(raw_captured["kv"].shape), (27, 1, 512))
        for t_idx in range(8):
            ln = int(raw_captured["topk_length"][t_idx].item())
            row_idx = raw_captured["indices"][t_idx, 0, :ln]
            self.assertTrue(bool(((row_idx >= 0) & (row_idx < 27)).all()))


if __name__ == "__main__":
    unittest.main()
