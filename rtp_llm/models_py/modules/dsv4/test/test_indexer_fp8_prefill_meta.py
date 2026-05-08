"""UT: ``IndexerFP8.prepare`` — per-call FP8 prefill metadata builder.

Validates the meta builder lifted out of ``IndexerFP8.forward``'s hot
path. Mirrors the stub-based pattern used by
``test_indexer_fmha_meta.py`` / ``test_csa_hca_prefill_meta.py`` in the
``refactor/dsv4_attention_prefill_split_back`` branch — exercises the
method without spinning up a real ``IndexerFP8`` (which needs FP8
weights, scales, the nested compressor, etc.).

Math under test:

  end_pos          = sp_int + seqlen
  T                = end_pos // ratio
  M                = bsz * seqlen
  positions_d      = [sp_int .. sp_int + seqlen - 1]      int32
  ke[r]            = clamp((positions_d[r] + 1) // ratio, max=T)  int32
  ks               = zeros(M)                              int32
  cu_kv_seqlens    = [0, T]                                int32
  block_table_i32  = self._kv_block_table[:bsz].int32      contig

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_indexer_fp8_prefill_meta
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
    IndexerFP8,
    _IndexerFP8PrefillMeta,
)


class _StubIndexerFP8:
    """Stand-in exposing only the attrs ``IndexerFP8.prepare`` reads.

    Lets us drive the builder without constructing a real
    ``IndexerFP8`` (which needs FP8 weights + DeepGEMM linear etc.)."""

    def __init__(
        self,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
    ) -> None:
        self.compress_ratio = compress_ratio
        self.freqs_cis = freqs_cis
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb


def _call(
    stub: _StubIndexerFP8,
    bsz: int,
    seqlen: int,
    sp_int: int,
    device: torch.device,
) -> _IndexerFP8PrefillMeta:
    # Bound-method dispatch via the unbound function — works on the stub
    # because ``prepare`` only touches the four attrs the stub exposes.
    # Pass the pool kwargs explicitly: the broadcast-meta refactor made
    # them required for non-warmup paths (was previously read off self).
    return IndexerFP8.prepare(
        stub,
        bsz,
        seqlen,
        sp_int,
        device,
        kv_block_table=stub._kv_block_table,
        kv_eb=stub._kv_eb,
    )


class IndexerFP8PrepareTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")
        self.ratio = 4
        self.max_pos = 4096
        self.rope_half = 32  # rope_head_dim // 2
        # Real ``self.freqs_cis`` is complex64 ``[max_seq_len, rope_half]``
        # but ``prepare`` only slices it; any 2D tensor works.
        self.freqs_cis = torch.arange(
            self.max_pos * self.rope_half, dtype=torch.float32, device=self.device
        ).view(self.max_pos, self.rope_half)
        self.kv_eb = 64
        # block_table is [B_max, max_blocks] — caller may pass int64 (we
        # cast inside ``prepare`` to int32 contig).
        self.bt = torch.arange(1, 17, dtype=torch.int64, device=self.device).view(1, 16)
        self.stub = _StubIndexerFP8(self.ratio, self.freqs_cis, self.bt, self.kv_eb)

    # ------------------------------------------------------------------
    # Geometry scalars
    # ------------------------------------------------------------------
    def test_cold_prefill_geometry(self) -> None:
        """sp=0, S=512, ratio=4 → M=512, T=128, fresh=True."""
        meta = _call(self.stub, bsz=1, seqlen=512, sp_int=0, device=self.device)
        self.assertIsInstance(meta, _IndexerFP8PrefillMeta)
        self.assertEqual(meta.bsz, 1)
        self.assertEqual(meta.seqlen, 512)
        self.assertEqual(meta.M, 512)
        self.assertEqual(meta.sp_int, 0)
        self.assertEqual(meta.end_pos, 512)
        self.assertTrue(meta.is_fresh_prefill)
        self.assertEqual(meta.T, 128)

    def test_continuation_prefill_geometry(self) -> None:
        """sp=200, S=64, ratio=4 → end_pos=264, T=66, fresh=False."""
        meta = _call(self.stub, bsz=1, seqlen=64, sp_int=200, device=self.device)
        self.assertFalse(meta.is_fresh_prefill)
        self.assertEqual(meta.sp_int, 200)
        self.assertEqual(meta.end_pos, 264)
        self.assertEqual(meta.T, 264 // 4)
        self.assertEqual(meta.M, 64)

    def test_cold_start_below_ratio_T_zero(self) -> None:
        """sp=0, S<ratio → T=0; cu_kv_seqlens collapses to [0, 0]."""
        meta = _call(self.stub, bsz=1, seqlen=2, sp_int=0, device=self.device)
        self.assertEqual(meta.T, 0)
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 0])

    def test_continuation_partial_emit(self) -> None:
        """sp=6, S=4 (total=10), ratio=4 → T=2; positions=[6,7,8,9]
        → ke=(7//4=1, 8//4=2, 9//4=2, 10//4=2)."""
        meta = _call(self.stub, bsz=1, seqlen=4, sp_int=6, device=self.device)
        self.assertEqual(meta.T, 2)
        self.assertEqual(meta.M, 4)
        self.assertEqual(meta.positions_d.tolist(), [6, 7, 8, 9])
        self.assertEqual(meta.ke.tolist(), [1, 2, 2, 2])

    def test_short_chunk_at_boundary_no_new_compressed(self) -> None:
        """sp=4, S=2, ratio=4 → T=1; positions=[4,5] → ke=[1, 1].
        Both rows see exactly the K compressed at sp boundary."""
        meta = _call(self.stub, bsz=1, seqlen=2, sp_int=4, device=self.device)
        self.assertEqual(meta.T, 1)
        self.assertEqual(meta.positions_d.tolist(), [4, 5])
        self.assertEqual(meta.ke.tolist(), [1, 1])
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 1])

    def test_long_continuation_offsets_positions(self) -> None:
        """sp=512, S=256, ratio=4 → T=192; positions=[512..767];
        ke[0]=(513//4)=128, ke[-1]=(768//4)=192."""
        meta = _call(self.stub, bsz=1, seqlen=256, sp_int=512, device=self.device)
        self.assertEqual(meta.T, 192)
        self.assertEqual(meta.positions_d[0].item(), 512)
        self.assertEqual(meta.positions_d[-1].item(), 767)
        self.assertEqual(meta.ke[0].item(), 128)
        self.assertEqual(meta.ke[-1].item(), 192)
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 192])

    # ------------------------------------------------------------------
    # Q-side: positions_d / freqs_cis_slice
    # ------------------------------------------------------------------
    def test_positions_d_global_range(self) -> None:
        """positions_d = arange(sp, sp+S) int32."""
        meta = _call(self.stub, bsz=1, seqlen=8, sp_int=4, device=self.device)
        self.assertEqual(meta.positions_d.shape, (8,))
        self.assertEqual(meta.positions_d.dtype, torch.int32)
        self.assertEqual(meta.positions_d.tolist(), list(range(4, 12)))

    def test_freqs_cis_slice_view(self) -> None:
        """freqs_cis_slice = self.freqs_cis[sp:sp+S] (view, byte-equal)."""
        meta = _call(self.stub, bsz=1, seqlen=8, sp_int=4, device=self.device)
        self.assertEqual(meta.freqs_cis_slice.shape, (8, self.rope_half))
        self.assertTrue(torch.equal(meta.freqs_cis_slice, self.freqs_cis[4:12]))

    # ------------------------------------------------------------------
    # Per-row visible-K window: ks / ke
    # ------------------------------------------------------------------
    def test_ks_all_zero(self) -> None:
        """ks ≡ 0 — fresh prefill never has a non-zero per-row K start."""
        meta = _call(self.stub, bsz=1, seqlen=32, sp_int=0, device=self.device)
        self.assertEqual(meta.ks.shape, (32,))
        self.assertEqual(meta.ks.dtype, torch.int32)
        self.assertTrue((meta.ks == 0).all().item())

    def test_ke_causal_clamped_to_T_fresh_prefill(self) -> None:
        """sp=0, S=16, ratio=4 → T=4. ke[r] = (r+1)//4 clamped to 4."""
        meta = _call(self.stub, bsz=1, seqlen=16, sp_int=0, device=self.device)
        self.assertEqual(meta.ke.dtype, torch.int32)
        positions = torch.arange(16, dtype=torch.int64)
        expected_ke = ((positions + 1) // 4).clamp(max=4).to(torch.int32)
        self.assertTrue(torch.equal(meta.ke.cpu(), expected_ke))

    def test_ke_continuation_prefill(self) -> None:
        """sp=8, S=8, ratio=4 → T=4. positions=[8..15]; ke = (pos+1)//4
        all >= 2 (so > T at the tail) → must clamp to T=4."""
        meta = _call(self.stub, bsz=1, seqlen=8, sp_int=8, device=self.device)
        self.assertEqual(meta.T, 4)
        positions = torch.arange(8, 16, dtype=torch.int64)
        expected_ke = ((positions + 1) // 4).clamp(max=4).to(torch.int32)
        self.assertTrue(torch.equal(meta.ke.cpu(), expected_ke))

    # ------------------------------------------------------------------
    # K-side: block_table / cu_kv_seqlens
    # ------------------------------------------------------------------
    def test_block_table_int32_contig_sliced_to_bsz(self) -> None:
        meta = _call(self.stub, bsz=1, seqlen=8, sp_int=0, device=self.device)
        self.assertEqual(meta.block_table_i32.shape, (1, 16))
        self.assertEqual(meta.block_table_i32.dtype, torch.int32)
        self.assertTrue(meta.block_table_i32.is_contiguous())
        # Values match input (cast to int32).
        self.assertTrue(torch.equal(meta.block_table_i32, self.bt[:1].to(torch.int32)))

    def test_cu_kv_seqlens_compressed(self) -> None:
        """sp=12, S=20, ratio=4 → end_pos=32, T=8 → [0, 8]."""
        meta = _call(self.stub, bsz=1, seqlen=20, sp_int=12, device=self.device)
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 8])
        self.assertEqual(meta.cu_kv_seqlens.dtype, torch.int32)

    def test_warmup_no_pool_emits_empty_block_table(self) -> None:
        """No bound block_table → empty (bsz, 0) int32 placeholder so
        the gather kernel can still be called (will zero-iterate)."""
        stub_no_bt = _StubIndexerFP8(self.ratio, self.freqs_cis, None, 0)
        meta = _call(stub_no_bt, bsz=1, seqlen=8, sp_int=0, device=self.device)
        self.assertEqual(meta.block_table_i32.shape, (1, 0))
        self.assertEqual(meta.block_table_i32.dtype, torch.int32)

    # ------------------------------------------------------------------
    # Sanity: missing freqs_cis raises
    # ------------------------------------------------------------------
    def test_missing_freqs_cis_raises(self) -> None:
        bad = _StubIndexerFP8(self.ratio, None, self.bt, self.kv_eb)
        with self.assertRaises(AssertionError):
            _call(bad, bsz=1, seqlen=8, sp_int=0, device=self.device)


# ---------------------------------------------------------------------------
# Phase-3a varlen path: B>1 with mixed-prefix requests.
# Asserts every per-request scalar from the legacy B==1 builder fans out
# correctly to per-request [B] / [B+1] / [T_total] tensors.
# ---------------------------------------------------------------------------
class IndexerFP8PrepareVarlenTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")
        self.ratio = 4
        self.max_pos = 4096
        self.rope_half = 32
        self.freqs_cis = torch.arange(
            self.max_pos * self.rope_half, dtype=torch.float32, device=self.device
        ).view(self.max_pos, self.rope_half)
        self.kv_eb = 64
        # B=2 block table — distinct per-request rows.
        self.bt = torch.arange(1, 33, dtype=torch.int64, device=self.device).view(2, 16)
        self.stub = _StubIndexerFP8(self.ratio, self.freqs_cis, self.bt, self.kv_eb)

    def _make_batched_kwargs(
        self,
        prefix_lengths: list[int],
        input_lengths: list[int],
    ) -> dict:
        """Build the canonical varlen kwargs the upper-layer broadcast
        builder hands down. Mirrors prefill/forward.py derivation."""
        device = self.device
        B = len(input_lengths)
        il = torch.tensor(input_lengths, dtype=torch.int32, device=device)
        pl = torch.tensor(prefix_lengths, dtype=torch.int32, device=device)
        cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(il, dim=0).to(torch.int32)
        T_total = int(cu[-1].item())
        positions = torch.cat(
            [
                torch.arange(p, p + L, dtype=torch.int64, device=device)
                for L, p in zip(input_lengths, prefix_lengths)
            ],
            dim=0,
        )
        req_id = (
            torch.searchsorted(
                cu.to(torch.int64),
                torch.arange(T_total, dtype=torch.int64, device=device),
                right=True,
            )
            .sub_(1)
            .to(torch.int32)
            .contiguous()
        )
        return dict(
            batch_size=B,
            cu_seqlens=cu,
            input_lengths=il,
            prefix_lengths=pl,
            position_ids=positions,
            req_id_per_token=req_id,
            max_seqlen_q=int(il.max().item()),
        )

    def test_b1_varlen_kwargs_bit_equal_to_legacy(self) -> None:
        """B==1 with full varlen kwargs must produce bit-equal meta to
        the scalar-only call. Guards against accidental divergence in
        the new branch."""
        kw = self._make_batched_kwargs(prefix_lengths=[12], input_lengths=[20])
        legacy = IndexerFP8.prepare(
            self.stub,
            1,
            20,
            12,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
        )
        new = IndexerFP8.prepare(
            self.stub,
            1,
            20,
            12,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
            **kw,
        )
        # Scalar fields
        self.assertEqual(legacy.M, new.M)
        self.assertEqual(legacy.T, new.T)
        self.assertEqual(legacy.sp_int, new.sp_int)
        self.assertEqual(legacy.end_pos, new.end_pos)
        self.assertEqual(legacy.is_fresh_prefill, new.is_fresh_prefill)
        # Tensor fields — bit-equal at B==1
        self.assertTrue(torch.equal(legacy.positions_d, new.positions_d))
        self.assertTrue(torch.equal(legacy.ke, new.ke))
        self.assertTrue(torch.equal(legacy.ks, new.ks))
        self.assertTrue(torch.equal(legacy.cu_kv_seqlens, new.cu_kv_seqlens))
        self.assertTrue(torch.equal(legacy.block_table_i32, new.block_table_i32))
        self.assertTrue(torch.equal(legacy.freqs_cis_slice, new.freqs_cis_slice))

    def test_b2_mixed_sp_geometry(self) -> None:
        """B=2: req0 (sp=0, S=16) + req1 (sp=8, S=12), ratio=4.
        T_per_req = [16//4, 20//4] = [4, 5] → cu_kv_seqlens = [0,4,9].
        T_total (M) = 28. positions = [0..15, 8..19]."""
        kw = self._make_batched_kwargs(
            prefix_lengths=[0, 8],
            input_lengths=[16, 12],
        )
        meta = IndexerFP8.prepare(
            self.stub,
            kw["batch_size"],
            int(kw["max_seqlen_q"]),
            0,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
            **kw,
        )
        self.assertEqual(meta.M, 28)
        self.assertEqual(meta.T, 9)  # 4 + 5
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 4, 9])
        self.assertEqual(meta.positions_d.shape, (28,))
        self.assertEqual(meta.positions_d[0].item(), 0)
        self.assertEqual(meta.positions_d[15].item(), 15)
        self.assertEqual(meta.positions_d[16].item(), 8)
        self.assertEqual(meta.positions_d[27].item(), 19)

    def test_b2_ke_clamps_per_request_T_b(self) -> None:
        """ke[t] must clamp to that token's request-local T_b, NOT the
        global T_total. Token in req0 with pos=15 → ke=(16//4)=4
        (clamped to req0's T_b=4). Token in req1 with pos=19 → ke=(20//4)=5
        (clamped to req1's T_b=5). If we accidentally clamped to global
        T=9, both would just take their // value with no clamping."""
        kw = self._make_batched_kwargs(
            prefix_lengths=[0, 8],
            input_lengths=[16, 12],
        )
        meta = IndexerFP8.prepare(
            self.stub,
            kw["batch_size"],
            int(kw["max_seqlen_q"]),
            0,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
            **kw,
        )
        ke_cpu = meta.ke.cpu()
        # Req0 last token: pos=15 → expected_ke = clamp((15+1)//4=4, max=4) = 4
        self.assertEqual(int(ke_cpu[15].item()), 4)
        # Req1 last token: pos=19 → expected_ke = clamp((19+1)//4=5, max=5) = 5
        self.assertEqual(int(ke_cpu[27].item()), 5)
        # Cross-check first req0 token: pos=0 → ke = clamp(1//4=0, max=4) = 0
        self.assertEqual(int(ke_cpu[0].item()), 0)
        # First req1 token: pos=8 → ke = clamp(9//4=2, max=5) = 2
        self.assertEqual(int(ke_cpu[16].item()), 2)

    def test_b2_block_table_sliced_to_B(self) -> None:
        kw = self._make_batched_kwargs(
            prefix_lengths=[0, 8],
            input_lengths=[16, 12],
        )
        meta = IndexerFP8.prepare(
            self.stub,
            kw["batch_size"],
            int(kw["max_seqlen_q"]),
            0,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
            **kw,
        )
        self.assertEqual(meta.block_table_i32.shape, (2, 16))
        self.assertEqual(meta.block_table_i32.dtype, torch.int32)
        self.assertTrue(torch.equal(meta.block_table_i32, self.bt[:2].to(torch.int32)))

    def test_b2_freqs_cis_per_token_gather(self) -> None:
        """freqs_cis_slice[t] must equal self.freqs_cis[position_ids[t]] —
        per-token gather, not a contiguous slice."""
        kw = self._make_batched_kwargs(
            prefix_lengths=[0, 8],
            input_lengths=[16, 12],
        )
        meta = IndexerFP8.prepare(
            self.stub,
            kw["batch_size"],
            int(kw["max_seqlen_q"]),
            0,
            self.device,
            kv_block_table=self.bt,
            kv_eb=self.kv_eb,
            **kw,
        )
        positions = kw["position_ids"]
        expected = self.freqs_cis.index_select(0, positions)
        self.assertTrue(torch.equal(meta.freqs_cis_slice, expected))

    def test_b2_env_flag_disables_varlen_path(self) -> None:
        """DSV4_VARLEN_PREFILL=0 should fall back to the legacy B==1
        scalar path even when B>1 kwargs are present (kill-switch)."""
        import os as _os

        kw = self._make_batched_kwargs(
            prefix_lengths=[0, 8],
            input_lengths=[16, 12],
        )
        prev = _os.environ.get("DSV4_VARLEN_PREFILL")
        _os.environ["DSV4_VARLEN_PREFILL"] = "0"
        try:
            meta = IndexerFP8.prepare(
                self.stub,
                1,
                16,
                0,
                self.device,
                kv_block_table=self.bt,
                kv_eb=self.kv_eb,
                **kw,
            )
        finally:
            if prev is None:
                _os.environ.pop("DSV4_VARLEN_PREFILL", None)
            else:
                _os.environ["DSV4_VARLEN_PREFILL"] = prev
        # Legacy path: M = 1*16, T = (0+16)//4 = 4, cu_kv_seqlens=[0,4]
        self.assertEqual(meta.M, 16)
        self.assertEqual(meta.T, 4)
        self.assertEqual(meta.cu_kv_seqlens.tolist(), [0, 4])


if __name__ == "__main__":
    unittest.main()
