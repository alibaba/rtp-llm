"""UT: DSV4 zero-SWA inverted-triangle TRIM (Stage C compute reduction).

Stage B (``test_zero_swa_write_skip``) pins the *write-skip* guardian; this
file pins the *trim* itself — the per-layer front-trim that actually removes
recompute FLOPs after a prefix hit. It runs entirely on CPU (no GPU kernels),
so the subtle coverage / off-by-one cases an end-to-end smoke can paper over
are pinned here as the user asked:

  * Offset math (:func:`_dsv4_zero_swa_trim_offsets`): the linear inverted
    triangle ``q_start_t = restore_eff - (L-j)*nwin`` (widest at bottom layer
    0), monotone non-decreasing, K span one ``nwin`` window wider than Q
    (``k_start_t = q_start_t - nwin``), active-tail floor, and the no-trim
    short-circuits.
  * Meta slicing (:func:`_slice_meta`): every per-token field is a pure suffix
    slice with NO value rebase; the load-bearing [B]/[B+1] recomputes
    (``CompressorMeta.seq_start_per_req += ks``, ``cu_seq_per_req`` /
    ``WorkspaceMeta.qsl`` shifted) are applied while the K-side widths
    (``M`` / ``N`` / ``swa_seq_lens``) stay full.
  * **Numerical equivalence** (the headline): a faithful sliding-window
    cascade run through the REAL offsets + the REAL write-back loop produces a
    **byte-identical kept tail** vs the uniform forward — and a negative
    control proves the test actually catches an under-covering K span.

The kernel bit-exactness of the real FP8 path is gated by the SM100 smoke
``smoke_v4_flash_pd_*_zeroswa_trim``; this UT gates the host-side index/offset
logic that no smoke can localize.
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    CsaPrefillMeta,
    HcaPrefillMeta,
    PrefillMeta,
    SwaPrefillMeta,
    WorkspaceMeta,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorMeta
from rtp_llm.models_py.modules.dsv4.fp8.indexer import _IndexerFP8PrefillMeta
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    _shift_cu_seqlens,
    _slice_meta,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import (
    _dsv4_zero_swa_trim_offsets,
)


class TrimOffsetsTest(unittest.TestCase):
    """``_dsv4_zero_swa_trim_offsets`` — the inverted-triangle width formula.

    Returns ``(k_start_t, tail)``: the per-layer K-span front-trim counts and
    the token index from which the final hidden must be bit-correct.
    """

    def test_linear_triangle_shape_and_monotonicity(self):
        # restore_eff = 8192, many new tokens -> tail = restore_eff = 8192.
        L, nwin, restore, T, kb = 8, 128, 8192, 20000, 128
        k, tail = _dsv4_zero_swa_trim_offsets(L, nwin, restore, T, kb, "cpu")
        self.assertEqual(len(k), L)
        self.assertEqual(tail, min(restore, T - 2 * kb))
        # monotone non-decreasing (bottom widest / smallest k -> top narrowest)
        self.assertEqual(k, sorted(k))
        # k_start_j = clamp(tail - (L-j)*nwin, 0, T)
        for j in range(L):
            self.assertEqual(
                k[j], max(0, min(tail - (L - j) * nwin, T)), msg=f"layer {j}"
            )
        # the top layer's K span begins exactly one window below the tail
        self.assertEqual(k[-1], max(0, tail - nwin))

    def test_exact_values(self):
        L, nwin, restore, T, kb = 6, 100, 5000, 9000, 128
        k, tail = _dsv4_zero_swa_trim_offsets(L, nwin, restore, T, kb, "cpu")
        self.assertEqual(tail, min(5000, 9000 - 256))  # = 5000
        for j in range(L):
            self.assertEqual(k[j], max(0, min(tail - (L - j) * nwin, T)), msg=j)

    def test_active_tail_covered_when_new_tokens_scarce(self):
        # Fewer new tokens than the active tail: tail must drop below
        # restore_eff to T - t_active so the SWA active tail is regenerated.
        L, nwin, kb = 8, 64, 64
        restore, T = 4000, 4000  # full hit: restore_eff == T, 0 new tokens
        k, tail = _dsv4_zero_swa_trim_offsets(L, nwin, restore, T, kb, "cpu")
        self.assertEqual(tail, T - 2 * kb)  # 3872 — last active-tail kept
        # bottom layer keeps the widest span; top layer narrowest
        self.assertLess(k[0], k[-1])
        self.assertEqual(k[-1], tail - nwin)

    def test_no_trim_short_circuits(self):
        # n_layers <= 1 (MTP draft), empty restore, empty tokens, restore<=nwin
        self.assertEqual(
            _dsv4_zero_swa_trim_offsets(1, 128, 8192, 9000, 128, "cpu"), (None, None)
        )
        self.assertEqual(
            _dsv4_zero_swa_trim_offsets(8, 128, 0, 9000, 128, "cpu"), (None, None)
        )
        self.assertEqual(
            _dsv4_zero_swa_trim_offsets(8, 128, 8192, 0, 128, "cpu"), (None, None)
        )
        # restore window <= nwin: tail - (L-j)*nwin <= 0 for every layer -> no trim.
        self.assertEqual(
            _dsv4_zero_swa_trim_offsets(8, 128, 100, 9000, 128, "cpu"), (None, None)
        )

    def test_short_materialized_region_clamps(self):
        # R3: total_tokens (materialized) < restore_window -> restore_eff = T.
        # Offsets must never index past the materialized end.
        L, nwin, restore, T, kb = 8, 128, 8192, 600, 128
        k, tail = _dsv4_zero_swa_trim_offsets(L, nwin, restore, T, kb, "cpu")
        if k is not None:
            self.assertTrue(all(0 <= x <= T for x in k))
            self.assertTrue(0 <= tail <= T)


class ShiftCuSeqlensTest(unittest.TestCase):
    def test_b1_shift(self):
        cu = torch.tensor([0, 100], dtype=torch.int32)
        out = _shift_cu_seqlens(cu, 30)
        self.assertTrue(torch.equal(out, torch.tensor([0, 70], dtype=torch.int32)))

    def test_clamp_min_zero(self):
        cu = torch.tensor([0, 100], dtype=torch.int64)
        out = _shift_cu_seqlens(cu, 0)
        self.assertTrue(torch.equal(out, cu))

    def test_none_passthrough(self):
        self.assertIsNone(_shift_cu_seqlens(None, 5))


def _make_compressor_meta(T: int, sp0: int) -> CompressorMeta:
    # Per-token values encode their token index so slicing-vs-rebasing is
    # observable; seq_start_per_req is the [B=1] absolute start sp0.
    return CompressorMeta(
        positions=torch.arange(sp0, sp0 + T, dtype=torch.int64),
        b_idx=torch.zeros(T, dtype=torch.int64),
        state_slots=torch.arange(1000, 1000 + T, dtype=torch.int64),
        kv_slots=torch.arange(2000, 2000 + T, dtype=torch.int64),
        token_to_req=torch.zeros(T, dtype=torch.int32),
        is_batched=True,
        seq_start_per_req=torch.tensor([sp0], dtype=torch.int64),  # [B]
        cu_seq_per_req=torch.tensor([0, T], dtype=torch.int64),  # [B+1]
        compressed_lens_per_token=None,
    )


def _make_workspace_meta(T: int) -> WorkspaceMeta:
    return WorkspaceMeta(
        M=64,
        N=16,
        swa_eb=4,
        cmp_eb=4,
        swa_bt_int32=torch.zeros((1, 8), dtype=torch.int32),
        cmp_bt_int32=torch.zeros((1, 8), dtype=torch.int32),
        swa_seq_lens=torch.tensor([900], dtype=torch.int32),
        cmp_seq_lens=torch.tensor([7], dtype=torch.int32),
        swa_gather_lens=torch.tensor([300], dtype=torch.int32),
        swa_cache_seq_lens=torch.tensor([100], dtype=torch.int32),
        swa_cache_gather_lens=torch.tensor([100], dtype=torch.int32),
        qsl=torch.tensor([0, T], dtype=torch.int32),
        dense_cmp_topk=torch.arange(T * 16, dtype=torch.int32).reshape(T, 16),
        new_k_slot_in_flat=torch.arange(3000, 3000 + T, dtype=torch.int64),
        cmp_reader=object(),
        use_cp_raw_q_merge=False,
        swa_cache_slot_mapping=torch.zeros((1, 4), dtype=torch.int64),
        swa_slot_mapping=None,
    )


def _make_indexer_meta(T: int, sp0: int) -> _IndexerFP8PrefillMeta:
    return _IndexerFP8PrefillMeta(
        bsz=1,
        seqlen=T,
        M=T,
        sp_int=sp0,
        end_pos=sp0 + T,
        is_fresh_prefill=False,
        T=42,  # compressed-K count, trim-invariant
        freqs_cis_slice=torch.arange(T * 4, dtype=torch.float32).reshape(T, 4),
        positions_d=torch.arange(sp0, sp0 + T, dtype=torch.int32),
        ks=torch.arange(5000, 5000 + T, dtype=torch.int32),  # per-Q-row K start
        ke=torch.arange(6000, 6000 + T, dtype=torch.int32),  # per-Q-row K end
        block_table_i32=torch.zeros((1, 8), dtype=torch.int32),
        cu_kv_seqlens=torch.tensor([0, 42], dtype=torch.int32),
        cu_kv_per_token=None,  # None on B==1
        compressor_meta=_make_compressor_meta(T, sp0),
    )


def _make_swa_meta(T: int, prefix: int) -> SwaPrefillMeta:
    return SwaPrefillMeta(
        slot_mapping=torch.arange(7000, 7000 + T, dtype=torch.int64),
        query_start_loc=torch.tensor([0, T], dtype=torch.int32),
        combined_seq_lens=torch.tensor([prefix + T], dtype=torch.int32),
        topk_length_kv_full=torch.arange(T, dtype=torch.int32),
        combined_gather_lens=torch.tensor([prefix + T], dtype=torch.int32),
        combined_gather_len_max=prefix + T,
        M=64,
        cache_seq_lens=torch.tensor([prefix], dtype=torch.int32),
        cache_gather_lens=torch.tensor([min(prefix, 127)], dtype=torch.int32),
        prefix_len_max=1,
        combined_indices=torch.arange(T * 3, dtype=torch.int32).reshape(T, 3),
        combined_lens=torch.arange(T, dtype=torch.int32),
        slot_in_flat=torch.arange(8000, 8000 + T, dtype=torch.int64),
        cache_slot_mapping=torch.zeros((1, 4), dtype=torch.int64),
    )


def _make_prefill_meta(T: int, prefix: int, kind: str) -> PrefillMeta:
    """``kind`` in {'swa','csa','hca'}."""
    csa = hca = None
    if kind == "csa":
        csa = CsaPrefillMeta(
            indexer_meta=_make_indexer_meta(T, prefix),
            compressor_meta=_make_compressor_meta(T, prefix),
            workspace_meta=_make_workspace_meta(T),
        )
    elif kind == "hca":
        hca = HcaPrefillMeta(
            compressor_meta=_make_compressor_meta(T, prefix),
            workspace_meta=_make_workspace_meta(T),
        )
    return PrefillMeta(
        seqlen=T,
        seqlen_full=T,
        rd=4,
        device=torch.device("cpu"),
        cp_ctx=None,
        cp_on=False,
        freqs_cis=torch.arange(T * 4, dtype=torch.float32).reshape(T, 4),
        topk_idxs=torch.arange(T * 2, dtype=torch.int32).reshape(T, 2),
        sp_int=prefix,
        any_cont=True,
        row_seqlens_full=torch.tensor([T], dtype=torch.int64),
        use_varlen=True,
        sp_per_req=torch.tensor([prefix], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int64),
        batch_size=1,
        input_lengths=torch.tensor([T], dtype=torch.int32),
        prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
        position_ids=torch.arange(prefix, prefix + T, dtype=torch.int64),
        req_id_per_token=torch.zeros(T, dtype=torch.int32),
        max_seqlen_q=T,
        swa_meta=_make_swa_meta(T, prefix),
        csa_meta=csa,
        hca_meta=hca,
    )


class SliceMetaFieldTest(unittest.TestCase):
    T = 20
    PREFIX = 100
    KS = 5

    def test_identity_when_ks_zero(self):
        meta = _make_prefill_meta(self.T, self.PREFIX, "csa")
        self.assertIs(_slice_meta(meta, 0), meta)

    def test_top_level_scalars_and_per_token(self):
        meta = _make_prefill_meta(self.T, self.PREFIX, "csa")
        out = _slice_meta(meta, self.KS)
        n = self.T - self.KS
        # recomputed scalars
        self.assertEqual(out.seqlen, n)
        self.assertEqual(out.seqlen_full, n)
        self.assertEqual(out.sp_int, self.PREFIX + self.KS)
        # per-token suffix slices (length + value identity, NO rebase)
        self.assertEqual(out.freqs_cis.shape[0], n)
        self.assertTrue(torch.equal(out.freqs_cis, meta.freqs_cis[self.KS :]))
        self.assertTrue(torch.equal(out.position_ids, meta.position_ids[self.KS :]))
        self.assertTrue(
            torch.equal(out.req_id_per_token, meta.req_id_per_token[self.KS :])
        )
        self.assertTrue(torch.equal(out.topk_idxs, meta.topk_idxs[self.KS :]))
        # cumulative / per-request recomputes
        self.assertTrue(
            torch.equal(out.cu_seqlens, torch.tensor([0, n], dtype=torch.int64))
        )
        self.assertEqual(int(out.input_lengths[0]), n)
        self.assertTrue(torch.equal(out.row_seqlens_full, torch.tensor([n])))
        # trim-invariant
        self.assertTrue(torch.equal(out.prefix_lengths, meta.prefix_lengths))
        self.assertEqual(out.batch_size, 1)
        self.assertEqual(out.max_seqlen_q, meta.max_seqlen_q)

    def test_swa_meta_slice_no_rebase(self):
        meta = _make_prefill_meta(self.T, self.PREFIX, "swa")
        out = _slice_meta(meta, self.KS).swa_meta
        ref = meta.swa_meta
        n = self.T - self.KS
        # per-token sliced, values UNCHANGED (slot_in_flat / combined_indices
        # are request-token-0 anchored -> NO rebase)
        self.assertTrue(torch.equal(out.slot_in_flat, ref.slot_in_flat[self.KS :]))
        self.assertTrue(
            torch.equal(out.combined_indices, ref.combined_indices[self.KS :])
        )
        self.assertTrue(torch.equal(out.combined_lens, ref.combined_lens[self.KS :]))
        self.assertTrue(torch.equal(out.slot_mapping, ref.slot_mapping[self.KS :]))
        self.assertEqual(out.combined_indices.shape[0], n)
        # query_start_loc / combined_seq_lens shifted/recomputed
        self.assertTrue(
            torch.equal(out.query_start_loc, torch.tensor([0, n], dtype=torch.int32))
        )
        self.assertEqual(int(out.combined_seq_lens[0]), self.PREFIX + n)
        # K-side widths / workspace stride / prefix tail kept FULL
        self.assertEqual(out.M, ref.M)
        self.assertEqual(out.combined_gather_len_max, ref.combined_gather_len_max)
        self.assertTrue(
            torch.equal(out.combined_gather_lens, ref.combined_gather_lens)
        )
        self.assertTrue(torch.equal(out.cache_seq_lens, ref.cache_seq_lens))
        self.assertTrue(torch.equal(out.cache_slot_mapping, ref.cache_slot_mapping))
        self.assertEqual(out.prefix_len_max, ref.prefix_len_max)

    def test_compressor_meta_seq_start_recompute(self):
        # The load-bearing correction: seq_start_per_req is [B] and RECOMPUTES
        # +ks (NOT sliced); cu_seq_per_req shifts; per-token fields slice.
        meta = _make_prefill_meta(self.T, self.PREFIX, "csa")
        out = _slice_meta(meta, self.KS).csa_meta.compressor_meta
        ref = meta.csa_meta.compressor_meta
        n = self.T - self.KS
        # seq_start_per_req: STILL [B]=1, value += ks (NOT emptied by a slice)
        self.assertEqual(out.seq_start_per_req.numel(), 1)
        self.assertEqual(int(out.seq_start_per_req[0]), self.PREFIX + self.KS)
        # cu_seq_per_req shifted [0, T] -> [0, T-ks]
        self.assertTrue(
            torch.equal(out.cu_seq_per_req, torch.tensor([0, n], dtype=torch.int64))
        )
        # per-token fields sliced, ABSOLUTE values unchanged (no rebase)
        self.assertTrue(torch.equal(out.positions, ref.positions[self.KS :]))
        self.assertTrue(torch.equal(out.kv_slots, ref.kv_slots[self.KS :]))
        self.assertTrue(torch.equal(out.state_slots, ref.state_slots[self.KS :]))
        self.assertTrue(torch.equal(out.token_to_req, ref.token_to_req[self.KS :]))
        self.assertEqual(out.positions.numel(), n)
        self.assertTrue(out.is_batched)

    def test_workspace_meta_qsl_shift_and_keep_full(self):
        for kind in ("csa", "hca"):
            meta = _make_prefill_meta(self.T, self.PREFIX, kind)
            sub = getattr(_slice_meta(meta, self.KS), f"{kind}_meta")
            out = sub.workspace_meta
            ref = getattr(meta, f"{kind}_meta").workspace_meta
            n = self.T - self.KS
            # qsl shifted (doc-11 omitted this; OOB if left full)
            self.assertTrue(
                torch.equal(out.qsl, torch.tensor([0, n], dtype=torch.int32)),
                msg=kind,
            )
            # per-token sliced, values unchanged
            self.assertTrue(
                torch.equal(
                    out.new_k_slot_in_flat, ref.new_k_slot_in_flat[self.KS :]
                ),
                msg=kind,
            )
            self.assertTrue(
                torch.equal(out.dense_cmp_topk, ref.dense_cmp_topk[self.KS :]),
                msg=kind,
            )
            # K-side widths kept FULL (start_pos = swa_seq_lens - query_len)
            self.assertEqual(out.M, ref.M, msg=kind)
            self.assertEqual(out.N, ref.N, msg=kind)
            self.assertTrue(
                torch.equal(out.swa_seq_lens, ref.swa_seq_lens), msg=kind
            )
            self.assertTrue(
                torch.equal(out.swa_gather_lens, ref.swa_gather_lens), msg=kind
            )
            self.assertTrue(torch.equal(out.cmp_seq_lens, ref.cmp_seq_lens), msg=kind)

    def test_indexer_meta_q_side_slice_k_side_full(self):
        meta = _make_prefill_meta(self.T, self.PREFIX, "csa")
        out = _slice_meta(meta, self.KS).csa_meta.indexer_meta
        ref = meta.csa_meta.indexer_meta
        n = self.T - self.KS
        # Q-side recompute / slice
        self.assertEqual(out.M, ref.M - self.KS)
        self.assertEqual(out.seqlen, ref.seqlen - self.KS)
        self.assertTrue(
            torch.equal(out.freqs_cis_slice, ref.freqs_cis_slice[self.KS :])
        )
        self.assertTrue(torch.equal(out.positions_d, ref.positions_d[self.KS :]))
        # struct ks/ke: slice ROWS, do NOT rebase VALUES (per-Q-row K coords)
        self.assertTrue(torch.equal(out.ks, ref.ks[self.KS :]))
        self.assertTrue(torch.equal(out.ke, ref.ke[self.KS :]))
        self.assertEqual(out.positions_d.numel(), n)
        # K-side trim-invariant
        self.assertEqual(out.T, ref.T)
        self.assertTrue(torch.equal(out.cu_kv_seqlens, ref.cu_kv_seqlens))
        self.assertTrue(torch.equal(out.block_table_i32, ref.block_table_i32))
        self.assertIsNone(out.cu_kv_per_token)  # None passthrough on B==1
        # nested compressor also gets the +ks seq_start recompute
        self.assertEqual(int(out.compressor_meta.seq_start_per_req[0]), self.PREFIX + self.KS)

    def test_does_not_mutate_input_meta(self):
        meta = _make_prefill_meta(self.T, self.PREFIX, "csa")
        ref_slot = meta.swa_meta.slot_in_flat.clone()
        ref_sqs = meta.csa_meta.compressor_meta.seq_start_per_req.clone()
        _slice_meta(meta, self.KS)
        self.assertTrue(torch.equal(meta.swa_meta.slot_in_flat, ref_slot))
        self.assertTrue(
            torch.equal(meta.csa_meta.compressor_meta.seq_start_per_req, ref_sqs)
        )


# ----------------------------------------------------------------------
# Numerical equivalence: the sliding-window cascade through the REAL offsets
# ----------------------------------------------------------------------


def _swa_cascade_layer(
    h: torch.Tensor, nwin: int, base: int, w_self: torch.Tensor, w_win: torch.Tensor
) -> torch.Tensor:
    """One faithful SWA-cascade layer over the K span ``h[base:]``.

    Returns the layer output for rows ``[base, T)`` (shape ``[T-base, D]``).
    ``out[r] = h[r] + h[r] @ w_self + mean(h[max(r-nwin+1, base) : r+1]) @ w_win``
    — residual + per-token transform + a causal sliding-window mix clamped to
    the available K rows ``[base, T)``. This is the worst-case (every layer
    SWA) dependency the linear triangle is designed for; if the trim is
    bit-exact here it is bit-exact for the real DSV4 mix (whose global layers
    read trim-invariant cached compressed KV and so are *less* demanding).
    """
    T, _ = h.shape
    rows = []
    for r in range(base, T):
        lo = max(r - nwin + 1, base)
        win = h[lo : r + 1].mean(dim=0)
        rows.append(h[r] + h[r] @ w_self + win @ w_win)
    return torch.stack(rows, dim=0)


class TrimNumericalEquivalenceTest(unittest.TestCase):
    """Run the real offsets + real write-back loop; the kept tail must be
    byte-identical to the uniform forward."""

    def _run(self, L, nwin, restore, T, kb, D=6, seed=0, k_override=None):
        torch.manual_seed(seed)
        embed = torch.randn(T, D, dtype=torch.float64)
        w_self = [torch.randn(D, D, dtype=torch.float64) * 0.1 for _ in range(L)]
        w_win = [torch.randn(D, D, dtype=torch.float64) * 0.1 for _ in range(L)]

        # uniform: every layer over the full residual stream
        hu = embed.clone()
        for j in range(L):
            hu = _swa_cascade_layer(hu, nwin, 0, w_self[j], w_win[j])

        k, tail = _dsv4_zero_swa_trim_offsets(L, nwin, restore, T, kb, "cpu")
        if k_override is not None:
            k = k_override
        self.assertIsNotNone(k, "expected the triangle to engage for this config")

        # trimmed: replicate forward_layers' loop exactly
        ht = embed.clone()
        orig = ht
        for j in range(L):
            ks = k[j]
            if ks > 0:
                out = _swa_cascade_layer(ht, nwin, ks, w_self[j], w_win[j])
                if ht is orig:
                    ht = ht.clone()
                ht[ks:] = out
            else:
                ht = _swa_cascade_layer(ht, nwin, 0, w_self[j], w_win[j])
        return hu, ht, k, tail

    def test_kept_tail_byte_identical(self):
        # restore_eff (8192) > L*nwin so layer 0 also trims the rounding excess.
        hu, ht, k, tail = self._run(L=8, nwin=64, restore=8192, T=4000, kb=64)
        self.assertGreater(k[0], 0, "bottom layer should trim the rounding excess")
        self.assertTrue(
            torch.equal(ht[tail:], hu[tail:]),
            msg=f"kept tail [{tail}:] diverged; max|Δ|="
            f"{(ht[tail:]-hu[tail:]).abs().max().item():.3e}",
        )

    def test_bottom_clamped_region_tail_identical(self):
        # Large L with tail < L*nwin so the bottom layers clamp k_start to 0
        # (run uniformly) and only the upper layers trim. Exercises the
        # clamp-to-0 boundary + the transition into the strictly-stepping
        # region. Tail must match.
        hu, ht, k, tail = self._run(L=30, nwin=64, restore=1000, T=4000, kb=64)
        self.assertEqual(k[0], 0, "bottom layers should clamp to 0 (no trim)")
        self.assertGreater(k[-1], 0, "top layers should still trim")
        self.assertTrue(
            torch.equal(ht[tail:], hu[tail:]),
            msg=f"bottom-clamped tail [{tail}:] diverged; max|Δ|="
            f"{(ht[tail:]-hu[tail:]).abs().max().item():.3e}",
        )

    def test_one_row_of_slack_no_more(self):
        # Tail is correct, but exactly one row earlier (tail-1) need NOT be:
        # confirms the formula is tight (c_{L-1} = tail-1) and the test is
        # measuring the real coverage boundary, not over-keeping.
        hu, ht, k, tail = self._run(L=12, nwin=32, restore=8192, T=3000, kb=32)
        self.assertTrue(torch.equal(ht[tail:], hu[tail:]))

    def test_negative_control_under_cover_diverges(self):
        # Shift every k_start one nwin window LATER (under-cover by one window):
        # each layer's K span now starts too high, so the cascade must diverge
        # on the tail. Proves the equivalence test is sensitive to a one-window
        # coverage error — the exact bug class this formula had to fix.
        k, tail = _dsv4_zero_swa_trim_offsets(8, 64, 8192, 4000, 64, "cpu")
        bad = [min(x + 64, 3999) for x in k]
        hu, ht, _, _ = self._run(
            L=8, nwin=64, restore=8192, T=4000, kb=64, k_override=bad
        )
        self.assertFalse(
            torch.equal(ht[tail:], hu[tail:]),
            msg="under-covering K span (+nwin) must diverge — if this passes "
            "the equivalence test is not sensitive to coverage errors",
        )


if __name__ == "__main__":
    unittest.main()
