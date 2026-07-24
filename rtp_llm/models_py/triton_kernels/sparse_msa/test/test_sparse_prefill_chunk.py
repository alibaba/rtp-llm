# SPDX-License-Identifier: MIT
"""Unit test: chunked vs non-chunked sparse attention precision.

Validates that ``_sparse_attn_chunked`` (the M3_SPARSE_ATTN_CHUNK_ENABLE
path in ``topk_bt_fused.py``) produces attention output matching the full
non-chunked ``sparse_atten_func`` reference. Chunking over the query dim is
lossless by construction (each query row only depends on its own top-k KV
blocks), so the two must agree to within bf16 numerical noise.

Covered cases:
  * single request, total_q divisible by chunk_size (even partition),
  * single request, total_q NOT divisible by chunk_size (trailing partial
    chunk -- exercises the boundary where the last chunk is smaller),
  * multi-batch ragged request (B=2, different q/k lengths) -- exercises
    per-batch cu_seqlens / page_table / seqused_k partitioning,
  * mixed fp8 KV cache + bf16 Q,
  * **CP zigzag regression** (B=2 with ``seqused_k < kv_segment_lens`` on the
    first segment). Under load-balanced/zigzag context parallelism a segment's
    causal ``seqused_k`` (= qo_offset + q_len) is smaller than the full KV run
    it physically owns (``kv_segment_lens``). The chunked page table must be
    sized / advanced by ``kv_segment_lens`` (the layout ``build_kv_page_indices``
    used), not by ``seqused_k``; otherwise the per-segment ``off`` cursor
    desyncs and later segments read the wrong physical pages. This case fails on
    that bug (it advances ``off`` by the shorter causal length) and passes on
    the fix. All the prefix=0 cases above have ``seqused_k == kv_segment_lens``
    so they cannot catch this regression on their own.
"""
import unittest

import torch


def _sm100_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import fmha_sm100  # noqa: F401
    except Exception:
        return False
    try:
        return torch.cuda.get_device_capability(0)[0] == 10
    except Exception:
        return False


def _seg(q_len: int, prefix: int = 0, kv_run: int = None) -> dict:
    """One request segment.

    ``q_len``   number of query rows,
    ``prefix``  causal KV prefix before the first query (seqused_k = prefix + q_len),
    ``kv_run``  full physical KV run the segment owns in the page table
                (defaults to seqused_k; > seqused_k models zigzag CP).
    """
    seqused = prefix + q_len
    if kv_run is None:
        kv_run = seqused
    assert kv_run >= seqused, "kv_run must cover the causal seqused_k"
    return dict(q_len=q_len, prefix=prefix, kv_run=kv_run)


def _num_blocks(kv_len: int, blk: int) -> int:
    return (kv_len + blk - 1) // blk


def _make_topk_idx(
    nkv: int,
    q_len: int,
    topk: int,
    blk: int,
    device,
    seed: int,
    prefix: int,
    num_run_blocks: int,
) -> torch.Tensor:
    """Causal-valid random topk block ids (segment-local) with -1 padding.

    Local query i sits at causal position ``prefix + i``, so it may attend
    blocks ``j`` with ``j <= (prefix + i) // blk`` (visible count
    ``(prefix + i) // blk + 1``). Block ids range over the segment's full run
    ``[0, num_run_blocks)``; slots beyond the per-query visible count are -1.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.arange(q_len)
    avail = (prefix + i) // blk + 1  # visible (causal) block count per query
    scores = torch.rand(nkv, q_len, num_run_blocks, generator=g)
    scores = scores.masked_fill(
        torch.arange(num_run_blocks)[None, None, :] >= avail[None, :, None], -1.0
    )
    k = min(topk, num_run_blocks)
    idx = scores.topk(k, dim=-1).indices.int()
    if idx.shape[-1] < topk:
        idx = torch.cat(
            [idx, idx.new_full((nkv, q_len, topk - idx.shape[-1]), -1)], dim=-1
        )
    pad = torch.arange(topk)[None, None, :] >= avail[None, :, None]
    return idx.masked_fill(pad, -1).contiguous().to(device)


def _make_topk_idx_segs(nkv, segs, topk, blk, device, seed) -> torch.Tensor:
    """Per-segment causal topk_idx concatenated along dim 1 ([nkv, total_q, topk])."""
    parts = [
        _make_topk_idx(
            nkv,
            s["q_len"],
            topk,
            blk,
            device,
            seed + b,
            s["prefix"],
            _num_blocks(s["kv_run"], blk),
        )
        for b, s in enumerate(segs)
    ]
    return torch.cat(parts, dim=1)


def _aligned_page_table(ids_per_batch, device) -> torch.Tensor:
    """Build a [B, max_n] int32 page table whose last dim is contiguous and
    whose every row is 16-byte aligned (the cute kernel asserts %16 == 0 on
    each row's base). Rows are padded with -1 beyond each batch's page count.
    """
    B = len(ids_per_batch)
    max_n = max(len(x) for x in ids_per_batch)
    # Pad the row width up to a multiple of 4 int32 (=16 bytes) so row 1..B-1
    # stay 16-byte aligned when laid out contiguously.
    max_n_pad = ((max_n + 3) // 4) * 4
    pt = torch.full((B, max_n_pad), -1, dtype=torch.int32, device=device)
    for b, ids in enumerate(ids_per_batch):
        pt[b, : len(ids)] = ids
    return pt[:, :max_n]


def _run_full_reference(
    builder, q, k_paged, v_paged, topk_idx, segs, blk, topk, sm_scale, device
):
    """Non-chunked ground truth: one CSR build + one sparse_atten_func call over
    the full (possibly multi-batch) sequence.

    The page table / cu_seqlens_k / total_k are sized by the FULL per-segment KV
    run (``kv_run`` == kv_segment_lens); ``seqused_k`` carries the (possibly
    smaller) causal length per segment. This mirrors the production
    non-chunked path and is exactly the geometry the chunked path must match.
    """
    from interface import sparse_atten_func

    hq = q.shape[1]
    hkv = k_paged.shape[1]
    qo_lens = [s["q_len"] for s in segs]
    kv_lens = [s["kv_run"] for s in segs]
    seqused = [s["prefix"] + s["q_len"] for s in segs]
    total_k = int(sum(kv_lens))
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    total_rows = (total_k + blk - 1) // blk
    sk = torch.tensor(seqused, dtype=torch.int32, device=device)

    # Physical page ids per batch: contiguous [0..n0-1], [n0..n0+n1-1], ...
    # sized by the FULL kv run (identity mapping shared with kv_indices below).
    off = 0
    ids_per_batch = []
    for kl in kv_lens:
        n = _num_blocks(kl, blk)
        ids_per_batch.append(
            torch.arange(off, off + n, dtype=torch.int32, device=device)
        )
        off += n
    pt = _aligned_page_table(ids_per_batch, device)

    row_ptr, q_ind, sched = builder(
        topk_idx,
        cu_q,
        cu_k,
        total_k=total_k,
        blk_kv=blk,
        max_seqlen_k=total_k,
        max_seqlen_q=max(qo_lens),
        total_rows=total_rows,
        qhead_per_kv=hq // hkv,
        return_schedule=True,
    )
    out = sparse_atten_func(
        q,
        k_paged,
        v_paged,
        row_ptr,
        q_ind,
        topk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(qo_lens),
        max_seqlen_k=total_k,
        blk_kv=blk,
        causal=True,
        softmax_scale=sm_scale,
        return_softmax_lse=False,
        page_table=pt,
        seqused_k=sk,
        schedule=sched,
        usable_SM_count=-1,
    )
    return out[0] if isinstance(out, tuple) else out


@unittest.skipUnless(_sm100_available(), "SM100 CUDA device + fmha_sm100 required")
class SparsePrefillChunkTest(unittest.TestCase):
    HQ, HKV, DIM, BLK, TOPK = 64, 4, 128, 128, 16

    def setUp(self):
        # Importing fmha_sm100 eagerly sets up the cute/ sys.path entries that
        # ``_sparse_attn_chunked`` relies on for its lazy ``from src.sm100...``
        # / ``from interface import ...`` imports.
        import fmha_sm100  # noqa: F401

        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            topk_bt_fused as tbf,
        )

        self.tbf = tbf
        # Reset the per-device workspace cache so each case starts clean and
        # so we can assert reuse within a case.
        tbf._M3_CHUNK_WS_CACHE.clear()
        torch.cuda.set_device(0)

    def _make_inputs(self, segs, device, seed=0, kv_dtype=torch.bfloat16):
        from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100

        nkv, hq, dim, blk, topk = self.HKV, self.HQ, self.DIM, self.BLK, self.TOPK
        total_q = int(sum(s["q_len"] for s in segs))
        # Physical KV pages: full per-segment runs laid out contiguously, which
        # is what build_kv_page_indices produces (kv_indices = identity here).
        num_pages = int(sum(_num_blocks(s["kv_run"], blk) for s in segs))
        torch.manual_seed(seed)
        q = torch.randn(total_q, hq, dim, dtype=torch.bfloat16, device=device)
        # fp8_e4m3 range is ~[-448, 448] but precision collapses outside ~[-2, 2];
        # scale randn down so the fp8 KV cache holds realistic (non-saturated)
        # values. Q stays bf16 -- the kernel's mixed fp8-KV / bf16-Q path.
        kv_scale = 0.3 if kv_dtype == torch.float8_e4m3fn else 1.0
        k = (torch.randn(num_pages, nkv, blk, dim, device=device) * kv_scale).to(
            kv_dtype
        )
        v = (torch.randn(num_pages, nkv, blk, dim, device=device) * kv_scale).to(
            kv_dtype
        )
        sm_scale = dim**-0.5
        topk_idx = _make_topk_idx_segs(nkv, segs, topk, blk, device, seed + 1)
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        builder = SparseK2qCsrBuilderSm100()
        builder._ensure_loaded()
        return q, k, v, topk_idx, kv_indices, builder, sm_scale

    def _run_chunked(
        self,
        q,
        k,
        v,
        topk_idx,
        kv_indices,
        segs,
        chunk_size,
        sm_scale,
        device,
        usable_sm=-1,
    ):
        nkv = self.HKV
        qo_lens = [s["q_len"] for s in segs]
        seqused = [s["prefix"] + s["q_len"] for s in segs]
        kv_runs = [s["kv_run"] for s in segs]
        plan = dict(
            num_kv_heads=nkv,
            qo_segment_lens=torch.tensor(qo_lens, dtype=torch.int32),
            seqused_k=torch.tensor(seqused, dtype=torch.int32, device=device),
            # The page table is sized/advanced by the FULL kv run, NOT seqused_k
            # -- this key is what distinguishes the fixed path from the bug.
            kv_segment_lens=torch.tensor(kv_runs, dtype=torch.int32, device=device),
            causal=True,
            # usable_SM_count > 0 makes the builder skip its schedule (return_schedule
            # =False) and sparse_atten_func build its own -- a distinct workspace path.
            usable_SM_count=usable_sm,
        )
        out = self.tbf._sparse_attn_chunked(
            q,
            k,
            v,
            topk_idx,
            kv_indices,
            plan,
            self.TOPK,
            self.BLK,
            sm_scale,
            chunk_size,
        )
        # Drop the per-forward chunk metadata so the next case rebuilds it.
        plan.pop("_chunk_meta", None)
        return out

    def _assert_close(self, out, ref, case_name):
        both = torch.cat([out.reshape(-1), ref.reshape(-1)])
        self.assertFalse(
            torch.isnan(both).any() or torch.isinf(both).any(),
            f"{case_name}: non-finite output",
        )
        max_abs = (out.float() - ref.float()).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.reshape(1, -1).float(), ref.reshape(1, -1).float(), dim=1
        ).item()
        bitwise = torch.equal(out, ref)
        print(f"[{case_name}] max_abs={max_abs:.3e} cos={cos:.6f} bitwise={bitwise}")
        self.assertGreater(
            cos, 0.999, f"{case_name}: cosine similarity {cos:.6f} below 0.999"
        )
        # bf16 attention with a different reduction schedule can differ in the
        # last bit; 0.05 is a comfortable bound well above noise and well below
        # any real regression (a chunking bug would diverge by O(1)).
        self.assertLess(max_abs, 0.05, f"{case_name}: max_abs {max_abs:.3e} too large")

    def _run_case(
        self, segs, chunk_size, case_name, seed=0, kv_dtype=torch.bfloat16, usable_sm=-1
    ):
        device = torch.device("cuda", 0)
        q, k, v, topk_idx, kv_indices, builder, sm = self._make_inputs(
            segs, device, seed=seed, kv_dtype=kv_dtype
        )
        ref = _run_full_reference(
            builder, q, k, v, topk_idx, segs, self.BLK, self.TOPK, sm, device
        )
        out = self._run_chunked(
            q, k, v, topk_idx, kv_indices, segs, chunk_size, sm, device, usable_sm
        )
        self._assert_close(out, ref, case_name)
        return out, ref

    def test_even_partition(self):
        # single request, total_q divisible by chunk_size.
        self._run_case([_seg(8192)], 2048, "even_partition")
        self.assertGreater(len(self.tbf._M3_CHUNK_WS_CACHE), 0)  # cache populated

    def test_uneven_trailing_chunk(self):
        # 5120 = 4096 + 1024: last chunk is a quarter of chunk_size.
        self._run_case([_seg(5120)], 4096, "uneven_trailing_chunk", seed=7)

    def test_multi_batch_ragged(self):
        # B=2 ragged, prefix=0 (seqused_k == kv_segment_lens).
        self._run_case([_seg(3072), _seg(2048)], 2048, "multi_batch_ragged", seed=13)

    def test_fp8_kv_cache(self):
        # Mixed fp8_e4m3fn KV cache + bf16 Q: the kernel auto-detects this via
        # _resolve_forward_mma_dtypes (qk_dtype=bf16, pv_dtype=bf16). Verifies
        # chunking is lossless on the fp8-KV path too -- both the chunked call
        # and the full reference consume the SAME fp8 K/V, so they must agree.
        self._run_case(
            [_seg(8192)], 2048, "fp8_kv_cache", seed=21, kv_dtype=torch.float8_e4m3fn
        )

    def test_cp_zigzag_prefix(self):
        # Regression for the page-table off-by-run bug under zigzag CP:
        #   seg0: q=256, prefix=128 -> seqused_k=384 (3 blocks),
        #         kv_run=640         -> 5 physical pages,
        #   seg1: q=256, prefix=0   -> seqused_k=256 (2 blocks), kv_run=256.
        # The buggy code advanced the page cursor by ceil(seqused_k/blk)=3 pages
        # after seg0, so seg1's page table pointed at seg0's tail pages {3,4}
        # instead of its own {5,6} -> seg1 output is garbage. The fix advances by
        # ceil(kv_run/blk)=5. seg0 output is identical either way (it only reads
        # pages {0,1,2}), so ~half the rows diverge under the bug -> cos << 0.999.
        segs = [_seg(256, prefix=128, kv_run=640), _seg(256)]
        self._run_case(segs, 128, "cp_zigzag_prefix", seed=29)

    def test_usable_sm_positive(self):
        # usable_SM_count > 0 exercises the OTHER workspace path: the builder
        # runs without a schedule (return_schedule=False -> _run) and
        # sparse_atten_func builds its own schedule (schedule=None). Result is
        # scheduling-invariant, so it must still match the full reference (built
        # with usable_SM_count=-1). All other cases only cover the -1 path.
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        self._run_case(
            [_seg(8192)], 2048, "usable_sm_positive", seed=37, usable_sm=sm_count
        )

    def test_workspace_reuse_and_growth(self):
        # The reusable per-device buffer (_M3_CHUNK_WS_CACHE) must grow when a
        # later plan needs more bytes and be reused (no realloc) when a later
        # plan fits. setUp() cleared the cache, so it starts empty here.
        dev = torch.device("cuda", 0)
        self._run_case([_seg(2048)], 512, "ws_small", seed=41)
        small = self.tbf._M3_CHUNK_WS_CACHE[dev]
        n_small = small.numel()

        # Larger chunk_size -> larger O_partial -> the buffer must grow.
        self._run_case([_seg(8192)], 4096, "ws_large", seed=42)
        large = self.tbf._M3_CHUNK_WS_CACHE[dev]
        self.assertGreater(large.numel(), n_small, "workspace did not grow")
        large_ptr, n_large = large.data_ptr(), large.numel()

        # Smaller plan again -> fits in the grown buffer -> no reallocation.
        self._run_case([_seg(2048)], 512, "ws_reuse", seed=43)
        reused = self.tbf._M3_CHUNK_WS_CACHE[dev]
        self.assertEqual(reused.data_ptr(), large_ptr, "workspace was reallocated")
        self.assertEqual(reused.numel(), n_large)


if __name__ == "__main__":
    unittest.main()
