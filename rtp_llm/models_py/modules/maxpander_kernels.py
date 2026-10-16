"""GPU-only two-replica WF kernels, extracted from the existing WF dispatcher.

No host/torch fallback; routing remains safe during CUDA graph capture.
"""

import triton
import triton.language as tl


@triton.jit
def _wf_count_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    addresses = offs // TOPK * row_stride + offs % TOPK
    e = tl.load(topk_ids_ptr + addresses, mask=mask, other=-1).to(tl.int32)
    valid = mask & (e >= 0) & (e < num_logical)
    safe_e = tl.where(valid, e, 0)
    tl.atomic_add(counts_ptr + safe_e, 1.0, mask=valid)


@triton.jit
def _wf_hot_order_kernel(
    counts_ptr,
    hot_order_ptr,
    rank_counter_ptr,
    num_logical,
    BLOCK_EXPERTS: tl.constexpr,
):
    """Sort demand descending with logical-ID tie breaks and reset scatter ranks."""
    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < num_logical
    counts = tl.load(counts_ptr + experts, mask=valid, other=0).to(tl.uint64)
    tie_break = tl.where(valid, num_logical - experts, 0).to(tl.uint64)
    keys = (counts << 32) | tie_break
    ordered = tl.sort(keys, descending=True)
    logical_ids = (num_logical - (ordered & 0xFFFFFFFF)).to(tl.int32)
    tl.store(hot_order_ptr + experts, logical_ids, mask=valid)
    tl.store(rank_counter_ptr + experts, 0, mask=valid)


@triton.jit
def _wf_count_hist_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_logical,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NBINS: tl.constexpr,
):
    """Same result as _wf_count_kernel, one vector atomic per block instead of one
    scalar atomic per id.

    _wf_count_kernel issues ``numel`` atomic_adds into only ``num_logical``
    addresses (256 at production scale), and triton does not warp-aggregate
    same-address atomics -- at 3.7M ids that serialization, not bandwidth,
    dominates the whole dispatch. Histogramming in registers first turns it into
    NBINS atomics per block.

    Bin ``num_logical`` is a dedicated overflow bin for -1 sentinels, out-of-range
    ids and masked tail lanes. It must exist (hence NBINS > num_logical): triton's
    histogram folds indices >= num_bins into bin 0 rather than dropping them, so
    parking invalid lanes past the end would corrupt logical expert 0.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (e >= 0) & (e < num_logical)
    hist = tl.histogram(tl.where(valid, e, num_logical), NBINS)
    bins = tl.arange(0, NBINS)
    tl.atomic_add(counts_ptr + bins, hist.to(tl.float32), mask=bins < num_logical)


@triton.jit
def _wf_prep_kernel(
    log2phy_ptr,
    logic_expert_cnt_ptr,
    cand_phys_ptr,
    cand_gpu_ptr,
    rep_count_ptr,
    num_logical,
    stride_e,
    stride_k,
    experts_per_gpu,
    BLOCK_E: tl.constexpr,
    HAS_SECOND: tl.constexpr,
):
    """Derive the compact [E, 2] candidate tables from the live placement.

    One program, one launch: replaces the slice/contiguous, ge, full_like,
    floordiv, where and clamp torch ops, whose launch overhead is pure latency at
    E <= 1024.

    ``HAS_SECOND`` is False when ``log2phy`` has a single column (pad_k == 1, i.e.
    no redundant experts): there is no second replica to read, and everything
    falls through the k == 1 branch of _fill2 / _round2.
    """
    offs = tl.arange(0, BLOCK_E)
    m = offs < num_logical
    p0 = tl.load(log2phy_ptr + offs * stride_e, mask=m, other=-1).to(tl.int32)
    if HAS_SECOND:
        p1 = tl.load(log2phy_ptr + offs * stride_e + stride_k, mask=m, other=-1).to(
            tl.int32
        )
    else:
        p1 = tl.full([BLOCK_E], -1, tl.int32)
    k = tl.load(logic_expert_cnt_ptr + offs, mask=m, other=1).to(tl.int32)
    tl.store(cand_phys_ptr + offs * 2 + 0, p0, mask=m)
    tl.store(cand_phys_ptr + offs * 2 + 1, p1, mask=m)
    tl.store(
        cand_gpu_ptr + offs * 2 + 0,
        tl.where(p0 >= 0, p0 // experts_per_gpu, -1),
        mask=m,
    )
    tl.store(
        cand_gpu_ptr + offs * 2 + 1,
        tl.where(p1 >= 0, p1 // experts_per_gpu, -1),
        mask=m,
    )
    # k == 2 only when a second replica really exists, so an inconsistent
    # logic_expert_cnt can never make the water-fill pour into a -1 slot.
    tl.store(rep_count_ptr + offs, tl.where(p1 >= 0, tl.minimum(k, 2), 1), mask=m)


@triton.jit
def _fill2(c, k, s0, s1):
    """Closed-form 2-way water-fill: pour ``c`` into replicas at loads s0,s1.

    Returns fractional allocation (a0, a1) in candidate order. For a single
    replica (k==1) all of ``c`` lands on replica 0.
    """
    gap = tl.abs(s0 - s1)
    level = (s0 + s1 + c) * 0.5
    two_a0 = tl.where(s0 <= s1, tl.minimum(c, gap), 0.0)
    two_a1 = tl.where(s0 <= s1, 0.0, tl.minimum(c, gap))
    split0 = level - s0
    split1 = level - s1
    use_split = c > gap
    a0_2 = tl.where(use_split, split0, two_a0)
    a1_2 = tl.where(use_split, split1, two_a1)
    a0 = tl.where(k == 2, a0_2, c)
    a1 = tl.where(k == 2, a1_2, 0.0)
    return a0, a1


@triton.jit
def _snap2(c, k, a0, a1, TILE: tl.constexpr):
    """Align the 2-way split point to a multiple of the grouped-GEMM tile.

    ``TILE == 0`` disables this and returns ``_fill2``'s allocation unchanged.

    Tile padding is charged on the row count of a physical slot summed over all
    source ranks, while this kernel only sees one rank's demand -- so a slot
    costs a tile if ANY rank routes anything there. The lever is cross-rank
    AGREEMENT on which replica the small experts use, and agreement here is
    free because the rule is (almost) input-independent: a rank's median demand
    for one expert is well under one tile, so ``a0 ~ c/2`` always rounds the
    same way on every rank without exchanging a byte. Balance survives because
    each snap freezes at most one tile of demand, and the coordinate-descent
    passes still have most of the mass to level with -- which makes
    MAX_PASSES > 1 (policy wf_iter) part of the design, not an option.
    """
    if TILE == 0:
        return a0, a1
    # Round-half-up; the tie direction is immaterial (measured offline).
    q = tl.floor(a0 / TILE + 0.5) * TILE
    q = tl.minimum(tl.maximum(q, 0.0), c)
    return tl.where(k == 2, q, a0), tl.where(k == 2, c - q, a1)


@triton.jit
def _round2(c, k, a0, a1):
    """Largest-remainder integer rounding of (a0,a1) summing to int(c)."""
    ci = (c + 0.5).to(tl.int32)
    f0 = tl.floor(a0)
    f1 = tl.floor(a1)
    i0 = f0.to(tl.int32)
    i1 = f1.to(tl.int32)
    rem = ci - i0 - i1
    # rem is 0 or 1; award it to the larger fractional part, ties -> replica 0.
    frac0 = a0 - f0
    frac1 = a1 - f1
    give0 = (rem == 1) & (frac0 >= frac1)
    give1 = (rem == 1) & (frac1 > frac0)
    q0 = i0 + give0.to(tl.int32)
    q1 = i1 + give1.to(tl.int32)
    # k==1: everything on replica 0.
    q0 = tl.where(k == 2, q0, ci)
    q1 = tl.where(k == 2, q1, 0)
    return q0, q1


@triton.jit
def _wf_quota_kernel(
    counts_ptr,
    hot_order_ptr,
    cand_gpu_ptr,
    rep_count_ptr,
    quota_ptr,
    alloc_ptr,
    num_logical,
    MAX_PASSES: tl.constexpr,
    BLOCK_G: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # Single program instance. recv[G] lives in registers across the whole
    # sequential scan; G is tiny (<= 8) so BLOCK_G lanes cover it.
    idx = tl.arange(0, BLOCK_G)
    recv = tl.zeros([BLOCK_G], dtype=tl.float32)

    # ---- Pass 0: hot-first single fill (this alone is policy "wf"). ----
    for i in range(num_logical):
        e = tl.load(hot_order_ptr + i)
        c = tl.load(counts_ptr + e)
        k = tl.load(rep_count_ptr + e)
        g0 = tl.load(cand_gpu_ptr + e * 2 + 0)
        g1 = tl.load(cand_gpu_ptr + e * 2 + 1)
        s0 = tl.sum(tl.where(idx == g0, recv, 0.0))
        s1 = tl.sum(tl.where(idx == g1, recv, 0.0))
        a0, a1 = _fill2(c, k, s0, s1)
        # Snap BEFORE folding into recv, so later experts water-fill against the
        # allocation that will actually happen rather than a discarded fraction.
        a0, a1 = _snap2(c, k, a0, a1, TILE_M)
        recv = recv + tl.where(idx == g0, a0, 0.0) + tl.where(idx == g1, a1, 0.0)
        q0, q1 = _round2(c, k, a0, a1)
        tl.store(quota_ptr + e * 2 + 0, q0)
        tl.store(quota_ptr + e * 2 + 1, q1)
        tl.store(alloc_ptr + e * 2 + 0, a0)
        tl.store(alloc_ptr + e * 2 + 1, a1)

    # ---- Passes 1..: coordinate descent (policy "wf_iter"). Only 2-replica
    # experts move; remove their old alloc, re-fill against the current recv. --
    for _p in range(MAX_PASSES - 1):
        for i in range(num_logical):
            e = tl.load(hot_order_ptr + i)
            k = tl.load(rep_count_ptr + e)
            c = tl.load(counts_ptr + e)
            do = (k == 2) & (c > 0.0)
            if do:
                g0 = tl.load(cand_gpu_ptr + e * 2 + 0)
                g1 = tl.load(cand_gpu_ptr + e * 2 + 1)
                oa0 = tl.load(alloc_ptr + e * 2 + 0)
                oa1 = tl.load(alloc_ptr + e * 2 + 1)
                recv = (
                    recv - tl.where(idx == g0, oa0, 0.0) - tl.where(idx == g1, oa1, 0.0)
                )
                s0 = tl.sum(tl.where(idx == g0, recv, 0.0))
                s1 = tl.sum(tl.where(idx == g1, recv, 0.0))
                a0, a1 = _fill2(c, k, s0, s1)
                a0, a1 = _snap2(c, k, a0, a1, TILE_M)
                recv = (
                    recv + tl.where(idx == g0, a0, 0.0) + tl.where(idx == g1, a1, 0.0)
                )
                q0, q1 = _round2(c, k, a0, a1)
                tl.store(quota_ptr + e * 2 + 0, q0)
                tl.store(quota_ptr + e * 2 + 1, q1)
                tl.store(alloc_ptr + e * 2 + 0, a0)
                tl.store(alloc_ptr + e * 2 + 1, a1)


@triton.jit
def _wf_scatter_kernel(
    topk_ids_ptr,
    quota_ptr,
    cand_phys_ptr,
    rank_counter_ptr,
    out_ids_ptr,
    num_logical,
    numel,
    row_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    addresses = offs // TOPK * row_stride + offs % TOPK
    e = tl.load(topk_ids_ptr + addresses, mask=mask, other=-1).to(tl.int32)
    valid = mask & (e >= 0) & (e < num_logical)
    safe_e = tl.where(valid, e, 0)
    rank = tl.atomic_add(rank_counter_ptr + safe_e, 1, mask=valid)
    q0 = tl.load(quota_ptr + safe_e * 2 + 0, mask=valid, other=0)
    p0 = tl.load(cand_phys_ptr + safe_e * 2 + 0, mask=valid, other=-1)
    p1 = tl.load(cand_phys_ptr + safe_e * 2 + 1, mask=valid, other=-1)
    phys = tl.where(rank < q0, p0, p1)
    phys = tl.where(valid, phys, -1)
    tl.store(out_ids_ptr + addresses, phys, mask=mask)
