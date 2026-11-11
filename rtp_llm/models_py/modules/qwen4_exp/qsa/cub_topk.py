"""Opt-in SM120 QSA TopK with native CUDA ordering and token expansion.

The model-local kernel uses NVIDIA CUB's installed stable block sort APIs. It does
not vendor vLLM/CUB source. Stable uint32 keys retain the pinned Torch TopK
selection and raw radix order, including signed zero and both NaN signs.
Compilation occurs during eager warmup; an unprimed CUDA graph falls back.
"""

import hashlib
import logging
import threading

import torch

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
_MODULE = None
_BUILD_FAILED = False
_BUILD_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_CUDA_SOURCE = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <type_traits>

namespace rtp_llm {

// O(N) radix-select of the native top-512 for decode shapes (SM90 L20X).
// One block per row; 4 passes of 8-bit shared-memory histograms narrow the
// 512th key, then the selected 512 are stable-sorted (equal keys keep the
// compacted column order, matching the native tie order) and emitted with
// the same token-expansion contract as qsaCubStableTopK.
template<int Threads = 512>
__global__ void qsaRadixTopK(const float*   scores,
                             const int32_t* lengths,
                             int32_t*       output,
                             int            columns,
                             bool           finite_only) {
    constexpr int K         = 512;
    constexpr int MAX_ITEMS = 16;
    using SelectedSort      = cub::BlockRadixSort<uint32_t, Threads, 1, uint32_t>;
    using BlockScan          = cub::BlockScan<uint32_t, Threads>;
    __shared__ union SharedOps {
        typename SelectedSort::TempStorage sort;
        typename BlockScan::TempStorage    scan;
    } ops;
    __shared__ uint32_t hist[K + 1];
    __shared__ uint32_t sel_key[K];
    __shared__ uint32_t sel_col[K];
    __shared__ uint32_t sh_threshold;
    __shared__ int      sh_ties;
    __shared__ uint32_t sh_boundary;

    const int      row      = blockIdx.x;
    const int64_t  length   = lengths[row];
    const int64_t  complete = (length - (length < 0 ? 3 : 0)) / 4;
    const float*   srow     = scores + (int64_t)row * columns;
    const int      items    = (columns + Threads - 1) / Threads;
    const uint32_t base     = threadIdx.x * items;

    uint32_t key[MAX_ITEMS];
    bool     cand[MAX_ITEMS];
#pragma unroll 4
    for (int i = 0; i < items; ++i) {
        const uint32_t col  = base + (uint32_t)i;
        const uint32_t bits = (col < (uint32_t)columns && col < (uint32_t)complete)
                                  ? __float_as_uint(srow[col])
                                  : 0xff800000u;
        const uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        const bool     is_nan =
            (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
        key[i]  = is_nan ? 0xffffffffu : ordered;
        cand[i] = col < (uint32_t)columns;
    }

    uint32_t threshold = 0;
    int      remaining  = K;
    for (int p = 3; p >= 0; --p) {
        for (int j = threadIdx.x; j <= K; j += Threads)
            hist[j] = 0;
        __syncthreads();
        const uint32_t shift = (uint32_t)p * 8;
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            if (cand[i])
                atomicAdd(&hist[((key[i] >> shift) & 0xffu) + 1], 1u);
        __syncthreads();
        if (threadIdx.x == 0) {
            // ge(d) = count(digit >= d), non-increasing in d; boundary is the
            // largest d with ge(d) >= remaining. ge_above tracks count(digit >
            // d) while scanning down, so at the boundary it is exactly the
            // strictly-above count.
            uint32_t ge_above = 0;
            int      boundary = 0;
            for (int d = 255; d >= 0; --d) {
                const uint32_t ge_d = ge_above + hist[d + 1];
                if (ge_d >= (uint32_t)remaining) {
                    boundary = d;
                    break;
                }
                ge_above = ge_d;
            }
            remaining -= (int)ge_above;
            threshold  = (threshold << 8) | (uint32_t)boundary;
            sh_boundary = (uint32_t)boundary;
        }
        __syncthreads();
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            cand[i] = cand[i] && (((key[i] >> shift) & 0xffu) == sh_boundary);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sh_threshold = threshold;
        sh_ties      = remaining;
    }
    __syncthreads();
    const uint32_t thr = sh_threshold;

    // Selection: strictly above the threshold plus the first `ties`
    // threshold-valued columns in column order. The tie cutoff depends on a
    // block-wide scan, so decide kept flags first, then scan the kept totals
    // for the compaction write offsets.
    uint32_t tie_n = 0;
#pragma unroll 4
    for (int i = 0; i < items; ++i) {
        const uint32_t col = base + (uint32_t)i;
        if (col < (uint32_t)columns && key[i] == thr)
            ++tie_n;
    }
    uint32_t tie_base = 0;
    BlockScan(ops.scan).ExclusiveSum(tie_n, tie_base);
    __syncthreads();
    bool kept[MAX_ITEMS];
    {
        uint32_t tie_rank = tie_base;
#pragma unroll 4
        for (int i = 0; i < items; ++i) {
            const uint32_t col = base + (uint32_t)i;
            kept[i] = false;
            if (col < (uint32_t)columns) {
                if (key[i] > thr)
                    kept[i] = true;
                else if (key[i] == thr) {
                    kept[i] = tie_rank < (uint32_t)sh_ties;
                    ++tie_rank;
                }
            }
        }
    }
    uint32_t keep_n = 0;
#pragma unroll 4
    for (int i = 0; i < items; ++i)
        if (kept[i])
            ++keep_n;
    uint32_t keep_base = 0;
    BlockScan(ops.scan).ExclusiveSum(keep_n, keep_base);
    __syncthreads();
    {
        uint32_t rank = keep_base;
#pragma unroll 4
        for (int i = 0; i < items; ++i) {
            if (!kept[i])
                continue;
            sel_key[rank] = key[i];
            sel_col[rank] = base + (uint32_t)i;
            ++rank;
        }
    }
    __syncthreads();

    // Stable descending sort by key; the compaction order (column ascending)
    // is preserved for ties, matching the frozen native TopK order.
    uint32_t s_key[1], s_col[1];
    s_key[0] = sel_key[threadIdx.x];
    s_col[0] = sel_col[threadIdx.x];
    SelectedSort(ops.sort).SortDescending(s_key, s_col);
    __syncthreads();

    // If a NaN was selected, re-sort by its raw radix bits so NaN payloads
    // sit exactly where the native kernel places them (positive NaNs toward
    // the top, negative NaNs at the selection tail).
    {
        const uint32_t bits = (s_col[0] < (uint32_t)columns && s_col[0] < (uint32_t)complete)
                                  ? __float_as_uint(srow[s_col[0]])
                                  : 0xff800000u;
        const bool nan_selected =
            (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
        if (__syncthreads_or(nan_selected)) {
            const uint32_t raw_ordered =
                (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
            sel_key[threadIdx.x] = raw_ordered;
            sel_col[threadIdx.x] = s_col[0];
            __syncthreads();
            uint32_t r_key[1], r_col[1];
            r_key[0] = sel_key[threadIdx.x];
            r_col[0] = sel_col[threadIdx.x];
            SelectedSort(ops.sort).SortDescending(r_key, r_col);
            __syncthreads();
            s_key[0] = r_key[0];
            s_col[0] = r_col[0];
        }
    }

    // Emit: one thread per selected rank, token expansion plus the tail.
    {
        const uint32_t bits = (s_col[0] < (uint32_t)columns && s_col[0] < (uint32_t)complete)
                                  ? __float_as_uint(srow[s_col[0]])
                                  : 0xff800000u;
        const bool valid = ((bits & 0x7f800000u) != 0x7f800000u
                            || (!finite_only && bits == 0x7f800000u))
                           && s_col[0] < (uint32_t)columns && s_col[0] < (uint32_t)complete;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            output[row * 2051 + threadIdx.x * 4 + j] =
                valid ? (int32_t)(s_col[0] * 4 + j) : -1;
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            output[row * 2051 + 2048 + j] =
                j < length - complete * 4 ? (int32_t)(complete * 4 + j) : -1;
    }
}

struct QsaGreater {
    __device__ bool operator()(uint32_t a, uint32_t b) const {
        return a > b;
    }
};

// Ported from sglang v0.5.22 fast_topk (kernels/jit/csrc/elementwise/
// fast_topk.cuh): one coarse fp16-truncated byte pass narrows the top-512
// boundary, the boundary-bin candidates are collected into double-buffered
// shared memory while the next-byte histogram is fused into the collection,
// and three refinement rounds touch only those candidates instead of
// rescanning every column (the frozen four-pass select below rescans four
// times). Complete-block masking, the stable CUB order restore, the NaN
// repair pass and the 2051-slot token expansion are shared with qsaRadixTopK,
// so the output contract (frozen native order, ties by ascending column) is
// unchanged. Rows are unrestricted (prefill batch x max_rows); inputs with NaN
// or boundary bins wider than the candidate buffers fall back to the frozen
// four-pass select, which handles every distribution.
template<int Threads = 512>
__global__ void qsaRadixTopKP(const float*   scores,
                              const int32_t* lengths,
                              int32_t*       output,
                              int            columns,
                              bool           finite_only) {
    constexpr int K         = 512;
    constexpr int MAX_ITEMS = 16;
    constexpr int CAND      = 2048;
    using SelectedSort      = cub::BlockRadixSort<uint32_t, Threads, 1, uint32_t>;
    using BlockScan         = cub::BlockScan<uint32_t, Threads>;
    __shared__ union SharedOps {
        typename SelectedSort::TempStorage sort;
        typename BlockScan::TempStorage    scan;
    } ops;
    __shared__ uint32_t hist[258];
    __shared__ uint32_t hist2[258];
    __shared__ uint32_t sel_key[K];
    __shared__ uint32_t sel_col[K];
    __shared__ uint32_t cbuf[2][CAND];
    __shared__ uint32_t sh_threshold;
    __shared__ uint32_t sh_ties;
    __shared__ uint32_t sh_count;
    __shared__ uint32_t sh_total;
    __shared__ uint32_t sh_boundary;
    __shared__ int      sh_fallback;

    const int      row      = blockIdx.x;
    const int64_t  length   = lengths[row];
    const int64_t  complete = (length - (length < 0 ? 3 : 0)) / 4;
    const float*   srow     = scores + (int64_t)row * columns;
    const int      items    = (columns + Threads - 1) / Threads;
    const uint32_t base     = threadIdx.x * items;

    uint32_t key[MAX_ITEMS];
    uint8_t  coarse[MAX_ITEMS];
    bool     cand[MAX_ITEMS];
    bool     has_nan = false;
#pragma unroll 4
    for (int i = 0; i < items; ++i) {
        const uint32_t col  = base + (uint32_t)i;
        const uint32_t bits = (col < (uint32_t)columns && col < (uint32_t)complete)
                                  ? __float_as_uint(srow[col]) : 0xff800000u;
        const uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        const bool     is_nan  =
            (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
        key[i]  = is_nan ? 0xffffffffu : ordered;
        // sglang convert_to_uint8: the fp16-rounded, sign-flipped top byte is
        // a weakly monotonic coarse key, so the boundary-bin partition stays
        // order-consistent with the exact fp32 keys.
        const __half   h  = __float2half_rn(__uint_as_float(bits));
        const uint16_t hb = __half_as_ushort(h);
        const uint16_t hk =
            (hb & 0x8000u) ? (uint16_t)~hb : (uint16_t)(hb | 0x8000u);
        coarse[i] = (uint8_t)(hk >> 8);
        cand[i]   = col < (uint32_t)columns;
        has_nan  |= is_nan && cand[i];
    }
    if (threadIdx.x == 0) {
        sh_threshold = 0;
        sh_ties      = 0;
        sh_fallback  = 0;
    }
    __syncthreads();

    // NaN visibility is per-thread (only the owning lane sees it); resolve it
    // block-wide before branching. The selection paths below are full of
    // __syncthreads, so a divergent branch here deadlocks the barrier.
    const bool any_nan = __syncthreads_or(has_nan);
    if (complete <= K) {
        // sglang naive_topk, order-corrected: every complete block fits the
        // 512 budget, so the boundary is the -inf key and the shared
        // selection machinery below fills the remainder with -inf ties.
        uint32_t mine = 0;
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            if (cand[i] && key[i] > 0x007fffffu) ++mine;
        uint32_t base_rank = 0, total = 0;
        BlockScan(ops.scan).ExclusiveSum(mine, base_rank, total);
        __syncthreads();
        if (threadIdx.x == 0) {
            sh_threshold = 0x007fffffu;
            sh_ties      = K > total ? K - total : 0u;
        }
    } else if (any_nan) {
        if (threadIdx.x == 0) sh_fallback = 1;
        __syncthreads();
    } else {
        // Stage 1: one coarse fp16-byte histogram over all candidates.
        for (int j = threadIdx.x; j <= 257; j += Threads) hist[j] = 0;
        __syncthreads();
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            if (cand[i]) atomicAdd(&hist[coarse[i] + 1], 1u);
        __syncthreads();
        // Suffix scan over hist, parallel across the block (sglang run_cumsum):
        // sfx[i] = count(coarse >= i - 1) with the raw counts at bin + 1, so
        // count(coarse >= d) = sfx[d+1] and count(coarse > d) = sfx[d+2].
        {
            uint32_t* cur = hist;
            uint32_t* nxt = hist2;
            for (int s = 0; s < 9; ++s) {
                const int j = 1 << s;
                if (threadIdx.x <= 257)
                    nxt[threadIdx.x] =
                        cur[threadIdx.x]
                        + (threadIdx.x + j <= 257 ? cur[threadIdx.x + j] : 0u);
                __syncthreads();
                uint32_t* swp = cur;
                cur           = nxt;
                nxt           = swp;
            }
            // Boundary: the unique d with count(coarse >= d) >= K >
            // count(coarse >= d + 1); one lane per bin finds it.
            if (threadIdx.x < 256) {
                const uint32_t d = threadIdx.x;
                if (cur[d + 1] >= (uint32_t)K && cur[d + 2] < (uint32_t)K)
                    sh_boundary = d;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t b = sh_boundary;
                sh_threshold = b << 24;
                sh_ties      = (uint32_t)K - cur[b + 2];
                sh_count     = 0;
            }
        }
        __syncthreads();
        // Collect the boundary-bin candidates and fuse the exact-key byte-3
        // histogram into the same pass (sglang stage 1). ties >= 1 always
        // holds at the boundary, so the collection runs unconditionally.
        {
            const uint8_t b3 = (uint8_t)(sh_threshold >> 24);
            uint32_t mine = 0;
#pragma unroll 4
            for (int i = 0; i < items; ++i)
                if (cand[i] && coarse[i] == b3) ++mine;
            const uint32_t start = atomicAdd(&sh_count, mine);
            __syncthreads();
            if (sh_count > CAND) {
                sh_fallback = 1;
            } else {
                for (int j = threadIdx.x; j <= 257; j += Threads) hist[j] = 0;
                __syncthreads();
                uint32_t rank = start;
#pragma unroll 4
                for (int i = 0; i < items; ++i) {
                    if (!cand[i] || coarse[i] != b3) continue;
                    cbuf[0][rank++] = base + (uint32_t)i;
                    atomicAdd(&hist[((key[i] >> 24) & 0xffu) + 1], 1u);
                }
                __syncthreads();
            }
        }
        // Rounds: refine the exact key bytes 3..0 over the candidates only
        // (sglang stage 2). The coarse fp16 byte stays a collection filter;
        // the threshold must end up a full exact key, or keys below the
        // coarse bin but with a larger exact top byte would mis-select.
        for (int round = 0; round < 4 && !sh_fallback; ++round) {
            const uint32_t     shift = (uint32_t)(3 - round) * 8;
            const uint32_t*    buf   = cbuf[round & 1];
            const uint32_t    rem   = sh_ties;
            // Parallel suffix scan over hist (same layout as stage 1).
            uint32_t* cur = hist;
            uint32_t* nxt = hist2;
            for (int s = 0; s < 9; ++s) {
                const int j = 1 << s;
                if (threadIdx.x <= 257)
                    nxt[threadIdx.x] =
                        cur[threadIdx.x]
                        + (threadIdx.x + j <= 257 ? cur[threadIdx.x + j] : 0u);
                __syncthreads();
                uint32_t* swp = cur;
                cur           = nxt;
                nxt           = swp;
            }
            if (threadIdx.x < 256) {
                const uint32_t d = threadIdx.x;
                if (cur[d + 1] >= rem && cur[d + 2] < rem) sh_boundary = d;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t b = sh_boundary;
                if (round == 0)
                    sh_threshold = b << 24;
                else
                    sh_threshold |= b << shift;
                sh_ties  = rem - cur[b + 2];
                sh_total = sh_count;
                sh_count = 0;
            }
            __syncthreads();
            if (round == 3) break;
            for (int j = threadIdx.x; j <= 257; j += Threads) hist[j] = 0;
            __syncthreads();
            const uint8_t  b = (uint8_t)((sh_threshold >> shift) & 0xffu);
            const uint32_t n = sh_total;
            uint32_t mine = 0;
            for (uint32_t j = threadIdx.x; j < n; j += Threads) {
                const uint32_t col = buf[j];
                const uint32_t bits = (col < (uint32_t)complete)
                                          ? __float_as_uint(srow[col]) : 0xff800000u;
                const uint32_t k =
                    (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
                if (((k >> shift) & 0xffu) == b) ++mine;
            }
            const uint32_t start = atomicAdd(&sh_count, mine);
            __syncthreads();
            if (sh_count > CAND) {
                sh_fallback = 1;
                break;
            }
            uint32_t rank = start;
            for (uint32_t j = threadIdx.x; j < n; j += Threads) {
                const uint32_t col = buf[j];
                const uint32_t bits = (col < (uint32_t)complete)
                                          ? __float_as_uint(srow[col]) : 0xff800000u;
                const uint32_t k =
                    (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
                if (((k >> shift) & 0xffu) != b) continue;
                cbuf[(round + 1) & 1][rank++] = col;
                atomicAdd(&hist[((k >> (shift - 8)) & 0xffu) + 1], 1u);
            }
            __syncthreads();
        }
    }
    if (sh_fallback) {
        // Frozen qsaRadixTopK select (four full passes): the fallback for
        // NaN inputs and boundary bins wider than the candidate buffers.
        uint32_t threshold = 0;
        int      remaining = K;
        for (int p = 3; p >= 0; --p) {
            for (int j = threadIdx.x; j <= 257; j += Threads) hist[j] = 0;
            __syncthreads();
            const uint32_t shift = (uint32_t)p * 8;
#pragma unroll 4
            for (int i = 0; i < items; ++i)
                if (cand[i]) atomicAdd(&hist[((key[i] >> shift) & 0xffu) + 1], 1u);
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t ge_above = 0;
                int      boundary = 0;
                for (int d = 255; d >= 0; --d) {
                    const uint32_t ge_d = ge_above + hist[d + 1];
                    if (ge_d >= (uint32_t)remaining) { boundary = d; break; }
                    ge_above = ge_d;
                }
                remaining -= (int)ge_above;
                threshold  = (threshold << 8) | (uint32_t)boundary;
                sh_boundary = (uint32_t)boundary;
            }
            __syncthreads();
#pragma unroll 4
            for (int i = 0; i < items; ++i)
                cand[i] = cand[i] && (((key[i] >> shift) & 0xffu) == sh_boundary);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sh_threshold = threshold;
            sh_ties      = (uint32_t)remaining;
        }
        __syncthreads();
    }
    __syncthreads();
    const uint32_t thr = sh_threshold;

    // Trailing slots: rows with fewer than 512 selectable blocks must emit -1
    // at the tail, exactly like the native expansion.
    for (int j = threadIdx.x; j < K; j += Threads) {
        sel_key[j] = 0u;
        sel_col[j] = (uint32_t)columns;
    }
    __syncthreads();

    // Selection: strictly above the threshold plus the first `ties`
    // threshold-valued columns in column order. Native topk gathers the
    // above-threshold items in column order and appends the boundary ties,
    // then value-sorts stably: the zero group (+0.0 and -0.0 share one value
    // but hold two radix keys) needs that exact compaction order, so above
    // items compact first and tie items second, and the sort key below
    // collapses -0.0 onto +0.0.
    {
        uint32_t tie_n = 0;
#pragma unroll 4
        for (int i = 0; i < items; ++i) {
            const uint32_t col = base + (uint32_t)i;
            if (col < (uint32_t)columns && key[i] == thr)
                ++tie_n;
        }
        uint32_t tie_base = 0;
        BlockScan(ops.scan).ExclusiveSum(tie_n, tie_base);
        __syncthreads();
        bool kept_above[MAX_ITEMS];
        bool kept_tie[MAX_ITEMS];
        {
            uint32_t tie_rank = tie_base;
#pragma unroll 4
            for (int i = 0; i < items; ++i) {
                const uint32_t col = base + (uint32_t)i;
                kept_above[i] = false;
                kept_tie[i] = false;
                if (col < (uint32_t)columns) {
                    if (key[i] > thr)
                        kept_above[i] = true;
                    else if (key[i] == thr) {
                        kept_tie[i] = tie_rank < sh_ties;
                        ++tie_rank;
                    }
                }
            }
        }
        uint32_t above_n = 0;
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            if (kept_above[i])
                ++above_n;
        uint32_t above_base = 0, above_total = 0;
        BlockScan(ops.scan).ExclusiveSum(above_n, above_base, above_total);
        __syncthreads();
        uint32_t tie2_n = 0;
#pragma unroll 4
        for (int i = 0; i < items; ++i)
            if (kept_tie[i])
                ++tie2_n;
        uint32_t tie2_base = 0;
        BlockScan(ops.scan).ExclusiveSum(tie2_n, tie2_base);
        __syncthreads();
        {
            uint32_t rank = above_base;
#pragma unroll 4
            for (int i = 0; i < items; ++i) {
                if (!kept_above[i])
                    continue;
                sel_key[rank] = key[i];
                sel_col[rank] = base + (uint32_t)i;
                ++rank;
            }
            rank = above_total + tie2_base;
#pragma unroll 4
            for (int i = 0; i < items; ++i) {
                if (!kept_tie[i])
                    continue;
                sel_key[rank] = key[i];
                sel_col[rank] = base + (uint32_t)i;
                ++rank;
            }
        }
        __syncthreads();
    }

    // Stable descending sort by key; the compaction order (above items first,
    // then tie items, each in column order) is preserved for ties, matching
    // the frozen native TopK order. The sort key collapses -0.0 onto +0.0:
    // natively they compare equal, so the stable sort keeps the compaction
    // order across the whole zero group.
    {
        uint32_t s_key[1], s_col[1];
        s_key[0] = sel_key[threadIdx.x] == 0x7fffffffu ? 0x80000000u
                                                       : sel_key[threadIdx.x];
        s_col[0] = sel_col[threadIdx.x];
        SelectedSort(ops.sort).SortDescending(s_key, s_col);
        __syncthreads();

        // If a NaN was selected, re-sort by its raw radix bits so NaN payloads
        // sit exactly where the native kernel places them.
        {
            const uint32_t bits = (s_col[0] < (uint32_t)columns && s_col[0] < (uint32_t)complete)
                                      ? __float_as_uint(srow[s_col[0]])
                                      : 0xff800000u;
            const bool nan_selected =
                (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
            if (__syncthreads_or(nan_selected)) {
                const uint32_t raw_ordered =
                    (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
                sel_key[threadIdx.x] = raw_ordered;
                sel_col[threadIdx.x] = s_col[0];
                __syncthreads();
                uint32_t r_key[1], r_col[1];
                r_key[0] = sel_key[threadIdx.x];
                r_col[0] = sel_col[threadIdx.x];
                SelectedSort(ops.sort).SortDescending(r_key, r_col);
                __syncthreads();
                s_key[0] = r_key[0];
                s_col[0] = r_col[0];
            }
            __syncthreads();

            // Emit: one thread per selected rank, token expansion plus the tail.
            const uint32_t bits2 = (s_col[0] < (uint32_t)columns && s_col[0] < (uint32_t)complete)
                                        ? __float_as_uint(srow[s_col[0]])
                                        : 0xff800000u;
            const bool valid = ((bits2 & 0x7f800000u) != 0x7f800000u
                                || (!finite_only && bits2 == 0x7f800000u))
                               && s_col[0] < (uint32_t)columns
                               && s_col[0] < (uint32_t)complete;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[row * 2051 + threadIdx.x * 4 + j] =
                    valid ? (int32_t)(s_col[0] * 4 + j) : -1;
        }
        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < 3; ++j)
                output[row * 2051 + 2048 + j] =
                    j < length - complete * 4 ? (int32_t)(complete * 4 + j) : -1;
        }
    }
}
template<int N, int Threads, bool UseMerge = false, bool DraftPacked = false>
__global__ void qsaCubStableTopK(const float*   scores,
                                 const int32_t* lengths,
                                 int32_t*       output,
                                 int            columns,
                                 bool           finite_only,
                                 const int32_t* draft_lengths = nullptr,
                                 const int32_t* draft_cu      = nullptr,
                                 int            draft_batch   = 0) {
    constexpr int Items = N / Threads;
    using FullSort      = std::conditional_t<UseMerge,
                                             cub::BlockMergeSort<uint32_t, Threads, Items, uint32_t>,
                                             cub::BlockRadixSort<uint32_t, Threads, Items, uint32_t>>;
    using SelectedSort  = cub::BlockRadixSort<uint32_t, Threads, 512 / Threads, uint32_t>;
    __shared__ union SharedSort {
        typename FullSort::TempStorage     full;
        typename SelectedSort::TempStorage selected;
    } storage;
    __shared__ uint32_t chosen[512];
    __shared__ uint32_t cutoff;
    const int           row        = blockIdx.x;
    int                 output_row = row;
    if constexpr (DraftPacked) {
        const int request = row / 4, lane = row % 4, count = draft_lengths[request];
        if (lane < count) {
            output_row = draft_cu[request] + lane;
        } else {
            output_row = draft_cu[draft_batch] + request * 4 - draft_cu[request] + lane - count;
            for (int j = threadIdx.x; j < 2051; j += Threads)
                output[output_row * 2051 + j] = -1;
            return;
        }
    }

    const int64_t length   = lengths[row];
    const int64_t complete = (length - (length < 0 ? 3 : 0)) / 4;
    uint32_t      key[Items], index[Items];
    bool          special = false;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        uint32_t col = threadIdx.x * Items + i;
        uint32_t bits =
            (col < columns && col < complete) ?
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u) :
                0xff800000u;
        uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        bool     isnan   = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
        special |= isnan || bits == 0x80000000u;
        key[i]   = isnan ? 0xffffffffu : ordered;
        index[i] = col;
    }
    bool any_special = __syncthreads_or(special);
    if constexpr (UseMerge) {
        FullSort(storage.full).StableSort(key, index, QsaGreater{});
    } else {
        FullSort(storage.full).SortDescending(key, index);
    }
    if (any_special) {
        // Torch first gathers values strictly above the selected radix, then
        // its equal boundary, in column order. Its float radix sort treats
        // signed zeros as equal. Recreate that order only for NaNs/-0.

#pragma unroll
        for (int i = 0; i < Items; ++i) {
            int rank = threadIdx.x * Items + i;
            if (rank < 512)
                chosen[rank] = index[i];
            if (rank == 511)
                cutoff = key[i];
        }
        __syncthreads();
        uint32_t raw_key[512 / Threads], raw_index[512 / Threads];
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = chosen[rank];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            uint32_t ordered   = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
            bool     isnan     = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
            uint32_t selection = isnan ? 0xffffffffu : ordered;
            raw_key[i]         = 2 * N - ((selection == cutoff ? N : 0) + col);
            raw_index[i]       = col;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t col = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            // CUB float keys canonicalize -0/+0; payloads retain original IDs.
            if ((bits & 0x7fffffffu) == 0)
                bits = 0;
            raw_key[i] = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u)) && col < columns
                         && col < complete;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
        }
    } else {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            uint32_t rank = threadIdx.x * Items + i;
            if (rank < 512) {
                uint32_t col = index[i];
                uint32_t bits =
                    (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
                bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u))
                             && col < columns && col < complete;
                if constexpr (N == 1024 && UseMerge) {
                    chosen[rank] = valid ? col : 0xffffffffu;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
                }
            }
        }
        if constexpr (N == 1024 && UseMerge) {
            // Sorted ranks are blocked across threads. Stage the selected
            // columns once, then let adjacent lanes write adjacent tokens.
            __syncthreads();
            for (int token = threadIdx.x; token < 2048; token += Threads) {
                const int col = static_cast<int>(chosen[token / 4]);
                output[output_row * 2051 + token] = col >= 0 ? col * 4 + token % 4 : -1;
            }
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            output[output_row * 2051 + 2048 + j] = j < length - complete * 4 ? complete * 4 + j : -1;
    }
}
}  // namespace rtp_llm

void qsa_cub_topk(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    TORCH_CHECK(scores.size(0) == lengths.numel() && output.size(0) == scores.size(0) && output.size(1) == 2051);
    TORCH_CHECK(scores.size(1) == 1023 || scores.size(1) == 1024 || scores.size(1) == 2047 || scores.size(1) == 2048
                || scores.size(1) == 2144);
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    const int            rows = scores.size(0), columns = scores.size(1);
    if (!rows)
        return;
    // A 4K decode graph reserves 1023 compressed columns. Sort the bounded
    // row directly and expand in the same launch, preserving native tie order.
    if (merge && columns <= 1024 && threads == 512) {
        rtp_llm::qsaCubStableTopK<1024, 512, true><<<rows, 512, 0, stream>>>(
            scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), columns, finite_only);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    // Measured 8K domains only. StableSort preserves equal-key column order;
    // the special-value second pass below remains the pinned Torch radix order.
    if (merge && (columns == 2047 || columns == 2048 || columns == 2144)) {
        if (columns == 2144) {
            rtp_llm::qsaCubStableTopK<2304, 256, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        } else {
            rtp_llm::qsaCubStableTopK<2048, 512, true><<<rows, 512, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
#define LAUNCH(N, T)                                                                                                   \
    rtp_llm::qsaCubStableTopK<N, T><<<rows, T, 0, stream>>>(                                                           \
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(), columns, finite_only)
    if (columns <= 1024) {
        if (threads == 128) {
            LAUNCH(1024, 128);
        } else if (threads == 256) {
            LAUNCH(1024, 256);
        } else {
            LAUNCH(1024, 512);
        }
    } else if (columns == 2144) {
        if (threads == 128) {
            LAUNCH(2176, 128);
        } else if (threads == 256) {
            LAUNCH(2304, 256);
        } else {
            LAUNCH(2560, 512);
        }
    } else {
        if (threads == 128) {
            LAUNCH(2048, 128);
        } else if (threads == 256) {
            LAUNCH(2048, 256);
        } else {
            LAUNCH(2048, 512);
        }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_cub_topk_draft(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() && row_lengths.is_cuda() && cu.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt && row_lengths.scalar_type() == at::kInt
                && cu.scalar_type() == at::kInt);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous()
                && row_lengths.is_contiguous() && cu.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device()
                && scores.device() == row_lengths.device() && scores.device() == cu.device());
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && row_lengths.dim() == 1 && cu.dim() == 1
                && output.dim() == 2);
    int rows = scores.size(0), cols = scores.size(1), batch = row_lengths.numel();
    TORCH_CHECK(rows == batch * 4 && batch >= 1 && batch <= 8 && lengths.numel() == rows && output.size(0) == rows
                && output.size(1) == 2051 && cu.numel() == batch + 1 && cols == 2144);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::qsaCubStableTopK<2304, 256, true, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                               lengths.data_ptr<int32_t>(),
                                                                               output.data_ptr<int32_t>(),
                                                                               cols,
                                                                               true,
                                                                               row_lengths.data_ptr<int32_t>(),
                                                                               cu.data_ptr<int32_t>(),
                                                                               batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_radix_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    const int rows = scores.size(0), columns = scores.size(1);
    TORCH_CHECK(rows >= 1 && rows <= 32, "radix topk expects decode row counts");
    TORCH_CHECK(columns >= 512 && columns <= 8192, "radix topk expects 512..8192 score columns");
    TORCH_CHECK(lengths.numel() == rows && output.size(0) == rows && output.size(1) == 2051);
    if (!rows)
        return;
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::qsaRadixTopK<512><<<rows, 512, 0, stream>>>(
        scores.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        columns,
        finite_only);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_radix_topk_p(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    const int rows = scores.size(0), columns = scores.size(1);
    TORCH_CHECK(rows >= 1 && rows <= 65536, "prefill radix topk expects 1..65536 rows");
    TORCH_CHECK(columns >= 512 && columns <= 8192, "prefill radix topk expects 512..8192 score columns");
    TORCH_CHECK(lengths.numel() == rows && output.size(0) == rows && output.size(1) == 2051);
    if (!rows)
        return;
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::qsaRadixTopKP<512><<<rows, 512, 0, stream>>>(
        scores.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        columns,
        finite_only);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def is_supported(scores, lengths):
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 64
        # The 8K MTP draft graph reserves three additional 128-token pages:
        # (64 + 3) * 32 compressed columns. Preserve the full-width Torch order.
        and scores.shape[1] in (1023, 1024, 2047, 2048, 2144)
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and (
            (
                torch.cuda.get_device_capability(scores.device) == (12, 0)
                and scores.shape[0] <= 32
                and scores.shape[1] in (2047, 2048, 2144)
            )
            or (
                torch.cuda.get_device_capability(scores.device) == (9, 0)
                and scores.shape[1] in (1023, 1024)
            )
        )
    )


def _load_module(device):
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _BUILD_FAILED:
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        with _BUILD_LOCK:
            if _MODULE is not None:
                return _MODULE
            if _BUILD_FAILED:
                return None
            try:
                from torch.utils.cpp_extension import load_inline

                major, minor = torch.cuda.get_device_capability(device)
                arch = f"{major}{minor}"
                flags = [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                    # The runtime image resolves a CUDA 13.0 cudart under an
                    # nvcc 13.2 toolchain; the CCCL compatibility guard rejects
                    # that pair even though the kernels only use stable CUB /
                    # runtime APIs. NVIDIA documents this define as the escape
                    # hatch for mismatched compiler/toolkit pairs.
                    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                ]
                identity = hashlib.sha256(
                    (_CUDA_SOURCE + _TORCH_GIT + repr(flags)).encode()
                ).hexdigest()[:16]
                _MODULE = load_inline(
                    name="qwen4_qsa_topk_" + identity,
                    cpp_sources="void qsa_cub_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge); void qsa_cub_topk_draft(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu); void qsa_radix_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only); void qsa_radix_topk_p(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only);",
                    cuda_sources=_CUDA_SOURCE,
                    functions=[
                        "qsa_cub_topk",
                        "qsa_cub_topk_draft",
                        "qsa_radix_topk",
                        "qsa_radix_topk_p",
                    ],
                    extra_cflags=["-O2"],
                    extra_cuda_cflags=flags,
                    with_cuda=True,
                    verbose=False,
                )
            except Exception:
                _BUILD_FAILED = True
                _LOGGER.warning(
                    "Qwen4 QSA CUB compilation unavailable; using native selection",
                    exc_info=True,
                )
            return _MODULE


def try_cub_qsa_tokens(scores, lengths, *, finite_only=True):
    if not is_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk(
        scores,
        lengths,
        output,
        512 if scores.shape[1] <= 1024 else 256,
        finite_only,
        True,
    )
    return output


def try_cub_draft_qsa_tokens(scores, visible, row_lengths, cu):
    """Select directly into packed draft order; metadata is owner-validated.

    Fixed four-lane input rows and packed output rows have a bijection even
    for empty requests. Inactive rows fill the output padding with -1.
    """
    if not (
        is_supported(scores, visible)
        and scores.shape[1] == 2144
        and row_lengths.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and scores.shape[0] == 4 * row_lengths.numel()
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32
            and t.device == scores.device
            and t.is_contiguous()
            and not t.requires_grad
            for t in (row_lengths, cu)
        )
    ):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk_draft(scores, visible, output, row_lengths, cu)
    return output


def radix_supported(scores, lengths):
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 32
        and 1024 <= scores.shape[1] <= 8192
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and torch.cuda.get_device_capability(scores.device) == (9, 0)
    )


def try_radix_qsa_tokens(scores, lengths, *, finite_only=True):
    """O(N) radix-select top-512 for SM90 L20X decode shapes.

    Replaces the native sbtopk + radixSortKVInPlace chain (measured 42-58us
    per call on L20X) with one block-per-row kernel: 4 passes of 8-bit
    shared-memory histograms select the native top-512 boundary, a stable
    CUB block sort restores the exact native order, and the token expansion
    runs in the same launch.
    """
    if not radix_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_radix_topk(scores, lengths, output, finite_only)
    return output


def prefill_radix_supported(scores, lengths):
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 32 < scores.shape[0] <= 65536
        and 1024 <= scores.shape[1] <= 8192
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and torch.cuda.get_device_capability(scores.device) == (9, 0)
    )


def try_prefill_radix_qsa_tokens(scores, lengths, *, finite_only=True):
    """Ported sglang v0.5.22 fast-topk select for SM90 prefill rows.

    One coarse fp16-truncated byte pass plus candidate-only refinement rounds
    replace the four full histogram passes, and rows are unrestricted (grid
    = rows). The output contract (frozen native order, ties by ascending
    column, 2051-slot expansion) matches try_radix_qsa_tokens; NaN inputs and
    boundary bins wider than the candidate buffers fall back to the frozen
    four-pass select inside the kernel.
    """
    if not prefill_radix_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_radix_topk_p(scores, lengths, output, finite_only)
    return output
