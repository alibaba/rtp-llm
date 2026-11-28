#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>

namespace rtp_llm {
namespace fast_renorm {

// 256 threads/CTA keeps per-CTA SMEM < 16 KB so 6+ CTAs/SM co-reside in a single wave.
// Piecewise Round-1 mapping over positive IEEE-754 float32 bits [0x00000001 .. 0x7f800000]:
//   - b1 = 0:          [0x00000001 .. 0x317fffff] (< 2^-28)         -> shift=20, then 10, then 0
//   - b1 in [1..928]:  [0x31800000 .. 0x3fffffff] ([2^-28, 2.0))    -> shift=8 (256 ULP/bin), then shift=0 (1-ULP)
//   - b1 in [929..960]:[0x40000000 .. 0x7f800000] ([2.0, +inf])     -> shift=15, then 5, then 0
constexpr int      kFastRenormThreads   = 256;
constexpr int      kR1Bins              = 961;
constexpr int      kR2Bins              = 1024;
constexpr int      kR3Bins              = 1024;
constexpr int      kPartHistStride      = 1024;  // R1 uses [0..960], Range/Leaf use [0..1023]
constexpr int      kFastRenormSortItems = 12;
constexpr uint32_t kFastRenormMaxCand   = kFastRenormThreads * kFastRenormSortItems;  // 3072
constexpr uint32_t kMultiCTABatchMax    = 256;
constexpr uint32_t kMultiCTAVocabMin    = 16384;
constexpr uint32_t kSampleTileSize      = 128;  // 4 warps x 32 lanes (or 1 warp x float4)
constexpr uint32_t kPickMaxChunks       = 64;   // 64 * 256 * 128 = 2,097,152 max vocab

enum class RenormCutoffMode : uint8_t {
    kTopPOnly        = 0,
    kTopKOnly        = 1,
    kBoth            = 2,
    kPickMaxInBucket = 3,
};

struct alignas(16) MultiRowState {
    double   eff_p_d;
    double   above_sum_d;
    float    row_sum;
    float    kept_sum;
    float    cutoff;
    uint32_t pos_cnt;
    uint32_t k;
    uint32_t rem_k;
    uint32_t range_lo;
    uint32_t range_hi;
    uint32_t cnt_target;
    uint32_t cand_cnt;
    int32_t  b_target;
    int32_t  shift_cur;
    uint8_t  mode;
    uint8_t  early_done;
    uint8_t  done_cutoff;
};

struct FastRenormWorkspaceViews {
    MultiRowState* states;
    float*         part_row_sum;
    uint32_t*      part_pos_cnt;
    float*         part_above_sum;
    float*         part_hist_sum;
    uint32_t*      part_hist_cnt;
    uint32_t*      cand_bits;
    float*         tile_sums;
};

struct IntMaxOp {
    __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const {
        return a > b ? a : b;
    }
};

inline size_t alignUp128(size_t n) {
    return (n + 127ull) & ~127ull;
}

__host__ __device__ __forceinline__ uint32_t fastChunkElems(uint32_t vocab_size, uint32_t c_per_row) {
    const uint32_t raw = (vocab_size + c_per_row - 1u) / c_per_row;
    return (raw + kSampleTileSize - 1u) & ~(kSampleTileSize - 1u);
}

inline uint32_t FastRenormCTAsPerRow(uint32_t batch_size, uint32_t vocab_size) {
    if (batch_size == 0 || batch_size > kMultiCTABatchMax || vocab_size <= kMultiCTAVocabMin) {
        return 1u;
    }
    uint32_t c = std::min<uint32_t>(16u, std::max<uint32_t>(2u, 512u / batch_size));
    while (c > 1u) {
        const uint32_t chunk_elems = fastChunkElems(vocab_size, c);
        if (static_cast<uint64_t>(chunk_elems) * (c - 1u) < vocab_size) {
            break;
        }
        --c;
    }
    return c;
}

inline bool FastSampleSupportsVocab(uint32_t vocab_size) {
    const size_t num_tiles  = (static_cast<size_t>(vocab_size) + kSampleTileSize - 1ull) / kSampleTileSize;
    const size_t num_chunks = (num_tiles + kFastRenormThreads - 1ull) / kFastRenormThreads;
    return vocab_size > 0u && num_chunks <= kPickMaxChunks;
}

inline size_t FastRenormWorkspaceBytes(uint32_t batch_size, uint32_t vocab_size) {
    const uint32_t c = FastRenormCTAsPerRow(batch_size, vocab_size);
    if (c <= 1u) {
        return 0u;
    }
    const size_t b         = batch_size;
    const size_t bc        = static_cast<size_t>(batch_size) * c;
    const size_t num_tiles = (static_cast<size_t>(vocab_size) + kSampleTileSize - 1ull) / kSampleTileSize;
    size_t       off       = 0;
    off += alignUp128(b * sizeof(MultiRowState));
    off += alignUp128(bc * sizeof(float));                         // part_row_sum
    off += alignUp128(bc * sizeof(uint32_t));                      // part_pos_cnt
    off += alignUp128(bc * sizeof(float));                         // part_above_sum
    off += alignUp128(bc * kPartHistStride * sizeof(float));       // part_hist_sum
    off += alignUp128(bc * kPartHistStride * sizeof(uint32_t));    // part_hist_cnt
    off += alignUp128(b * kFastRenormMaxCand * sizeof(uint32_t));  // cand_bits
    off += alignUp128(b * num_tiles * sizeof(float));              // tile_sums
    return off;
}

inline FastRenormWorkspaceViews makeWorkspaceViews(void* workspace, uint32_t batch_size, uint32_t c) {
    FastRenormWorkspaceViews v{};
    uint8_t*                 base = reinterpret_cast<uint8_t*>(workspace);
    const size_t             b    = batch_size;
    const size_t             bc   = static_cast<size_t>(batch_size) * c;
    size_t                   off  = 0;
    v.states                      = reinterpret_cast<MultiRowState*>(base + off);
    off += alignUp128(b * sizeof(MultiRowState));
    v.part_row_sum = reinterpret_cast<float*>(base + off);
    off += alignUp128(bc * sizeof(float));
    v.part_pos_cnt = reinterpret_cast<uint32_t*>(base + off);
    off += alignUp128(bc * sizeof(uint32_t));
    v.part_above_sum = reinterpret_cast<float*>(base + off);
    off += alignUp128(bc * sizeof(float));
    v.part_hist_sum = reinterpret_cast<float*>(base + off);
    off += alignUp128(bc * kPartHistStride * sizeof(float));
    v.part_hist_cnt = reinterpret_cast<uint32_t*>(base + off);
    off += alignUp128(bc * kPartHistStride * sizeof(uint32_t));
    v.cand_bits = reinterpret_cast<uint32_t*>(base + off);
    off += alignUp128(b * kFastRenormMaxCand * sizeof(uint32_t));
    v.tile_sums = reinterpret_cast<float*>(base + off);
    return v;
}

__device__ __forceinline__ uint32_t r1BucketOfBits(uint32_t bits) {
    if (bits < 0x31800000u) {
        return 0u;
    }
    if (bits < 0x40000000u) {
        return 1u + ((bits - 0x31800000u) >> 18);
    }
    return 929u + ((bits - 0x40000000u) >> 25);
}

__device__ __forceinline__ void r1RangeForBucket(uint32_t b1, uint32_t& out_lo, uint32_t& out_hi) {
    if (b1 == 0u) {
        out_lo = 1u;
        out_hi = 0x317fffffu;
    } else if (b1 < 929u) {
        out_lo = 0x31800000u + ((b1 - 1u) << 18);
        out_hi = out_lo + 0x3ffffu;
    } else {
        out_lo = 0x40000000u + ((b1 - 929u) << 25);
        out_hi = (b1 < 960u) ? (out_lo + 0x1ffffffu) : 0x7f800000u;
    }
}

__device__ __forceinline__ int32_t shiftForRange(uint32_t lo, uint32_t hi) {
    const uint32_t span = hi - lo;
    if (span < static_cast<uint32_t>(kR2Bins)) {
        return 0;
    }
    return (32 - __clz(span)) - 10;
}

__device__ __forceinline__ void refineSubRange(
    uint32_t lo, uint32_t hi, int32_t shift, uint32_t b, uint32_t& out_lo, uint32_t& out_hi, int32_t& out_shift) {
    const uint32_t step = 1u << shift;
    out_lo              = lo + (b << shift);
    out_hi              = min(hi, out_lo + (step - 1u));
    out_shift           = shiftForRange(out_lo, out_hi);
}

template<int NUM_BINS>
__device__ __forceinline__ void histAccumSubBank2(float (&sum_hist)[NUM_BINS][2],
                                                  uint32_t (&cnt_hist)[NUM_BINS][2],
                                                  int      tx,
                                                  bool     active,
                                                  uint32_t bucket,
                                                  float    val) {
    const unsigned mask_pos = __ballot_sync(0xffffffffu, active);
    if (mask_pos) {
        const int first_lane = __ffs(mask_pos) - 1;
        const int b0         = __shfl_sync(0xffffffffu, static_cast<int>(bucket), first_lane);
        if (__all_sync(0xffffffffu, !active || static_cast<int>(bucket) == b0)) {
            float ws = active ? val : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ws += __shfl_down_sync(0xffffffffu, ws, off);
            }
            if ((tx & 31) == 0) {
                atomicAdd(&sum_hist[b0][0], ws);
                atomicAdd(&cnt_hist[b0][0], static_cast<uint32_t>(__popc(mask_pos)));
            }
        } else if (active) {
            const int sub = tx & 1;
            atomicAdd(&sum_hist[bucket][sub], val);
            atomicAdd(&cnt_hist[bucket][sub], 1u);
        }
    }
}

template<int NUM_BINS>
__device__ __forceinline__ void histAccumSingleBank(
    float (&sum_hist)[NUM_BINS], uint32_t (&cnt_hist)[NUM_BINS], int tx, bool active, uint32_t bucket, float val) {
    const unsigned mask_pos = __ballot_sync(0xffffffffu, active);
    if (mask_pos) {
        const int first_lane = __ffs(mask_pos) - 1;
        const int b0         = __shfl_sync(0xffffffffu, static_cast<int>(bucket), first_lane);
        if (__all_sync(0xffffffffu, !active || static_cast<int>(bucket) == b0)) {
            float ws = active ? val : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ws += __shfl_down_sync(0xffffffffu, ws, off);
            }
            if ((tx & 31) == 0) {
                atomicAdd(&sum_hist[b0], ws);
                atomicAdd(&cnt_hist[b0], static_cast<uint32_t>(__popc(mask_pos)));
            }
        } else if (active) {
            atomicAdd(&sum_hist[bucket], val);
            atomicAdd(&cnt_hist[bucket], 1u);
        }
    }
}

template<int NUM_BINS>
__device__ __forceinline__ void
histAccumCountSubBank2(uint32_t (&cnt_hist)[NUM_BINS][2], int tx, bool active, uint32_t bucket) {
    const unsigned mask_pos = __ballot_sync(0xffffffffu, active);
    if (mask_pos) {
        const int first_lane = __ffs(mask_pos) - 1;
        const int b0         = __shfl_sync(0xffffffffu, static_cast<int>(bucket), first_lane);
        if (__all_sync(0xffffffffu, !active || static_cast<int>(bucket) == b0)) {
            if ((tx & 31) == 0) {
                atomicAdd(&cnt_hist[b0][0], static_cast<uint32_t>(__popc(mask_pos)));
            }
        } else if (active) {
            atomicAdd(&cnt_hist[bucket][tx & 1], 1u);
        }
    }
}

struct alignas(16) ScanControl {
    double   d_scan_prefix[kFastRenormThreads + 1];
    uint32_t c_scan_prefix[kFastRenormThreads + 1];
    int32_t  b_first;
    int32_t  b_last;
    int32_t  b_k;
    int32_t  b_p;
    int32_t  b_target;
    uint32_t cnt_first;
    uint32_t cnt_last;
    uint32_t cnt_k;
    uint32_t cnt_p;
    uint32_t cnt_target;
    uint32_t above_cnt_last;
    uint32_t above_cnt_k;
    uint32_t above_cnt_p;
    uint32_t rem_k_out;
    uint8_t  mode_out;
};

__device__ __forceinline__ void resolveTargetBucket(
    ScanControl& sc, bool check_k, bool check_p, double target_p_d, uint32_t rem_k, bool fallback_p_to_last) {
    const int b_first = sc.b_first;
    const int b_last  = sc.b_last;

    int      eff_b_p       = fallback_p_to_last ? b_last : -1;
    uint32_t eff_cnt_p     = fallback_p_to_last ? sc.cnt_last : 0u;
    uint32_t eff_above_p   = fallback_p_to_last ? sc.above_cnt_last : 0u;
    bool     p_is_max_only = false;

    if (check_p) {
        if (target_p_d <= 0.0) {
            eff_b_p       = b_first;
            eff_cnt_p     = sc.cnt_first;
            eff_above_p   = 0u;
            p_is_max_only = true;
        } else if (sc.b_p >= 0) {
            eff_b_p     = sc.b_p;
            eff_cnt_p   = sc.cnt_p;
            eff_above_p = sc.above_cnt_p;
        }
    }

    if (!check_k) {
        sc.b_target   = eff_b_p;
        sc.cnt_target = eff_cnt_p;
        sc.rem_k_out  = rem_k;
        sc.mode_out =
            static_cast<uint8_t>(p_is_max_only ? RenormCutoffMode::kPickMaxInBucket : RenormCutoffMode::kTopPOnly);
    } else if (!check_p || eff_b_p < 0) {
        sc.b_target   = sc.b_k;
        sc.cnt_target = sc.cnt_k;
        sc.rem_k_out  = rem_k - sc.above_cnt_k;
        sc.mode_out   = static_cast<uint8_t>(RenormCutoffMode::kTopKOnly);
    } else {
        const int b_k = sc.b_k;
        if (b_k > eff_b_p) {
            sc.b_target   = b_k;
            sc.cnt_target = sc.cnt_k;
            sc.rem_k_out  = rem_k - sc.above_cnt_k;
            sc.mode_out   = static_cast<uint8_t>(RenormCutoffMode::kTopKOnly);
        } else if (eff_b_p > b_k) {
            sc.b_target   = eff_b_p;
            sc.cnt_target = eff_cnt_p;
            sc.rem_k_out  = (rem_k > eff_above_p) ? (rem_k - eff_above_p) : 1u;
            sc.mode_out =
                static_cast<uint8_t>(p_is_max_only ? RenormCutoffMode::kPickMaxInBucket : RenormCutoffMode::kTopPOnly);
        } else {
            sc.b_target   = b_k;
            sc.cnt_target = sc.cnt_k;
            sc.rem_k_out  = rem_k - sc.above_cnt_k;
            sc.mode_out =
                static_cast<uint8_t>(p_is_max_only ? RenormCutoffMode::kPickMaxInBucket : RenormCutoffMode::kBoth);
        }
    }
}

template<int ITEMS_PER_THREAD, int TOTAL_BINS>
__device__ __forceinline__ double
scanBucketsFP64(ScanControl&                                                        sc,
                typename cub::BlockScan<double, kFastRenormThreads>::TempStorage&   scan_d_smem,
                typename cub::BlockScan<uint32_t, kFastRenormThreads>::TempStorage& scan_u32_smem,
                const double (&bs)[ITEMS_PER_THREAD],
                const uint32_t (&bc)[ITEMS_PER_THREAD],
                int      tx,
                bool     check_k,
                bool     check_p,
                double   target_p_d,
                uint32_t rem_k,
                bool     fallback_p_to_last) {
    double   local_s  = 0.0;
    uint32_t local_c  = 0u;
    int      hi_local = -1;
    int      lo_local = TOTAL_BINS;

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        const int b = (TOTAL_BINS - 1) - (tx * ITEMS_PER_THREAD + j);
        local_s += bs[j];
        local_c += bc[j];
        if (b >= 0 && bc[j] > 0u) {
            if (hi_local < 0) {
                hi_local = b;
            }
            lo_local = b;
        }
    }

    double thread_incl_s = 0.0;
    cub::BlockScan<double, kFastRenormThreads>(scan_d_smem).InclusiveSum(local_s, thread_incl_s);
    __syncthreads();

    uint32_t thread_incl_c = 0u;
    cub::BlockScan<uint32_t, kFastRenormThreads>(scan_u32_smem).InclusiveSum(local_c, thread_incl_c);
    if (tx == 0) {
        sc.d_scan_prefix[0] = 0.0;
        sc.c_scan_prefix[0] = 0u;
        sc.b_first          = -1;
        sc.b_last           = TOTAL_BINS;
        sc.b_k              = -1;
        sc.b_p              = -1;
    }
    sc.d_scan_prefix[tx + 1] = thread_incl_s;
    sc.c_scan_prefix[tx + 1] = thread_incl_c;
    __syncthreads();

    if (hi_local >= 0) {
        atomicMax(&sc.b_first, hi_local);
        atomicMin(&sc.b_last, lo_local);
    }
    __syncthreads();

    const int b_first = sc.b_first;
    const int b_last  = sc.b_last;

    if (local_c > 0u) {
        uint32_t cur_c = sc.c_scan_prefix[tx];
#pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const int b = (TOTAL_BINS - 1) - (tx * ITEMS_PER_THREAD + j);
            if (b >= 0) {
                if (b == b_first) {
                    sc.cnt_first = bc[j];
                }
                if (b == b_last) {
                    sc.cnt_last       = bc[j];
                    sc.above_cnt_last = cur_c;
                }
            }
            cur_c += bc[j];
        }
    }

    if (check_k && sc.c_scan_prefix[tx] < rem_k && sc.c_scan_prefix[tx + 1] >= rem_k) {
        uint32_t cur_c = sc.c_scan_prefix[tx];
#pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t prev_c = cur_c;
            cur_c += bc[j];
            if (bc[j] > 0u && cur_c >= rem_k) {
                sc.b_k         = (TOTAL_BINS - 1) - (tx * ITEMS_PER_THREAD + j);
                sc.above_cnt_k = prev_c;
                sc.cnt_k       = bc[j];
                break;
            }
        }
    }

    if (check_p && local_c > 0u && target_p_d > 0.0 && sc.d_scan_prefix[tx] < target_p_d
        && sc.d_scan_prefix[tx + 1] >= target_p_d) {
        double   cur_s          = sc.d_scan_prefix[tx];
        uint32_t cur_c          = sc.c_scan_prefix[tx];
        int      last_nz_j      = -1;
        int      chosen_j       = -1;
        uint32_t chosen_above_c = 0u;
#pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t prev_c = cur_c;
            cur_s += bs[j];
            cur_c += bc[j];
            if (bc[j] > 0u) {
                last_nz_j = j;
                if (chosen_j < 0 && cur_s >= target_p_d) {
                    chosen_j       = j;
                    chosen_above_c = prev_c;
                }
            }
        }
        if (chosen_j < 0) {
            chosen_j       = last_nz_j;
            chosen_above_c = sc.c_scan_prefix[tx];
            for (int j = 0; j < chosen_j; ++j) {
                chosen_above_c += bc[j];
            }
        }
        sc.b_p         = (TOTAL_BINS - 1) - (tx * ITEMS_PER_THREAD + chosen_j);
        sc.above_cnt_p = chosen_above_c;
        sc.cnt_p       = bc[chosen_j];
    }
    __syncthreads();

    if (tx == 0) {
        resolveTargetBucket(sc, check_k, check_p, target_p_d, rem_k, fallback_p_to_last);
    }
    __syncthreads();

    double target_run_d = 0.0;
    if (local_c > 0u) {
        double cur_s = sc.d_scan_prefix[tx];
#pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            cur_s += bs[j];
            const int b = (TOTAL_BINS - 1) - (tx * ITEMS_PER_THREAD + j);
            if (b >= 0 && b == sc.b_target) {
                target_run_d = cur_s;
            }
        }
    }
    return target_run_d;
}

struct alignas(16) CandidateSortControl {
    typename cub::BlockRadixSort<uint32_t, kFastRenormThreads, kFastRenormSortItems>::TempStorage sort_u32;
    typename cub::BlockScan<double, kFastRenormThreads>::TempStorage                              scan_d;
    double                                                                                        tie_incl_sum_d;
    uint32_t                                                                                      raw_p_idx;
    uint32_t                                                                                      k_target;
    uint32_t                                                                                      cutoff_bits;
    uint32_t                                                                                      last_tie_idx;
};

__device__ __forceinline__ void sortAndSelectCutoff(CandidateSortControl& csc,
                                                    const uint32_t*       cand_src,
                                                    uint32_t              cnt_raw,
                                                    uint32_t              above_cutoff_bits,
                                                    uint8_t               mode_u8,
                                                    double                eff_p_d,
                                                    double                row_sum_d,
                                                    double                above_sum_d,
                                                    uint32_t              rem_k,
                                                    int                   tx,
                                                    float&                out_cutoff,
                                                    float&                out_kept_sum) {
    const auto     mode    = static_cast<RenormCutoffMode>(mode_u8);
    const bool     check_p = (mode != RenormCutoffMode::kPickMaxInBucket) && (eff_p_d > 0.0) && (eff_p_d < row_sum_d);
    const double   rem_p_d = eff_p_d - above_sum_d;
    const uint32_t cnt     = min(cnt_raw, kFastRenormMaxCand);

    if (check_p && rem_p_d <= 0.0 && above_sum_d > 0.0) {
        if (tx == 0) {
            out_cutoff   = __uint_as_float(above_cutoff_bits);
            out_kept_sum = static_cast<float>(above_sum_d);
        }
        return;
    }
    if (cnt <= 1u) {
        if (tx == 0) {
            if (cnt == 1u) {
                const float c0 = __uint_as_float(cand_src[0]);
                out_cutoff     = c0;
                out_kept_sum   = static_cast<float>(above_sum_d + static_cast<double>(c0));
            } else {
                out_cutoff   = __uint_as_float(above_cutoff_bits);
                out_kept_sum = static_cast<float>(above_sum_d);
            }
        }
        return;
    }

    uint32_t keys[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        keys[j]            = (idx < cnt) ? cand_src[idx] : 0u;
    }
    __syncthreads();

    cub::BlockRadixSort<uint32_t, kFastRenormThreads, kFastRenormSortItems>(csc.sort_u32).SortDescending(keys);
    __syncthreads();

    double local_sum_d = 0.0;
    double local_run_d[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        const double   v   = (idx < cnt) ? static_cast<double>(__uint_as_float(keys[j])) : 0.0;
        local_sum_d += v;
        local_run_d[j] = local_sum_d;
    }

    double thread_excl_d = 0.0;
    cub::BlockScan<double, kFastRenormThreads>(csc.scan_d).ExclusiveSum(local_sum_d, thread_excl_d);
    if (tx == 0) {
        thread_excl_d      = 0.0;
        csc.raw_p_idx      = cnt;
        csc.last_tie_idx   = 0u;
        csc.tie_incl_sum_d = 0.0;
    }
    __syncthreads();

    double incl_d[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        incl_d[j] = thread_excl_d + local_run_d[j];
    }

    if (check_p) {
#pragma unroll
        for (int j = 0; j < kFastRenormSortItems; ++j) {
            const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
            if (idx < cnt && incl_d[j] >= rem_p_d) {
                atomicMin(&csc.raw_p_idx, idx);
            }
        }
    }
    __syncthreads();

    if (tx == 0) {
        const uint32_t k_clamped = min((rem_k > 0u) ? (rem_k - 1u) : 0u, cnt - 1u);
        uint32_t       pick_idx;
        if (mode == RenormCutoffMode::kPickMaxInBucket) {
            pick_idx = 0u;
        } else if (mode == RenormCutoffMode::kTopKOnly && !check_p) {
            pick_idx = k_clamped;
        } else if (mode == RenormCutoffMode::kTopPOnly) {
            pick_idx = min(csc.raw_p_idx, cnt - 1u);
        } else {
            pick_idx = min(k_clamped, csc.raw_p_idx);
        }
        csc.k_target = pick_idx;
    }
    __syncthreads();

    const uint32_t pick_idx = csc.k_target;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        if (static_cast<uint32_t>(tx * kFastRenormSortItems + j) == pick_idx) {
            csc.cutoff_bits = keys[j];
        }
    }
    __syncthreads();

    const uint32_t target_bits = csc.cutoff_bits;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        if (idx < cnt && keys[j] == target_bits) {
            atomicMax(&csc.last_tie_idx, idx);
        }
    }
    __syncthreads();

    const uint32_t last_tie = csc.last_tie_idx;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        if (static_cast<uint32_t>(tx * kFastRenormSortItems + j) == last_tie) {
            csc.tie_incl_sum_d = incl_d[j];
        }
    }
    __syncthreads();

    if (tx == 0) {
        out_cutoff   = __uint_as_float(target_bits);
        out_kept_sum = static_cast<float>(above_sum_d + csc.tie_incl_sum_d);
    }
    __syncthreads();
}

// ============================================================================
// Multi-CTA Pipeline (C = 2..16 CTAs/row when vocab > 16384 && batch <= 256)
// ============================================================================

struct alignas(16) MapR1Shared {
    float                                                                sum_hist[kR1Bins][2];
    uint32_t                                                             cnt_hist[kR1Bins][2];
    typename cub::BlockReduce<float, kFastRenormThreads>::TempStorage    red_f;
    typename cub::BlockReduce<uint32_t, kFastRenormThreads>::TempStorage red_u32;
};

template<uint32_t VEC_SIZE>
__global__ void __launch_bounds__(kFastRenormThreads)
    FastRenormMapR1Kernel(const float* probs, FastRenormWorkspaceViews ws, uint32_t d, uint32_t c_per_row) {
    __shared__ MapR1Shared smem;
    const uint32_t         c   = blockIdx.x;
    const uint32_t         row = blockIdx.y;
    const int              tx  = static_cast<int>(threadIdx.x);

    for (int i = tx; i < kR1Bins; i += kFastRenormThreads) {
        smem.sum_hist[i][0] = 0.0f;
        smem.sum_hist[i][1] = 0.0f;
        smem.cnt_hist[i][0] = 0u;
        smem.cnt_hist[i][1] = 0u;
    }
    __syncthreads();

    const uint32_t chunk_elems = fastChunkElems(d, c_per_row);
    const uint32_t chunk_start = c * chunk_elems;
    const uint32_t chunk_end   = min(d, chunk_start + chunk_elems);
    const uint32_t chunk_len   = (chunk_start < chunk_end) ? (chunk_end - chunk_start) : 0u;
    const float*   chunk_ptr   = probs + static_cast<size_t>(row) * d + chunk_start;

    float          thread_sum     = 0.0f;
    uint32_t       thread_pos_cnt = 0u;
    const uint32_t num_iters      = (chunk_len + kFastRenormThreads * VEC_SIZE - 1u) / (kFastRenormThreads * VEC_SIZE);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
        if constexpr (VEC_SIZE == 4) {
            float4      v       = (base_col < chunk_len) ? *reinterpret_cast<const float4*>(chunk_ptr + base_col) :
                                                           make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool     pos    = (vals[j] > 0.0f);
                const uint32_t bits   = pos ? min(__float_as_uint(vals[j]), 0x7f800000u) : 0u;
                const uint32_t bucket = pos ? r1BucketOfBits(bits) : 0u;
                if (pos) {
                    thread_sum += vals[j];
                    thread_pos_cnt += 1u;
                }
                histAccumSubBank2<kR1Bins>(smem.sum_hist, smem.cnt_hist, tx, pos, bucket, vals[j]);
            }
        } else {
            const float    val    = (base_col < chunk_len) ? chunk_ptr[base_col] : 0.0f;
            const bool     pos    = (val > 0.0f);
            const uint32_t bits   = pos ? min(__float_as_uint(val), 0x7f800000u) : 0u;
            const uint32_t bucket = pos ? r1BucketOfBits(bits) : 0u;
            if (pos) {
                thread_sum += val;
                thread_pos_cnt += 1u;
            }
            histAccumSubBank2<kR1Bins>(smem.sum_hist, smem.cnt_hist, tx, pos, bucket, val);
        }
    }
    __syncthreads();

    const float    part_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_sum);
    const uint32_t part_cnt = cub::BlockReduce<uint32_t, kFastRenormThreads>(smem.red_u32).Sum(thread_pos_cnt);

    const size_t slot      = static_cast<size_t>(row) * c_per_row + c;
    float*       dst_sum_h = ws.part_hist_sum + slot * kPartHistStride;
    uint32_t*    dst_cnt_h = ws.part_hist_cnt + slot * kPartHistStride;
    for (int b = tx; b < kR1Bins; b += kFastRenormThreads) {
        dst_sum_h[b] = smem.sum_hist[b][0] + smem.sum_hist[b][1];
        dst_cnt_h[b] = smem.cnt_hist[b][0] + smem.cnt_hist[b][1];
    }
    if (tx == 0) {
        ws.part_row_sum[slot] = part_sum;
        ws.part_pos_cnt[slot] = part_cnt;
    }
}

struct alignas(16) ReduceShared {
    union {
        struct {
            typename cub::BlockScan<double, kFastRenormThreads>::TempStorage   scan_d;
            typename cub::BlockScan<uint32_t, kFastRenormThreads>::TempStorage scan_u32;
            ScanControl                                                        sc;
            double                                                             target_incl_sum_d;
        } bucket;
        CandidateSortControl cand;
    } u;
    uint8_t early_done;
    uint8_t done_cutoff;
};

template<bool FUSED_TOPK, bool ZERO_P_OR_K_INVALID>
__global__ void __launch_bounds__(kFastRenormThreads) FastRenormReduceR1Kernel(FastRenormWorkspaceViews ws,
                                                                               const int32_t*           top_k_arr,
                                                                               int32_t                  top_k_val,
                                                                               const float*             top_p_arr,
                                                                               float                    top_p_val,
                                                                               uint32_t                 d,
                                                                               uint32_t                 c_per_row) {
    __shared__ ReduceShared smem;
    const uint32_t          row = blockIdx.x;
    const int               tx  = static_cast<int>(threadIdx.x);

    if (tx == 0) {
        double       row_sum_d = 0.0;
        uint32_t     pos_cnt   = 0u;
        const size_t base      = static_cast<size_t>(row) * c_per_row;
        for (uint32_t c = 0; c < c_per_row; ++c) {
            row_sum_d += static_cast<double>(ws.part_row_sum[base + c]);
            pos_cnt += ws.part_pos_cnt[base + c];
        }
        const int32_t k_raw = (top_k_arr != nullptr) ? top_k_arr[row] : top_k_val;
        const float   raw_p = (top_p_arr != nullptr) ? top_p_arr[row] : top_p_val;
        uint32_t      k;
        bool          invalid_params = false;
        if constexpr (ZERO_P_OR_K_INVALID) {
            invalid_params = (FUSED_TOPK && k_raw == 0) || (raw_p <= 0.0f);
            k              = FUSED_TOPK ? static_cast<uint32_t>(k_raw) : d;
        } else {
            k = (FUSED_TOPK && k_raw > 0) ? static_cast<uint32_t>(k_raw) : d;
        }

        const float  row_sum = static_cast<float>(row_sum_d);
        const double eff_p_d =
            (fabs(row_sum_d - 1.0) <= 1e-5) ? static_cast<double>(raw_p) : static_cast<double>(raw_p) * row_sum_d;
        const bool no_top_k = !FUSED_TOPK || (k >= pos_cnt);
        const bool no_top_p = (raw_p >= 1.0f) || (eff_p_d >= row_sum_d);

        auto& st       = ws.states[row];
        st.row_sum     = row_sum;
        st.eff_p_d     = eff_p_d;
        st.above_sum_d = 0.0;
        st.pos_cnt     = pos_cnt;
        st.k           = k;
        st.cand_cnt    = 0u;
        st.done_cutoff = 0u;

        if (invalid_params || pos_cnt == 0u) {
            st.cutoff       = 1.0f;
            st.kept_sum     = 0.0f;
            st.early_done   = 1u;
            st.done_cutoff  = 1u;
            smem.early_done = 1u;
        } else if (no_top_k && no_top_p) {
            st.cutoff       = 0.0f;
            st.kept_sum     = row_sum;
            st.early_done   = 2u;
            st.done_cutoff  = 1u;
            smem.early_done = 2u;
        } else {
            st.early_done   = 0u;
            smem.early_done = 0u;
        }
    }
    __syncthreads();

    if (smem.early_done != 0u) {
        return;
    }

    double       bs[4] = {0.0, 0.0, 0.0, 0.0};
    uint32_t     bc[4] = {0u, 0u, 0u, 0u};
    const size_t base  = static_cast<size_t>(row) * c_per_row;
    for (uint32_t c = 0; c < c_per_row; ++c) {
        const size_t h_off = (base + c) * kPartHistStride;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = (kR1Bins - 1) - (tx * 4 + j);
            if (b >= 0) {
                bs[j] += static_cast<double>(ws.part_hist_sum[h_off + b]);
                bc[j] += ws.part_hist_cnt[h_off + b];
            }
        }
    }

    const auto& st      = ws.states[row];
    const bool  check_k = FUSED_TOPK && (st.k < st.pos_cnt);
    const bool  check_p = (st.eff_p_d < static_cast<double>(st.row_sum));
    scanBucketsFP64<4, kR1Bins>(smem.u.bucket.sc,
                                smem.u.bucket.scan_d,
                                smem.u.bucket.scan_u32,
                                bs,
                                bc,
                                tx,
                                check_k,
                                check_p,
                                st.eff_p_d,
                                st.k,
                                true);

    if (tx == 0) {
        auto&    dst = ws.states[row];
        uint32_t lo  = 1u;
        uint32_t hi  = 0x7f800000u;
        r1RangeForBucket(static_cast<uint32_t>(smem.u.bucket.sc.b_target), lo, hi);
        dst.range_lo   = lo;
        dst.range_hi   = hi;
        dst.shift_cur  = shiftForRange(lo, hi);
        dst.b_target   = smem.u.bucket.sc.b_target;
        dst.cnt_target = smem.u.bucket.sc.cnt_target;
        dst.rem_k      = smem.u.bucket.sc.rem_k_out;
        dst.mode       = smem.u.bucket.sc.mode_out;
    }
}

struct alignas(16) MapRangeOrCollectShared {
    union {
        struct {
            float    sum_hist[kR2Bins];
            uint32_t cnt_hist[kR2Bins];
        } hist;
        struct {
            uint32_t cnt_hist[kR3Bins][2];
        } hist_leaf;
        uint32_t cand_bits[kFastRenormMaxCand];
    } u;
    typename cub::BlockReduce<float, kFastRenormThreads>::TempStorage red_f;
    uint32_t                                                          local_cand_cnt;
    uint32_t                                                          global_off;
};

template<uint32_t VEC_SIZE>
__global__ void __launch_bounds__(kFastRenormThreads)
    FastRenormMapRangeOrCollectKernel(const float* probs, FastRenormWorkspaceViews ws, uint32_t d, uint32_t c_per_row) {
    const uint32_t row = blockIdx.y;
    const auto&    st  = ws.states[row];
    if (st.early_done != 0u || st.done_cutoff != 0u) {
        return;
    }

    __shared__ MapRangeOrCollectShared smem;
    const uint32_t                     c       = blockIdx.x;
    const int                          tx      = static_cast<int>(threadIdx.x);
    const uint32_t                     lo      = st.range_lo;
    const uint32_t                     hi      = st.range_hi;
    const int32_t                      shift   = st.shift_cur;
    const bool                         collect = (st.cnt_target <= kFastRenormMaxCand);
    const bool                         leaf    = (!collect && shift == 0);

    if (collect) {
        if (tx == 0) {
            smem.local_cand_cnt = 0u;
        }
    } else if (leaf) {
        for (int i = tx; i < kR3Bins; i += kFastRenormThreads) {
            smem.u.hist_leaf.cnt_hist[i][0] = 0u;
            smem.u.hist_leaf.cnt_hist[i][1] = 0u;
        }
    } else {
        for (int i = tx; i < kR2Bins; i += kFastRenormThreads) {
            smem.u.hist.sum_hist[i] = 0.0f;
            smem.u.hist.cnt_hist[i] = 0u;
        }
    }
    __syncthreads();

    const uint32_t chunk_elems = fastChunkElems(d, c_per_row);
    const uint32_t chunk_start = c * chunk_elems;
    const uint32_t chunk_end   = min(d, chunk_start + chunk_elems);
    const uint32_t chunk_len   = (chunk_start < chunk_end) ? (chunk_end - chunk_start) : 0u;
    const float*   chunk_ptr   = probs + static_cast<size_t>(row) * d + chunk_start;

    float          thread_above_sum = 0.0f;
    const uint32_t num_iters = (chunk_len + kFastRenormThreads * VEC_SIZE - 1u) / (kFastRenormThreads * VEC_SIZE);

    if (collect) {
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
            if constexpr (VEC_SIZE == 4) {
                float4      v       = (base_col < chunk_len) ? *reinterpret_cast<const float4*>(chunk_ptr + base_col) :
                                                               make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (vals[j] > 0.0f) {
                        const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                        if (bits > hi) {
                            thread_above_sum += vals[j];
                        } else if (bits >= lo) {
                            const uint32_t pos = atomicAdd(&smem.local_cand_cnt, 1u);
                            if (pos < kFastRenormMaxCand) {
                                smem.u.cand_bits[pos] = bits;
                            }
                        }
                    }
                }
            } else {
                const float val = (base_col < chunk_len) ? chunk_ptr[base_col] : 0.0f;
                if (val > 0.0f) {
                    const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                    if (bits > hi) {
                        thread_above_sum += val;
                    } else if (bits >= lo) {
                        const uint32_t pos = atomicAdd(&smem.local_cand_cnt, 1u);
                        if (pos < kFastRenormMaxCand) {
                            smem.u.cand_bits[pos] = bits;
                        }
                    }
                }
            }
        }
        __syncthreads();

        const float above_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_above_sum);
        if (tx == 0) {
            ws.part_above_sum[static_cast<size_t>(row) * c_per_row + c] = above_sum;
            const uint32_t lcnt                                         = min(smem.local_cand_cnt, kFastRenormMaxCand);
            smem.local_cand_cnt                                         = lcnt;
            smem.global_off = (lcnt > 0u) ? atomicAdd(&ws.states[row].cand_cnt, lcnt) : 0u;
        }
        __syncthreads();

        const uint32_t lcnt  = smem.local_cand_cnt;
        const uint32_t g_off = smem.global_off;
        uint32_t*      dst   = ws.cand_bits + static_cast<size_t>(row) * kFastRenormMaxCand;
        for (uint32_t idx = tx; idx < lcnt; idx += kFastRenormThreads) {
            if (g_off + idx < kFastRenormMaxCand) {
                dst[g_off + idx] = smem.u.cand_bits[idx];
            }
        }
    } else if (leaf) {
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
            if constexpr (VEC_SIZE == 4) {
                float4      v       = (base_col < chunk_len) ? *reinterpret_cast<const float4*>(chunk_ptr + base_col) :
                                                               make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bool     in_range = false;
                    uint32_t b        = 0u;
                    if (vals[j] > 0.0f) {
                        const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                        if (bits > hi) {
                            thread_above_sum += vals[j];
                        } else if (bits >= lo) {
                            in_range = true;
                            b        = bits - lo;
                        }
                    }
                    histAccumCountSubBank2<kR3Bins>(smem.u.hist_leaf.cnt_hist, tx, in_range, b);
                }
            } else {
                const float val      = (base_col < chunk_len) ? chunk_ptr[base_col] : 0.0f;
                bool        in_range = false;
                uint32_t    b        = 0u;
                if (val > 0.0f) {
                    const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                    if (bits > hi) {
                        thread_above_sum += val;
                    } else if (bits >= lo) {
                        in_range = true;
                        b        = bits - lo;
                    }
                }
                histAccumCountSubBank2<kR3Bins>(smem.u.hist_leaf.cnt_hist, tx, in_range, b);
            }
        }
        __syncthreads();

        const float  above_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_above_sum);
        const size_t slot      = static_cast<size_t>(row) * c_per_row + c;
        uint32_t*    dst_cnt_h = ws.part_hist_cnt + slot * kPartHistStride;
        for (int b = tx; b < kR3Bins; b += kFastRenormThreads) {
            dst_cnt_h[b] = smem.u.hist_leaf.cnt_hist[b][0] + smem.u.hist_leaf.cnt_hist[b][1];
        }
        if (tx == 0) {
            ws.part_above_sum[slot] = above_sum;
        }
    } else {
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
            if constexpr (VEC_SIZE == 4) {
                float4      v       = (base_col < chunk_len) ? *reinterpret_cast<const float4*>(chunk_ptr + base_col) :
                                                               make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bool     in_range = false;
                    uint32_t b        = 0u;
                    if (vals[j] > 0.0f) {
                        const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                        if (bits > hi) {
                            thread_above_sum += vals[j];
                        } else if (bits >= lo) {
                            in_range = true;
                            b        = (bits - lo) >> shift;
                        }
                    }
                    histAccumSingleBank<kR2Bins>(smem.u.hist.sum_hist, smem.u.hist.cnt_hist, tx, in_range, b, vals[j]);
                }
            } else {
                const float val      = (base_col < chunk_len) ? chunk_ptr[base_col] : 0.0f;
                bool        in_range = false;
                uint32_t    b        = 0u;
                if (val > 0.0f) {
                    const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                    if (bits > hi) {
                        thread_above_sum += val;
                    } else if (bits >= lo) {
                        in_range = true;
                        b        = (bits - lo) >> shift;
                    }
                }
                histAccumSingleBank<kR2Bins>(smem.u.hist.sum_hist, smem.u.hist.cnt_hist, tx, in_range, b, val);
            }
        }
        __syncthreads();

        const float  above_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_above_sum);
        const size_t slot      = static_cast<size_t>(row) * c_per_row + c;
        float*       dst_sum_h = ws.part_hist_sum + slot * kPartHistStride;
        uint32_t*    dst_cnt_h = ws.part_hist_cnt + slot * kPartHistStride;
        for (int b = tx; b < kR2Bins; b += kFastRenormThreads) {
            dst_sum_h[b] = smem.u.hist.sum_hist[b];
            dst_cnt_h[b] = smem.u.hist.cnt_hist[b];
        }
        if (tx == 0) {
            ws.part_above_sum[slot] = above_sum;
        }
    }
}

__global__ void __launch_bounds__(kFastRenormThreads)
    FastRenormReduceRangeOrFinishKernel(FastRenormWorkspaceViews ws, uint32_t c_per_row) {
    const uint32_t row = blockIdx.x;
    auto&          st  = ws.states[row];
    if (st.early_done != 0u || st.done_cutoff != 0u) {
        return;
    }

    __shared__ ReduceShared smem;
    const int               tx         = static_cast<int>(threadIdx.x);
    const size_t            base       = static_cast<size_t>(row) * c_per_row;
    const uint32_t          above_bits = (st.range_hi < 0x7f800000u) ? (st.range_hi + 1u) : 0x7f800000u;

    if (tx == 0) {
        double above_sum_d = 0.0;
        for (uint32_t c = 0; c < c_per_row; ++c) {
            above_sum_d += static_cast<double>(ws.part_above_sum[base + c]);
        }
        st.above_sum_d      = above_sum_d;
        const auto mode     = static_cast<RenormCutoffMode>(st.mode);
        const bool p_active = (mode != RenormCutoffMode::kPickMaxInBucket) && (st.eff_p_d > 0.0)
                              && (st.eff_p_d < static_cast<double>(st.row_sum));
        if (p_active && (st.eff_p_d - above_sum_d) <= 0.0 && above_sum_d > 0.0) {
            st.cutoff        = __uint_as_float(above_bits);
            st.kept_sum      = static_cast<float>(above_sum_d);
            st.done_cutoff   = 1u;
            smem.done_cutoff = 1u;
        } else {
            smem.done_cutoff = 0u;
        }
    }
    __syncthreads();

    if (smem.done_cutoff != 0u) {
        return;
    }

    if (st.cnt_target <= kFastRenormMaxCand) {
        float out_cutoff   = 0.0f;
        float out_kept_sum = 0.0f;
        sortAndSelectCutoff(smem.u.cand,
                            ws.cand_bits + static_cast<size_t>(row) * kFastRenormMaxCand,
                            st.cand_cnt,
                            above_bits,
                            st.mode,
                            st.eff_p_d,
                            static_cast<double>(st.row_sum),
                            st.above_sum_d,
                            st.rem_k,
                            tx,
                            out_cutoff,
                            out_kept_sum);
        if (tx == 0) {
            st.cutoff      = out_cutoff;
            st.kept_sum    = out_kept_sum;
            st.done_cutoff = 1u;
        }
        return;
    }

    const uint32_t lo    = st.range_lo;
    const uint32_t hi    = st.range_hi;
    const int32_t  shift = st.shift_cur;
    const bool     leaf  = (shift == 0);
    double         bs[4] = {0.0, 0.0, 0.0, 0.0};
    uint32_t       bc[4] = {0u, 0u, 0u, 0u};

    if (leaf) {
        for (uint32_t c = 0; c < c_per_row; ++c) {
            const size_t h_off = (base + c) * kPartHistStride;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b = (kR3Bins - 1) - (tx * 4 + j);
                bc[j] += ws.part_hist_cnt[h_off + b];
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (bc[j] > 0u) {
                const uint32_t b         = static_cast<uint32_t>((kR3Bins - 1) - (tx * 4 + j));
                const float    exact_val = __uint_as_float(lo + b);
                bs[j]                    = static_cast<double>(bc[j]) * static_cast<double>(exact_val);
            }
        }
    } else {
        for (uint32_t c = 0; c < c_per_row; ++c) {
            const size_t h_off = (base + c) * kPartHistStride;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b = (kR2Bins - 1) - (tx * 4 + j);
                bs[j] += static_cast<double>(ws.part_hist_sum[h_off + b]);
                bc[j] += ws.part_hist_cnt[h_off + b];
            }
        }
    }

    const auto     mode_c    = static_cast<RenormCutoffMode>(st.mode);
    const bool     check_k_c = (mode_c == RenormCutoffMode::kTopKOnly || mode_c == RenormCutoffMode::kBoth);
    const bool     p_act_c   = (st.eff_p_d > 0.0 && st.eff_p_d < static_cast<double>(st.row_sum));
    const bool     check_p_c = p_act_c || (mode_c == RenormCutoffMode::kPickMaxInBucket);
    const double   rem_p_d   = (mode_c == RenormCutoffMode::kPickMaxInBucket) ? 0.0 : (st.eff_p_d - st.above_sum_d);
    const uint32_t rem_k_c   = st.rem_k;
    const bool     fallback  = (mode_c != RenormCutoffMode::kTopKOnly);

    const double target_run_d = scanBucketsFP64<4, kR2Bins>(smem.u.bucket.sc,
                                                            smem.u.bucket.scan_d,
                                                            smem.u.bucket.scan_u32,
                                                            bs,
                                                            bc,
                                                            tx,
                                                            check_k_c,
                                                            check_p_c,
                                                            rem_p_d,
                                                            rem_k_c,
                                                            fallback);

    if (leaf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = (kR3Bins - 1) - (tx * 4 + j);
            if (b == smem.u.bucket.sc.b_target) {
                smem.u.bucket.target_incl_sum_d = target_run_d;
            }
        }
        __syncthreads();
        if (tx == 0) {
            const uint32_t exact_bits = lo + static_cast<uint32_t>(smem.u.bucket.sc.b_target);
            st.cutoff                 = __uint_as_float(exact_bits);
            st.kept_sum               = static_cast<float>(st.above_sum_d + smem.u.bucket.target_incl_sum_d);
            st.done_cutoff            = 1u;
        }
    } else if (tx == 0) {
        uint32_t next_lo = lo;
        uint32_t next_hi = hi;
        int32_t  next_sh = 0;
        refineSubRange(lo, hi, shift, static_cast<uint32_t>(smem.u.bucket.sc.b_target), next_lo, next_hi, next_sh);
        st.range_lo   = next_lo;
        st.range_hi   = next_hi;
        st.shift_cur  = next_sh;
        st.b_target   = smem.u.bucket.sc.b_target;
        st.cnt_target = smem.u.bucket.sc.cnt_target;
        st.rem_k      = smem.u.bucket.sc.rem_k_out;
        st.mode       = smem.u.bucket.sc.mode_out;
        st.cand_cnt   = 0u;
    }
}

template<uint32_t VEC_SIZE>
__global__ void __launch_bounds__(kFastRenormThreads) FastRenormNormalizeKernel(
    const float* probs, float* renormed_prob, FastRenormWorkspaceViews ws, uint32_t d, uint32_t c_per_row) {
    const uint32_t row        = blockIdx.y;
    const uint32_t c          = blockIdx.x;
    const int      tx         = static_cast<int>(threadIdx.x);
    const auto&    st         = ws.states[row];
    const uint8_t  early      = st.early_done;
    const float    cutoff     = st.cutoff;
    const float    denom      = (st.kept_sum <= 1e-8f) ? 1.0f : st.kept_sum;
    const float    normalizer = 1.0f / denom;

    const uint32_t chunk_elems = fastChunkElems(d, c_per_row);
    const uint32_t chunk_start = c * chunk_elems;
    const uint32_t chunk_end   = min(d, chunk_start + chunk_elems);
    const uint32_t chunk_len   = (chunk_start < chunk_end) ? (chunk_end - chunk_start) : 0u;
    const float*   chunk_in    = probs + static_cast<size_t>(row) * d + chunk_start;
    float*         chunk_out   = renormed_prob + static_cast<size_t>(row) * d + chunk_start;
    const uint32_t num_iters   = (chunk_len + kFastRenormThreads * VEC_SIZE - 1u) / (kFastRenormThreads * VEC_SIZE);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
        if (base_col >= chunk_len) {
            continue;
        }
        if (early == 1u) {
            if constexpr (VEC_SIZE == 4) {
                *reinterpret_cast<float4*>(chunk_out + base_col) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                chunk_out[base_col] = 0.0f;
            }
        } else if (early == 2u) {
            if constexpr (VEC_SIZE == 4) {
                float4 v = *reinterpret_cast<const float4*>(chunk_in + base_col);
                v.x      = (v.x > 0.0f) ? (v.x * normalizer) : 0.0f;
                v.y      = (v.y > 0.0f) ? (v.y * normalizer) : 0.0f;
                v.z      = (v.z > 0.0f) ? (v.z * normalizer) : 0.0f;
                v.w      = (v.w > 0.0f) ? (v.w * normalizer) : 0.0f;
                *reinterpret_cast<float4*>(chunk_out + base_col) = v;
            } else {
                const float v       = chunk_in[base_col];
                chunk_out[base_col] = (v > 0.0f) ? (v * normalizer) : 0.0f;
            }
        } else {
            if constexpr (VEC_SIZE == 4) {
                float4 v = *reinterpret_cast<const float4*>(chunk_in + base_col);
                v.x      = (v.x >= cutoff && v.x > 0.0f) ? (v.x * normalizer) : 0.0f;
                v.y      = (v.y >= cutoff && v.y > 0.0f) ? (v.y * normalizer) : 0.0f;
                v.z      = (v.z >= cutoff && v.z > 0.0f) ? (v.z * normalizer) : 0.0f;
                v.w      = (v.w >= cutoff && v.w > 0.0f) ? (v.w * normalizer) : 0.0f;
                *reinterpret_cast<float4*>(chunk_out + base_col) = v;
            } else {
                const float v       = chunk_in[base_col];
                chunk_out[base_col] = (v >= cutoff && v > 0.0f) ? (v * normalizer) : 0.0f;
            }
        }
    }
}

// ============================================================================
// Multi-CTA Warp-Tile Sampling & Fused Renorm+Sample Kernels
// ============================================================================

template<bool WRITE_RENORM, uint32_t VEC_SIZE>
__global__ void __launch_bounds__(kFastRenormThreads) FastSampleChunkTileSumsKernel(
    const float* probs, float* renormed_prob, FastRenormWorkspaceViews ws, uint32_t d, uint32_t c_per_row) {
    const uint32_t row     = blockIdx.y;
    const uint32_t c       = blockIdx.x;
    const int      tx      = static_cast<int>(threadIdx.x);
    const int      warp_id = tx >> 5;
    const int      lane    = tx & 31;

    const auto&   st         = ws.states[row];
    const uint8_t early      = st.early_done;
    const float   cutoff     = st.cutoff;
    float         normalizer = 1.0f;
    if constexpr (WRITE_RENORM) {
        const float denom = (st.kept_sum <= 1e-8f) ? 1.0f : st.kept_sum;
        normalizer        = 1.0f / denom;
    }
    if (!WRITE_RENORM && early == 1u) {
        return;
    }

    const uint32_t chunk_elems     = fastChunkElems(d, c_per_row);
    const uint32_t tiles_per_chunk = chunk_elems / kSampleTileSize;
    const uint32_t chunk_tile_off  = c * tiles_per_chunk;
    const uint32_t num_tiles       = (d + kSampleTileSize - 1u) / kSampleTileSize;
    const uint32_t chunk_tiles = (chunk_tile_off < num_tiles) ? min(tiles_per_chunk, num_tiles - chunk_tile_off) : 0u;
    const float*   probs_row   = probs + static_cast<size_t>(row) * d;
    float*         out_row     = WRITE_RENORM ? (renormed_prob + static_cast<size_t>(row) * d) : nullptr;
    float*         tile_dst    = ws.tile_sums + static_cast<size_t>(row) * num_tiles;

    for (uint32_t local_tile = static_cast<uint32_t>(warp_id); local_tile < chunk_tiles; local_tile += 8u) {
        const uint32_t tile_id   = chunk_tile_off + local_tile;
        const uint32_t tile_base = tile_id * kSampleTileSize;
        float          lane_sum  = 0.0f;

        if constexpr (VEC_SIZE == 4) {
            const uint32_t col = tile_base + static_cast<uint32_t>(lane) * 4u;
            if (col < d) {
                if (early == 1u) {
                    if constexpr (WRITE_RENORM) {
                        *reinterpret_cast<float4*>(out_row + col) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else {
                    float4     v         = *reinterpret_cast<const float4*>(probs_row + col);
                    const bool full_tile = (early == 2u);
                    const bool k0        = full_tile ? (v.x > 0.0f) : (v.x >= cutoff && v.x > 0.0f);
                    const bool k1        = full_tile ? (v.y > 0.0f) : (v.y >= cutoff && v.y > 0.0f);
                    const bool k2        = full_tile ? (v.z > 0.0f) : (v.z >= cutoff && v.z > 0.0f);
                    const bool k3        = full_tile ? (v.w > 0.0f) : (v.w >= cutoff && v.w > 0.0f);
                    lane_sum = (k0 ? v.x : 0.0f) + (k1 ? v.y : 0.0f) + (k2 ? v.z : 0.0f) + (k3 ? v.w : 0.0f);
                    if constexpr (WRITE_RENORM) {
                        v.x                                       = k0 ? (v.x * normalizer) : 0.0f;
                        v.y                                       = k1 ? (v.y * normalizer) : 0.0f;
                        v.z                                       = k2 ? (v.z * normalizer) : 0.0f;
                        v.w                                       = k3 ? (v.w * normalizer) : 0.0f;
                        *reinterpret_cast<float4*>(out_row + col) = v;
                    }
                }
            }
        } else {
            const bool full_tile = (early == 2u);
#pragma unroll
            for (uint32_t sub = 0; sub < 4u; ++sub) {
                const uint32_t col = tile_base + sub * 32u + static_cast<uint32_t>(lane);
                if (col < d) {
                    if (early == 1u) {
                        if constexpr (WRITE_RENORM) {
                            out_row[col] = 0.0f;
                        }
                    } else {
                        const float v    = probs_row[col];
                        const bool  kept = full_tile ? (v > 0.0f) : (v >= cutoff && v > 0.0f);
                        if (kept) {
                            lane_sum += v;
                        }
                        if constexpr (WRITE_RENORM) {
                            out_row[col] = kept ? (v * normalizer) : 0.0f;
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lane_sum += __shfl_down_sync(0xffffffffu, lane_sum, off);
        }
        if (lane == 0) {
            tile_dst[tile_id] = lane_sum;
        }
    }
}

struct alignas(16) PickTileShared {
    union {
        typename cub::BlockReduce<double, kFastRenormThreads>::TempStorage  red_d;
        typename cub::BlockReduce<int32_t, kFastRenormThreads>::TempStorage red_i;
        typename cub::BlockScan<double, kFastRenormThreads>::TempStorage    scan_d;
    } u;
    double  chunk_sums[kPickMaxChunks];
    double  chunk_prefix[kPickMaxChunks + 1];
    double  target_u_d;
    double  tile_base_prefix_d;
    int32_t win_chunk;
    int32_t win_tile;
    int32_t last_nz_tile;
    int32_t chosen_col;
    int32_t last_valid_in_tile;
    uint8_t early_invalid;
    uint8_t early;
    float   cutoff;
};

template<typename IdType>
__global__ void __launch_bounds__(kFastRenormThreads) FastSamplePickTileKernel(const float*             probs,
                                                                               IdType*                  output,
                                                                               bool*                    valid,
                                                                               FastRenormWorkspaceViews ws,
                                                                               uint32_t                 d,
                                                                               uint64_t*                seed_arr,
                                                                               uint64_t                 seed_val,
                                                                               uint64_t*                offset_arr,
                                                                               uint64_t                 offset_val) {
    __shared__ PickTileShared smem;
    const uint32_t            row = blockIdx.x;
    const int                 tx  = static_cast<int>(threadIdx.x);

    const auto& st = ws.states[row];
    if (st.early_done == 1u) {
        if (tx == 0) {
            output[row] = 0;
            valid[row]  = false;
        }
        return;
    }
    if (tx == 0) {
        smem.early  = st.early_done;
        smem.cutoff = st.cutoff;
    }
    __syncthreads();

    const uint32_t num_tiles  = (d + kSampleTileSize - 1u) / kSampleTileSize;
    const uint32_t num_chunks = (num_tiles + kFastRenormThreads - 1u) / kFastRenormThreads;
    const float*   tile_row   = ws.tile_sums + static_cast<size_t>(row) * num_tiles;

    int32_t thread_last_nz = -1;
    for (uint32_t ch = 0; ch < num_chunks; ++ch) {
        const uint32_t t_idx = ch * kFastRenormThreads + static_cast<uint32_t>(tx);
        const double   t_w   = (t_idx < num_tiles) ? static_cast<double>(tile_row[t_idx]) : 0.0;
        if (t_w > 0.0) {
            thread_last_nz = static_cast<int32_t>(t_idx);
        }
        const double c_sum = cub::BlockReduce<double, kFastRenormThreads>(smem.u.red_d).Sum(t_w);
        __syncthreads();
        if (tx == 0) {
            smem.chunk_sums[ch] = c_sum;
        }
    }

    const int32_t max_nz =
        cub::BlockReduce<int32_t, kFastRenormThreads>(smem.u.red_i).Reduce(thread_last_nz, IntMaxOp{});
    __syncthreads();

    if (tx == 0) {
        smem.last_nz_tile    = max_nz;
        smem.chunk_prefix[0] = 0.0;
        double total_mass_d  = 0.0;
        for (uint32_t ch = 0; ch < num_chunks; ++ch) {
            total_mass_d += smem.chunk_sums[ch];
            smem.chunk_prefix[ch + 1] = total_mass_d;
        }

        if (total_mass_d <= 0.0 || max_nz < 0) {
            smem.early_invalid = 1u;
            output[row]        = 0;
            valid[row]         = false;
        } else {
            smem.early_invalid = 0u;

            const uint64_t             philox_seed   = seed_arr ? seed_arr[row] : seed_val;
            const uint64_t             philox_offset = offset_arr ? offset_arr[row] : offset_val;
            const uint64_t             subseq        = (seed_arr != nullptr || offset_arr != nullptr) ? 0ull : row;
            curandStatePhilox4_32_10_t rng_state;
            curand_init(philox_seed, subseq, philox_offset, &rng_state);
            const float  u01      = curand_uniform(&rng_state);
            const double target_u = static_cast<double>(u01) * total_mass_d;
            smem.target_u_d       = target_u;

            int32_t win_ch = -1;
            for (uint32_t ch = 0; ch < num_chunks; ++ch) {
                if (smem.chunk_sums[ch] > 0.0 && smem.chunk_prefix[ch + 1] > target_u) {
                    win_ch = static_cast<int32_t>(ch);
                    break;
                }
            }
            smem.win_chunk          = win_ch;
            smem.win_tile           = (win_ch < 0) ? max_nz : static_cast<int32_t>(num_tiles);
            smem.tile_base_prefix_d = target_u;
            smem.chosen_col         = static_cast<int32_t>(d);
            smem.last_valid_in_tile = -1;
        }
    }
    __syncthreads();

    if (smem.early_invalid != 0u) {
        return;
    }

    if (smem.win_chunk >= 0) {
        const uint32_t t_idx = static_cast<uint32_t>(smem.win_chunk) * kFastRenormThreads + static_cast<uint32_t>(tx);
        const double   t_w   = (t_idx < num_tiles) ? static_cast<double>(tile_row[t_idx]) : 0.0;
        double         blk_incl = 0.0;
        cub::BlockScan<double, kFastRenormThreads>(smem.u.scan_d).InclusiveSum(t_w, blk_incl);
        const double incl = smem.chunk_prefix[smem.win_chunk] + blk_incl;
        if (t_w > 0.0 && incl > smem.target_u_d) {
            atomicMin(&smem.win_tile, static_cast<int32_t>(t_idx));
        }
        __syncthreads();
        if (tx == 0 && smem.win_tile >= static_cast<int32_t>(num_tiles)) {
            smem.win_tile = smem.last_nz_tile;
        }
        __syncthreads();
        if (static_cast<int32_t>(t_idx) == smem.win_tile) {
            smem.tile_base_prefix_d = incl - t_w;
        }
        __syncthreads();
    }

    const float*   row_probs = probs + static_cast<size_t>(row) * d;
    const uint32_t col       = static_cast<uint32_t>(smem.win_tile) * kSampleTileSize + static_cast<uint32_t>(tx);
    double         e_w       = 0.0;
    bool           kept      = false;
    if (tx < static_cast<int>(kSampleTileSize) && col < d) {
        const float v = row_probs[col];
        kept          = (smem.early == 2u) ? (v > 0.0f) : (v >= smem.cutoff && v > 0.0f);
        if (kept) {
            e_w = static_cast<double>(v);
        }
    }

    double elem_incl = 0.0;
    cub::BlockScan<double, kFastRenormThreads>(smem.u.scan_d).InclusiveSum(e_w, elem_incl);
    if (kept) {
        atomicMax(&smem.last_valid_in_tile, static_cast<int32_t>(col));
        if (smem.win_chunk >= 0 && smem.tile_base_prefix_d + elem_incl > smem.target_u_d) {
            atomicMin(&smem.chosen_col, static_cast<int32_t>(col));
        }
    }
    __syncthreads();

    if (tx == 0) {
        int32_t picked = smem.chosen_col;
        if (picked >= static_cast<int32_t>(d)) {
            picked = smem.last_valid_in_tile;
        }
        if (picked < 0) {
            output[row] = 0;
            valid[row]  = false;
        } else {
            output[row] = static_cast<IdType>(picked);
            valid[row]  = true;
        }
    }
}

// ============================================================================
// Single-CTA Fallback (d <= 16384 || batch_size > 256 || no workspace)
// ============================================================================

struct alignas(16) SingleCTAShared {
    union ScratchUnion {
        struct {
            float    sum_hist[kR1Bins][2];
            uint32_t cnt_hist[kR1Bins][2];
        } r1;
        struct {
            float    sum_hist[kR2Bins];
            uint32_t cnt_hist[kR2Bins];
        } r2;
        struct {
            uint32_t cnt_hist[kR3Bins][2];
        } r3;
        uint32_t             cand_bits[kFastRenormMaxCand];
        CandidateSortControl cand_sort;
    } u;

    ScanControl                                                          sc;
    typename cub::BlockScan<double, kFastRenormThreads>::TempStorage     scan_d;
    typename cub::BlockScan<uint32_t, kFastRenormThreads>::TempStorage   scan_u32;
    typename cub::BlockReduce<float, kFastRenormThreads>::TempStorage    red_f;
    typename cub::BlockReduce<uint32_t, kFastRenormThreads>::TempStorage red_u32;

    double   eff_p_d;
    double   above_sum_d;
    double   leaf_target_sum_d;
    float    row_sum;
    float    kept_sum;
    float    cutoff;
    uint32_t pos_cnt;
    uint32_t k;
    uint32_t rem_k;
    uint32_t range_lo;
    uint32_t range_hi;
    int32_t  shift_cur;
    int32_t  b_target;
    uint32_t cnt_target;
    uint32_t cand_cnt;
    uint8_t  mode;
    uint8_t  early_done;
    uint8_t  done_cutoff;
};

template<uint32_t VEC_SIZE>
__device__ __forceinline__ void
singleCTAFinishInRange(SingleCTAShared& smem, int tx, const float* row_probs, uint32_t d, uint32_t lo, uint32_t hi) {
    if (tx == 0) {
        smem.cand_cnt    = 0u;
        smem.done_cutoff = 0u;
    }
    __syncthreads();

    float          thread_above_sum = 0.0f;
    const uint32_t num_iters        = (d + kFastRenormThreads * VEC_SIZE - 1u) / (kFastRenormThreads * VEC_SIZE);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
        if constexpr (VEC_SIZE == 4) {
            float4      v       = (base_col < d) ? *reinterpret_cast<const float4*>(row_probs + base_col) :
                                                   make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (vals[j] > 0.0f) {
                    const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                    if (bits > hi) {
                        thread_above_sum += vals[j];
                    } else if (bits >= lo) {
                        const uint32_t pos = atomicAdd(&smem.cand_cnt, 1u);
                        if (pos < kFastRenormMaxCand) {
                            smem.u.cand_bits[pos] = bits;
                        }
                    }
                }
            }
        } else {
            const float val = (base_col < d) ? row_probs[base_col] : 0.0f;
            if (val > 0.0f) {
                const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                if (bits > hi) {
                    thread_above_sum += val;
                } else if (bits >= lo) {
                    const uint32_t pos = atomicAdd(&smem.cand_cnt, 1u);
                    if (pos < kFastRenormMaxCand) {
                        smem.u.cand_bits[pos] = bits;
                    }
                }
            }
        }
    }
    __syncthreads();

    const float    above_sum  = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_above_sum);
    const uint32_t above_bits = (hi < 0x7f800000u) ? (hi + 1u) : 0x7f800000u;
    if (tx == 0) {
        smem.above_sum_d = static_cast<double>(above_sum);
    }
    __syncthreads();

    const uint32_t cnt     = min(smem.cand_cnt, kFastRenormMaxCand);
    const auto     mode    = static_cast<RenormCutoffMode>(smem.mode);
    const bool     check_p = (mode != RenormCutoffMode::kPickMaxInBucket) && (smem.eff_p_d > 0.0)
                         && (smem.eff_p_d < static_cast<double>(smem.row_sum));
    const double rem_p_d = smem.eff_p_d - smem.above_sum_d;
    if (check_p && rem_p_d <= 0.0 && smem.above_sum_d > 0.0) {
        if (tx == 0) {
            smem.cutoff      = __uint_as_float(above_bits);
            smem.kept_sum    = static_cast<float>(smem.above_sum_d);
            smem.done_cutoff = 1u;
        }
        __syncthreads();
        return;
    }
    if (cnt <= 1u) {
        if (tx == 0) {
            if (cnt == 1u) {
                const float c0 = __uint_as_float(smem.u.cand_bits[0]);
                smem.cutoff    = c0;
                smem.kept_sum  = static_cast<float>(smem.above_sum_d + static_cast<double>(c0));
            } else {
                smem.cutoff   = __uint_as_float(above_bits);
                smem.kept_sum = static_cast<float>(smem.above_sum_d);
            }
            smem.done_cutoff = 1u;
        }
        __syncthreads();
        return;
    }

    uint32_t keys[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        keys[j]            = (idx < cnt) ? smem.u.cand_bits[idx] : 0u;
    }
    __syncthreads();

    auto& csc = smem.u.cand_sort;
    cub::BlockRadixSort<uint32_t, kFastRenormThreads, kFastRenormSortItems>(csc.sort_u32).SortDescending(keys);
    __syncthreads();

    double local_sum_d = 0.0;
    double local_run_d[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        const double   v   = (idx < cnt) ? static_cast<double>(__uint_as_float(keys[j])) : 0.0;
        local_sum_d += v;
        local_run_d[j] = local_sum_d;
    }

    double thread_excl_d = 0.0;
    cub::BlockScan<double, kFastRenormThreads>(csc.scan_d).ExclusiveSum(local_sum_d, thread_excl_d);
    if (tx == 0) {
        thread_excl_d      = 0.0;
        csc.raw_p_idx      = cnt;
        csc.last_tie_idx   = 0u;
        csc.tie_incl_sum_d = 0.0;
    }
    __syncthreads();

    double incl_d[kFastRenormSortItems];
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        incl_d[j] = thread_excl_d + local_run_d[j];
    }

    if (check_p) {
#pragma unroll
        for (int j = 0; j < kFastRenormSortItems; ++j) {
            const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
            if (idx < cnt && incl_d[j] >= rem_p_d) {
                atomicMin(&csc.raw_p_idx, idx);
            }
        }
    }
    __syncthreads();

    if (tx == 0) {
        const uint32_t k_clamped = min((smem.rem_k > 0u) ? (smem.rem_k - 1u) : 0u, cnt - 1u);
        uint32_t       pick_idx;
        if (mode == RenormCutoffMode::kPickMaxInBucket) {
            pick_idx = 0u;
        } else if (mode == RenormCutoffMode::kTopKOnly && !check_p) {
            pick_idx = k_clamped;
        } else if (mode == RenormCutoffMode::kTopPOnly) {
            pick_idx = min(csc.raw_p_idx, cnt - 1u);
        } else {
            pick_idx = min(k_clamped, csc.raw_p_idx);
        }
        csc.k_target = pick_idx;
    }
    __syncthreads();

    const uint32_t pick_idx = csc.k_target;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        if (static_cast<uint32_t>(tx * kFastRenormSortItems + j) == pick_idx) {
            csc.cutoff_bits = keys[j];
        }
    }
    __syncthreads();

    const uint32_t target_bits = csc.cutoff_bits;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        const uint32_t idx = static_cast<uint32_t>(tx * kFastRenormSortItems + j);
        if (idx < cnt && keys[j] == target_bits) {
            atomicMax(&csc.last_tie_idx, idx);
        }
    }
    __syncthreads();

    const uint32_t last_tie = csc.last_tie_idx;
#pragma unroll
    for (int j = 0; j < kFastRenormSortItems; ++j) {
        if (static_cast<uint32_t>(tx * kFastRenormSortItems + j) == last_tie) {
            csc.tie_incl_sum_d = incl_d[j];
        }
    }
    __syncthreads();

    if (tx == 0) {
        smem.cutoff      = __uint_as_float(target_bits);
        smem.kept_sum    = static_cast<float>(smem.above_sum_d + csc.tie_incl_sum_d);
        smem.done_cutoff = 1u;
    }
    __syncthreads();
}

template<bool FUSED_TOPK, uint32_t VEC_SIZE>
__global__ void __launch_bounds__(kFastRenormThreads) FastRenormKernel(const float*   probs,
                                                                       float*         renormed_prob,
                                                                       const int32_t* top_k_arr,
                                                                       int32_t        top_k_val,
                                                                       const float*   top_p_arr,
                                                                       float          top_p_val,
                                                                       uint32_t       d) {
    __shared__ SingleCTAShared smem;
    const uint32_t             row       = blockIdx.x;
    const int                  tx        = static_cast<int>(threadIdx.x);
    const float*               row_probs = probs + static_cast<size_t>(row) * d;
    float*                     row_out   = renormed_prob + static_cast<size_t>(row) * d;

    if (tx == 0) {
        const int32_t k_raw = (top_k_arr != nullptr) ? top_k_arr[row] : top_k_val;
        smem.k              = (FUSED_TOPK && k_raw > 0) ? static_cast<uint32_t>(k_raw) : d;
        smem.eff_p_d        = static_cast<double>((top_p_arr != nullptr) ? top_p_arr[row] : top_p_val);
        smem.early_done     = 0u;
        smem.done_cutoff    = 0u;
    }

    for (int i = tx; i < kR1Bins; i += kFastRenormThreads) {
        smem.u.r1.sum_hist[i][0] = 0.0f;
        smem.u.r1.sum_hist[i][1] = 0.0f;
        smem.u.r1.cnt_hist[i][0] = 0u;
        smem.u.r1.cnt_hist[i][1] = 0u;
    }
    __syncthreads();

    float          thread_sum     = 0.0f;
    uint32_t       thread_pos_cnt = 0u;
    const uint32_t num_iters      = (d + kFastRenormThreads * VEC_SIZE - 1u) / (kFastRenormThreads * VEC_SIZE);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
        if constexpr (VEC_SIZE == 4) {
            float4      v       = (base_col < d) ? *reinterpret_cast<const float4*>(row_probs + base_col) :
                                                   make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool     pos    = (vals[j] > 0.0f);
                const uint32_t bits   = pos ? min(__float_as_uint(vals[j]), 0x7f800000u) : 0u;
                const uint32_t bucket = pos ? r1BucketOfBits(bits) : 0u;
                if (pos) {
                    thread_sum += vals[j];
                    thread_pos_cnt += 1u;
                }
                histAccumSubBank2<kR1Bins>(smem.u.r1.sum_hist, smem.u.r1.cnt_hist, tx, pos, bucket, vals[j]);
            }
        } else {
            const float    val    = (base_col < d) ? row_probs[base_col] : 0.0f;
            const bool     pos    = (val > 0.0f);
            const uint32_t bits   = pos ? min(__float_as_uint(val), 0x7f800000u) : 0u;
            const uint32_t bucket = pos ? r1BucketOfBits(bits) : 0u;
            if (pos) {
                thread_sum += val;
                thread_pos_cnt += 1u;
            }
            histAccumSubBank2<kR1Bins>(smem.u.r1.sum_hist, smem.u.r1.cnt_hist, tx, pos, bucket, val);
        }
    }
    __syncthreads();

    const float    row_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_sum);
    const uint32_t pos_cnt = cub::BlockReduce<uint32_t, kFastRenormThreads>(smem.red_u32).Sum(thread_pos_cnt);

    if (tx == 0) {
        smem.row_sum        = row_sum;
        smem.pos_cnt        = pos_cnt;
        const double raw_p  = smem.eff_p_d;
        const double rsum_d = static_cast<double>(row_sum);
        const double scaled = (fabs(rsum_d - 1.0) <= 1e-5) ? raw_p : (raw_p * rsum_d);
        smem.eff_p_d        = scaled;
        const bool no_top_k = !FUSED_TOPK || (smem.k >= pos_cnt);
        const bool no_top_p = (raw_p >= 1.0) || (scaled >= rsum_d);

        if (pos_cnt == 0u) {
            smem.cutoff     = 1.0f;
            smem.kept_sum   = 0.0f;
            smem.early_done = 1u;
        } else if (no_top_k && no_top_p) {
            smem.cutoff     = 0.0f;
            smem.kept_sum   = row_sum;
            smem.early_done = 2u;
        }
    }
    __syncthreads();

    if (smem.early_done == 0u) {
        double   bs1[4] = {0.0, 0.0, 0.0, 0.0};
        uint32_t bc1[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = (kR1Bins - 1) - (tx * 4 + j);
            if (b >= 0) {
                bs1[j] = static_cast<double>(smem.u.r1.sum_hist[b][0]) + static_cast<double>(smem.u.r1.sum_hist[b][1]);
                bc1[j] = smem.u.r1.cnt_hist[b][0] + smem.u.r1.cnt_hist[b][1];
            }
        }
        __syncthreads();

        const bool check_k1 = FUSED_TOPK && (smem.k < smem.pos_cnt);
        const bool check_p1 = (smem.eff_p_d < static_cast<double>(smem.row_sum));
        scanBucketsFP64<4, kR1Bins>(
            smem.sc, smem.scan_d, smem.scan_u32, bs1, bc1, tx, check_k1, check_p1, smem.eff_p_d, smem.k, true);
        if (tx == 0) {
            uint32_t lo = 1u;
            uint32_t hi = 0x7f800000u;
            r1RangeForBucket(static_cast<uint32_t>(smem.sc.b_target), lo, hi);
            smem.range_lo   = lo;
            smem.range_hi   = hi;
            smem.shift_cur  = shiftForRange(lo, hi);
            smem.b_target   = smem.sc.b_target;
            smem.cnt_target = smem.sc.cnt_target;
            smem.rem_k      = smem.sc.rem_k_out;
            smem.mode       = smem.sc.mode_out;
        }
        __syncthreads();

        while (!smem.done_cutoff) {
            const uint32_t lo = smem.range_lo;
            const uint32_t hi = smem.range_hi;
            if (smem.cnt_target <= kFastRenormMaxCand) {
                singleCTAFinishInRange<VEC_SIZE>(smem, tx, row_probs, d, lo, hi);
                break;
            }

            const int32_t  shift      = smem.shift_cur;
            const bool     leaf       = (shift == 0);
            const uint32_t above_bits = (hi < 0x7f800000u) ? (hi + 1u) : 0x7f800000u;
            if (leaf) {
                for (int i = tx; i < kR3Bins; i += kFastRenormThreads) {
                    smem.u.r3.cnt_hist[i][0] = 0u;
                    smem.u.r3.cnt_hist[i][1] = 0u;
                }
            } else {
                for (int i = tx; i < kR2Bins; i += kFastRenormThreads) {
                    smem.u.r2.sum_hist[i] = 0.0f;
                    smem.u.r2.cnt_hist[i] = 0u;
                }
            }
            __syncthreads();

            float thread_above_sum = 0.0f;
            if (leaf) {
                for (uint32_t iter = 0; iter < num_iters; ++iter) {
                    const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
                    if constexpr (VEC_SIZE == 4) {
                        float4      v       = (base_col < d) ? *reinterpret_cast<const float4*>(row_probs + base_col) :
                                                               make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            bool     in_range = false;
                            uint32_t b        = 0u;
                            if (vals[j] > 0.0f) {
                                const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                                if (bits > hi) {
                                    thread_above_sum += vals[j];
                                } else if (bits >= lo) {
                                    in_range = true;
                                    b        = bits - lo;
                                }
                            }
                            histAccumCountSubBank2<kR3Bins>(smem.u.r3.cnt_hist, tx, in_range, b);
                        }
                    } else {
                        const float val      = (base_col < d) ? row_probs[base_col] : 0.0f;
                        bool        in_range = false;
                        uint32_t    b        = 0u;
                        if (val > 0.0f) {
                            const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                            if (bits > hi) {
                                thread_above_sum += val;
                            } else if (bits >= lo) {
                                in_range = true;
                                b        = bits - lo;
                            }
                        }
                        histAccumCountSubBank2<kR3Bins>(smem.u.r3.cnt_hist, tx, in_range, b);
                    }
                }
            } else {
                for (uint32_t iter = 0; iter < num_iters; ++iter) {
                    const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
                    if constexpr (VEC_SIZE == 4) {
                        float4      v       = (base_col < d) ? *reinterpret_cast<const float4*>(row_probs + base_col) :
                                                               make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            bool     in_range = false;
                            uint32_t b        = 0u;
                            if (vals[j] > 0.0f) {
                                const uint32_t bits = min(__float_as_uint(vals[j]), 0x7f800000u);
                                if (bits > hi) {
                                    thread_above_sum += vals[j];
                                } else if (bits >= lo) {
                                    in_range = true;
                                    b        = (bits - lo) >> shift;
                                }
                            }
                            histAccumSingleBank<kR2Bins>(
                                smem.u.r2.sum_hist, smem.u.r2.cnt_hist, tx, in_range, b, vals[j]);
                        }
                    } else {
                        const float val      = (base_col < d) ? row_probs[base_col] : 0.0f;
                        bool        in_range = false;
                        uint32_t    b        = 0u;
                        if (val > 0.0f) {
                            const uint32_t bits = min(__float_as_uint(val), 0x7f800000u);
                            if (bits > hi) {
                                thread_above_sum += val;
                            } else if (bits >= lo) {
                                in_range = true;
                                b        = (bits - lo) >> shift;
                            }
                        }
                        histAccumSingleBank<kR2Bins>(smem.u.r2.sum_hist, smem.u.r2.cnt_hist, tx, in_range, b, val);
                    }
                }
            }
            __syncthreads();

            const float above_sum = cub::BlockReduce<float, kFastRenormThreads>(smem.red_f).Sum(thread_above_sum);
            if (tx == 0) {
                const double above_d = static_cast<double>(above_sum);
                smem.above_sum_d     = above_d;
                const auto m         = static_cast<RenormCutoffMode>(smem.mode);
                const bool p_act     = (m != RenormCutoffMode::kPickMaxInBucket) && (smem.eff_p_d > 0.0)
                                   && (smem.eff_p_d < static_cast<double>(smem.row_sum));
                if (p_act && (smem.eff_p_d - above_d) <= 0.0 && above_d > 0.0) {
                    smem.cutoff      = __uint_as_float(above_bits);
                    smem.kept_sum    = static_cast<float>(above_d);
                    smem.done_cutoff = 1u;
                }
            }
            __syncthreads();

            if (smem.done_cutoff) {
                break;
            }

            double   bs[4] = {0.0, 0.0, 0.0, 0.0};
            uint32_t bc[4] = {0u, 0u, 0u, 0u};
            if (leaf) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int b = (kR3Bins - 1) - (tx * 4 + j);
                    bc[j]       = smem.u.r3.cnt_hist[b][0] + smem.u.r3.cnt_hist[b][1];
                    if (bc[j] > 0u) {
                        const float exact_val = __uint_as_float(lo + static_cast<uint32_t>(b));
                        bs[j]                 = static_cast<double>(bc[j]) * static_cast<double>(exact_val);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int b = (kR2Bins - 1) - (tx * 4 + j);
                    bs[j]       = static_cast<double>(smem.u.r2.sum_hist[b]);
                    bc[j]       = smem.u.r2.cnt_hist[b];
                }
            }
            __syncthreads();

            const auto   m        = static_cast<RenormCutoffMode>(smem.mode);
            const bool   check_k  = (m == RenormCutoffMode::kTopKOnly || m == RenormCutoffMode::kBoth);
            const bool   p_active = (smem.eff_p_d > 0.0 && smem.eff_p_d < static_cast<double>(smem.row_sum));
            const bool   check_p  = p_active || (m == RenormCutoffMode::kPickMaxInBucket);
            const double rem_p_d  = (m == RenormCutoffMode::kPickMaxInBucket) ? 0.0 : (smem.eff_p_d - smem.above_sum_d);
            const uint32_t rem_k  = smem.rem_k;
            const bool     fallback = (m != RenormCutoffMode::kTopKOnly);

            const double target_run_d = scanBucketsFP64<4, kR2Bins>(
                smem.sc, smem.scan_d, smem.scan_u32, bs, bc, tx, check_k, check_p, rem_p_d, rem_k, fallback);

            if (leaf) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int b = (kR3Bins - 1) - (tx * 4 + j);
                    if (b == smem.sc.b_target) {
                        smem.leaf_target_sum_d = target_run_d;
                    }
                }
                __syncthreads();
                if (tx == 0) {
                    const uint32_t exact_bits = lo + static_cast<uint32_t>(smem.sc.b_target);
                    smem.cutoff               = __uint_as_float(exact_bits);
                    smem.kept_sum             = static_cast<float>(smem.above_sum_d + smem.leaf_target_sum_d);
                    smem.done_cutoff          = 1u;
                }
                __syncthreads();
            } else {
                if (tx == 0) {
                    uint32_t next_lo = lo;
                    uint32_t next_hi = hi;
                    int32_t  next_sh = 0;
                    refineSubRange(lo, hi, shift, static_cast<uint32_t>(smem.sc.b_target), next_lo, next_hi, next_sh);
                    smem.range_lo   = next_lo;
                    smem.range_hi   = next_hi;
                    smem.shift_cur  = next_sh;
                    smem.b_target   = smem.sc.b_target;
                    smem.cnt_target = smem.sc.cnt_target;
                    smem.rem_k      = smem.sc.rem_k_out;
                    smem.mode       = smem.sc.mode_out;
                }
                __syncthreads();
            }
        }
    }

    const uint8_t early      = smem.early_done;
    const float   cutoff     = smem.cutoff;
    const float   denom      = (smem.kept_sum <= 1e-8f) ? 1.0f : smem.kept_sum;
    const float   normalizer = 1.0f / denom;

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t base_col = (iter * kFastRenormThreads + tx) * VEC_SIZE;
        if (base_col >= d) {
            continue;
        }
        if (early == 1u) {
            if constexpr (VEC_SIZE == 4) {
                *reinterpret_cast<float4*>(row_out + base_col) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                row_out[base_col] = 0.0f;
            }
        } else if (early == 2u) {
            if constexpr (VEC_SIZE == 4) {
                float4 v                                       = *reinterpret_cast<const float4*>(row_probs + base_col);
                v.x                                            = (v.x > 0.0f) ? (v.x * normalizer) : 0.0f;
                v.y                                            = (v.y > 0.0f) ? (v.y * normalizer) : 0.0f;
                v.z                                            = (v.z > 0.0f) ? (v.z * normalizer) : 0.0f;
                v.w                                            = (v.w > 0.0f) ? (v.w * normalizer) : 0.0f;
                *reinterpret_cast<float4*>(row_out + base_col) = v;
            } else {
                const float v     = row_probs[base_col];
                row_out[base_col] = (v > 0.0f) ? (v * normalizer) : 0.0f;
            }
        } else {
            if constexpr (VEC_SIZE == 4) {
                float4 v = *reinterpret_cast<const float4*>(row_probs + base_col);
                v.x      = (v.x >= cutoff && v.x > 0.0f) ? (v.x * normalizer) : 0.0f;
                v.y      = (v.y >= cutoff && v.y > 0.0f) ? (v.y * normalizer) : 0.0f;
                v.z      = (v.z >= cutoff && v.z > 0.0f) ? (v.z * normalizer) : 0.0f;
                v.w      = (v.w >= cutoff && v.w > 0.0f) ? (v.w * normalizer) : 0.0f;
                *reinterpret_cast<float4*>(row_out + base_col) = v;
            } else {
                const float v     = row_probs[base_col];
                row_out[base_col] = (v >= cutoff && v > 0.0f) ? (v * normalizer) : 0.0f;
            }
        }
    }
}

// ============================================================================
// Public Launch Entrypoints
// ============================================================================

template<bool FUSED_TOPK>
inline cudaError_t FastRenormProb(const float*   probs,
                                  float*         renormed_prob,
                                  const int32_t* top_k_arr,
                                  int32_t        top_k_val,
                                  const float*   top_p_arr,
                                  float          top_p_val,
                                  uint32_t       batch_size,
                                  uint32_t       vocab_size,
                                  void*          workspace       = nullptr,
                                  size_t         workspace_bytes = 0,
                                  cudaStream_t   stream          = 0) {
    if (batch_size == 0 || vocab_size == 0) {
        return cudaSuccess;
    }
    const bool aligned_vec4 = (vocab_size % 4 == 0) && (reinterpret_cast<uintptr_t>(probs) % 16 == 0)
                              && (reinterpret_cast<uintptr_t>(renormed_prob) % 16 == 0);
    const uint32_t c_per_row    = FastRenormCTAsPerRow(batch_size, vocab_size);
    const size_t   needed_bytes = FastRenormWorkspaceBytes(batch_size, vocab_size);
    if (c_per_row > 1u && workspace != nullptr && workspace_bytes >= needed_bytes) {
        auto       ws = makeWorkspaceViews(workspace, batch_size, c_per_row);
        const dim3 grid_map(c_per_row, batch_size);
        const dim3 grid_red(batch_size);
        if (aligned_vec4) {
            FastRenormMapR1Kernel<4><<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
            FastRenormReduceR1Kernel<FUSED_TOPK, false><<<grid_red, kFastRenormThreads, 0, stream>>>(
                ws, top_k_arr, top_k_val, top_p_arr, top_p_val, vocab_size, c_per_row);
            for (int r = 0; r < 3; ++r) {
                FastRenormMapRangeOrCollectKernel<4>
                    <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
                FastRenormReduceRangeOrFinishKernel<<<grid_red, kFastRenormThreads, 0, stream>>>(ws, c_per_row);
            }
            FastRenormNormalizeKernel<4>
                <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, renormed_prob, ws, vocab_size, c_per_row);
        } else {
            FastRenormMapR1Kernel<1><<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
            FastRenormReduceR1Kernel<FUSED_TOPK, false><<<grid_red, kFastRenormThreads, 0, stream>>>(
                ws, top_k_arr, top_k_val, top_p_arr, top_p_val, vocab_size, c_per_row);
            for (int r = 0; r < 3; ++r) {
                FastRenormMapRangeOrCollectKernel<1>
                    <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
                FastRenormReduceRangeOrFinishKernel<<<grid_red, kFastRenormThreads, 0, stream>>>(ws, c_per_row);
            }
            FastRenormNormalizeKernel<1>
                <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, renormed_prob, ws, vocab_size, c_per_row);
        }
        return cudaGetLastError();
    }

    if (aligned_vec4) {
        FastRenormKernel<FUSED_TOPK, 4><<<batch_size, kFastRenormThreads, 0, stream>>>(
            probs, renormed_prob, top_k_arr, top_k_val, top_p_arr, top_p_val, vocab_size);
    } else {
        FastRenormKernel<FUSED_TOPK, 1><<<batch_size, kFastRenormThreads, 0, stream>>>(
            probs, renormed_prob, top_k_arr, top_k_val, top_p_arr, top_p_val, vocab_size);
    }
    return cudaGetLastError();
}

template<bool FUSED_TOPK, bool WRITE_RENORM, typename IdType>
inline cudaError_t FastRenormOrSampleProb(const float*   probs,
                                          float*         renormed_prob,
                                          IdType*        output,
                                          bool*          valid,
                                          const int32_t* top_k_arr,
                                          int32_t        top_k_val,
                                          const float*   top_p_arr,
                                          float          top_p_val,
                                          uint32_t       batch_size,
                                          uint32_t       vocab_size,
                                          uint64_t*      seed_arr,
                                          uint64_t       seed_val,
                                          uint64_t*      offset_arr,
                                          uint64_t       offset_val,
                                          void*          workspace,
                                          size_t         workspace_bytes,
                                          cudaStream_t   stream) {
    if (batch_size == 0 || vocab_size == 0) {
        return cudaSuccess;
    }
    const uint32_t c_per_row    = FastRenormCTAsPerRow(batch_size, vocab_size);
    const size_t   needed_bytes = FastRenormWorkspaceBytes(batch_size, vocab_size);
    if (c_per_row <= 1u || !FastSampleSupportsVocab(vocab_size) || workspace == nullptr
        || workspace_bytes < needed_bytes) {
        return cudaErrorInvalidConfiguration;
    }

    auto       ws = makeWorkspaceViews(workspace, batch_size, c_per_row);
    const dim3 grid_map(c_per_row, batch_size);
    const dim3 grid_red(batch_size);
    const bool aligned_vec4_in = (vocab_size % 4 == 0) && (reinterpret_cast<uintptr_t>(probs) % 16 == 0);
    const bool aligned_vec4_rw =
        aligned_vec4_in && (!WRITE_RENORM || (reinterpret_cast<uintptr_t>(renormed_prob) % 16 == 0));

    if (aligned_vec4_in) {
        FastRenormMapR1Kernel<4><<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
    } else {
        FastRenormMapR1Kernel<1><<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
    }
    FastRenormReduceR1Kernel<FUSED_TOPK, true><<<grid_red, kFastRenormThreads, 0, stream>>>(
        ws, top_k_arr, top_k_val, top_p_arr, top_p_val, vocab_size, c_per_row);
    for (int r = 0; r < 3; ++r) {
        if (aligned_vec4_in) {
            FastRenormMapRangeOrCollectKernel<4>
                <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
        } else {
            FastRenormMapRangeOrCollectKernel<1>
                <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, ws, vocab_size, c_per_row);
        }
        FastRenormReduceRangeOrFinishKernel<<<grid_red, kFastRenormThreads, 0, stream>>>(ws, c_per_row);
    }

    if (aligned_vec4_rw) {
        FastSampleChunkTileSumsKernel<WRITE_RENORM, 4>
            <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, renormed_prob, ws, vocab_size, c_per_row);
    } else {
        FastSampleChunkTileSumsKernel<WRITE_RENORM, 1>
            <<<grid_map, kFastRenormThreads, 0, stream>>>(probs, renormed_prob, ws, vocab_size, c_per_row);
    }
    FastSamplePickTileKernel<IdType><<<grid_red, kFastRenormThreads, 0, stream>>>(
        probs, output, valid, ws, vocab_size, seed_arr, seed_val, offset_arr, offset_val);
    return cudaGetLastError();
}

}  // namespace fast_renorm
}  // namespace rtp_llm
