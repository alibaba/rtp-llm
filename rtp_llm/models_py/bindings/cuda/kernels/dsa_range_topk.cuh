// SPDX-License-Identifier: Apache-2.0
// GLM deterministic radix selection, alibaba/rtp-llm 15ae7b0d652d.
// Adapted bounded reads and true global-position tie keys.
#include <math_constants.h>
#pragma once

#include "topk_v3.cuh"

#include <cfloat>
#include <cstdint>
#include <limits>

namespace device::k31_range_topk {

namespace base = device::topk;

// DSA candidate selection uses exact bin bounds without changing the existing
// DSV4 TopK implementation or its emission order.
constexpr float infinity_value() {
    return std::numeric_limits<float>::infinity();
}

// FP32 bin bounds reused from SGLang 5e79333c95c162abf079548fe40a6a2bca6c5409.
// Includes round-to-even and signed-zero handling; no score quantization.
template<uint32_t kBits>
SGL_DEVICE float2 coarse_bin_bounds(uint32_t bin) {
    constexpr uint32_t kShift = 16 - kBits;
    static_assert(kShift >= 1, "edge keys must be even for the rounding rule below");
    constexpr uint32_t kNegInfBin = 0x03FFu >> kShift;  // bin holding the -inf key
    constexpr uint32_t kInfBin    = 0xFC00u >> kShift;  // bin holding the +inf key
    // ordered16 edge keys, {lo, hi} packed in the {low, high} halves.
    const uint32_t keys = (bin << kShift) | ((bin + 1) << (kShift + 16));
    // ordered16 -> fp16 bits per half: ~key below 0x8000 (negative), key ^ 0x8000 above.
    const auto to_fp32 = [](uint32_t k) {
        const uint32_t bits = k ^ 0x80008000u ^ ((~k >> 15) & 0x00010001u) * 0x7FFFu;
        return __half22float2(*reinterpret_cast<const __half2*>(&bits));
    };
    const auto [v_lo, v_hi] = to_fp32(keys);
    const auto [u_lo, u_hi] = to_fp32(keys - 0x00010001u);
    // Exact midpoints of adjacent fp16 values; the clamp puts one against inf on the
    // +/-65520 overflow point.
    const auto    midpoint = [](float a, float b) { return fminf(fmaxf(0.5f * (a + b), -65520.0f), 65520.0f); };
    const int32_t lo       = __float_as_int(midpoint(v_lo, u_lo));
    const int32_t hi       = __float_as_int(midpoint(v_hi, u_hi));
    // Ties round to even and edge keys are even, so negative edges sit one ulp above
    // the midpoint. A zero hi steps past +0.
    const float lo_edge = __int_as_float(lo + (lo >> 31));
    const float hi_edge = __int_as_float(hi + (hi >> 31) + (hi == 0));
    // Edges inside the NaN key space: every score is >= lo, none is >= hi.
    return {bin <= kNegInfBin ? -infinity_value() : lo_edge, bin >= kInfBin ? infinity_value() : hi_edge};
}

template<uint32_t kBits>
struct DsaRadixBase: base::TopKRadixBase<kBits> {
    using Parent = base::TopKRadixBase<kBits>;
    using typename Parent::vec_t;
    static constexpr uint32_t kVecSize   = Parent::kVecSize;
    static constexpr uint32_t kBlockSize = Parent::kBlockSize;
    template<bool kAlignedInput, typename Address, typename F>
    SGL_DEVICE static void for_each_input_address(Address address, uint32_t seq_len, F&& fn) {
        const auto     tx       = threadIdx.x;
        const uint32_t num_full = seq_len / kVecSize;  // fully-in-bounds vectors

        vec_t    next_vec;
        uint32_t vi = tx;
        if (vi < num_full) {
            if constexpr (kAlignedInput) {
                next_vec.load(address(vi * kVecSize), 0);
            } else {
                next_vec.load_unaligned(address(vi * kVecSize), 0);
            }
        }
        while (vi < num_full) {
            const auto cur  = next_vec;
            const auto base = vi * kVecSize;
            vi += kBlockSize;
            if (vi < num_full) {
                if constexpr (kAlignedInput) {
                    next_vec.load(address(vi * kVecSize), 0);
                } else {
                    next_vec.load_unaligned(address(vi * kVecSize), 0);
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j) {
                fn(cur[j], base + j);
            }
        }

        // Tail: at most one partial vector, `rem` in [0, kVecSize).
        static_assert(kVecSize <= kBlockSize);  // ensure tail correctness
        const uint32_t tail_start = num_full * kVecSize;
        if (tx < seq_len - tail_start) {
            const auto idx = tail_start + tx;
            fn(*address(idx), idx);
        }
    }
};

// The GLM radix algorithm consumes this request-local span. Input holes and
// non-finite scores are canonicalized in registers, never in the score matrix.
struct DenseScoreInput {
    const float*            scores;
    SGL_DEVICE const float* pointer(uint32_t index) const {
        return scores + index;
    }
    SGL_DEVICE int32_t position(const int32_t* positions, uint32_t index) const {
        return positions[index];
    }
};

struct PageRRScoreInput {
    const float* scores;
    uint64_t     owner_stride;
    uint32_t     page_shift, owner_shift, page_mask, owner_mask;
    int32_t      first_position;

    // Visit logical columns in the same order as the dense selector.
    SGL_DEVICE const float* pointer(uint32_t index) const {
        const uint32_t page   = index >> page_shift;
        const uint32_t owner  = page & owner_mask;
        const uint32_t column = ((page >> owner_shift) << page_shift) | (index & page_mask);
        return scores + owner * owner_stride + column;
    }
    SGL_DEVICE int32_t position(const int32_t*, uint32_t index) const {
        return first_position + index;
    }
};

template<typename Input>
struct TopKProblemWithInput {
    Input                   input;
    const int32_t*          positions;
    float*                  output_scores;
    int32_t*                out;
    uint32_t                topk;
    uint32_t                seq_len;
    SGL_DEVICE const float* score_ptr(uint32_t index) const {
        return input.pointer(index);
    }
    SGL_DEVICE int32_t position(uint32_t index) const {
        return input.position(positions, index);
    }
    SGL_DEVICE float normalize(float value, uint32_t idx) const {
        const bool valid = (__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u && position(idx) >= 0;
        return valid ? (value == 0.0f ? 0.0f : value) : -CUDART_INF_F;
    }
    SGL_DEVICE float load(uint32_t idx) const {
        return normalize(*score_ptr(idx), idx);
    }
    SGL_DEVICE uint32_t tie_position(uint32_t idx) const {
        const float value = *score_ptr(idx);
        return ((__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u && position(idx) >= 0) ?
                   static_cast<uint32_t>(position(idx)) :
                   idx;
    }
    SGL_DEVICE void emit(uint32_t slot, uint32_t idx) const {
        if (slot >= topk)
            return;
        const float value   = *score_ptr(idx);
        const bool  valid   = (__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u && position(idx) >= 0;
        out[slot]           = valid ? position(idx) : -1;
        output_scores[slot] = valid ? value : -CUDART_INF_F;
    }
};

using TopKProblem       = TopKProblemWithInput<DenseScoreInput>;
using PageRRTopKProblem = TopKProblemWithInput<PageRRScoreInput>;

using base::TieValue;

inline constexpr uint32_t kBlockSize = base::TopKConfig::kBlockSize;
inline constexpr uint32_t kMaxNumTie = base::TopKConfig::kMaxNumTie;
inline constexpr uint32_t kRadixSize = base::TopKConfig::kRadixSize;
inline constexpr uint32_t kWarpSize  = base::kWarpSize;

// Keep the exact score ordering identical to
// dsv4_top_k_per_row_prefill::make_deterministic_sort_key(). In particular,
// +0 (0x80000000) ranks above -0 (0x7fffffff); only bitwise-equal scores reach
// the index tie-break.
SGL_DEVICE uint32_t stable_score_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template<typename Predicate, typename KeyFn, typename Problem>
SGL_DEVICE uint32_t radix_select_problem(const Problem&                            problem,
                                         uint32_t                                  rank,
                                         typename base::TopKConfig::TieHandleSmem* smem,
                                         Predicate                                 predicate,
                                         KeyFn                                     key_fn) {
    const uint32_t tx        = threadIdx.x;
    uint32_t       prefix    = 0;
    uint32_t       remaining = rank;

#pragma unroll
    for (uint32_t round = 0; round < 4; ++round) {
        if (tx < kRadixSize) {
            smem->histogram[0][tx] = 0;
        }
        __syncthreads();

        const uint32_t shift = 24u - round * 8u;
        const uint32_t mask  = round == 0 ? 0u : (~0u << (32u - round * 8u));
        for (uint32_t idx = tx; idx < problem.seq_len; idx += kBlockSize) {
            const float    value = problem.load(idx);
            const uint32_t key   = key_fn(value, idx);
            if (predicate(value, idx) && (key & mask) == prefix) {
                atomicAdd(&smem->histogram[0][(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

#pragma unroll
        for (uint32_t step = 0; step < 8; ++step) {
            if (tx < kRadixSize) {
                const uint32_t distance = 1u << step;
                const uint32_t src      = step & 1u;
                const uint32_t dst      = src ^ 1u;
                uint32_t       value    = smem->histogram[src][tx];
                if (tx + distance < kRadixSize) {
                    value += smem->histogram[src][tx + distance];
                }
                smem->histogram[dst][tx] = value;
            }
            __syncthreads();
        }

        if (tx < kRadixSize) {
            const uint32_t count_ge = smem->histogram[0][tx];
            const uint32_t count_gt = tx + 1u < kRadixSize ? smem->histogram[0][tx + 1u] : 0u;
            if (count_ge >= remaining && count_gt < remaining) {
                smem->match.bin = tx;
            }
        }
        __syncthreads();
        if (tx == 0) {
            const uint32_t threshold = smem->match.bin;
            const uint32_t count_gt  = threshold + 1u < kRadixSize ? smem->histogram[0][threshold + 1u] : 0u;
            prefix |= threshold << shift;
            smem->match.above_count = prefix;
            smem->match.equal_count = remaining - count_gt;
        }
        __syncthreads();
        prefix    = smem->match.above_count;
        remaining = smem->match.equal_count;
    }
    return prefix;
}

template<uint32_t kItems, typename Problem>
SGL_DEVICE void stable_handle_tie_impl(const TieValue*                           tie_buffer,
                                       const Problem&                            problem,
                                       uint32_t                                  base_out,
                                       uint32_t                                  num_ties,
                                       uint32_t                                  remaining_topk,
                                       typename base::TopKConfig::TieHandleSmem* smem) {
    const uint32_t tx      = threadIdx.x;
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;

    bool     active[kItems];
    uint32_t score_key[kItems];
    uint32_t idx[kItems];
    uint32_t write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
        const uint32_t t   = tx + i * kBlockSize;
        active[i]          = t < num_ties;
        const TieValue tie = active[i] ? tie_buffer[t] : TieValue::invalid();
        score_key[i]       = stable_score_key(tie.value);
        idx[i]             = tie.idx;
        write_pos[i]       = remaining_topk;
    }

    if (tx < kRadixSize) {
        smem->histogram[0][tx] = 0;
    }
    if (tx == kRadixSize) {
        smem->counter       = 0;
        smem->counter_final = 0;
    }
    __syncthreads();

    uint32_t score_rank   = remaining_topk;
    uint32_t total_active = num_ties;
#pragma unroll
    for (uint32_t round = 0; round < 4; ++round) {
        const uint32_t shift     = 24u - round * 8u;
        const uint32_t hist_idx  = round & 1u;
        uint32_t*      histogram = smem->histogram[hist_idx];
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            if (active[i]) {
                atomicAdd(&histogram[(score_key[i] >> shift) & 0xffu], 1u);
            }
        }
        if (round < 3 && tx < kRadixSize) {
            smem->histogram[hist_idx ^ 1u][tx] = 0;
        }
        __syncthreads();

        uint32_t hist_value  = 0;
        uint32_t warp_prefix = 0;
        if (tx < kRadixSize) {
            hist_value  = histogram[tx];
            warp_prefix = base::warp_inclusive_sum(lane_id, hist_value);
            if (lane_id == kWarpSize - 1u) {
                smem->warp_sum[warp_id] = warp_prefix;
            }
        }
        __syncthreads();
        if (tx < kRadixSize) {
            const uint32_t inter_warp = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0u);
            const uint32_t prefix     = inter_warp + warp_prefix;
            const uint32_t above      = total_active - prefix;
            if (above < score_rank && above + hist_value >= score_rank) {
                smem->match = {tx, above, hist_value};
            }
        }
        __syncthreads();

        const uint32_t threshold_bin = smem->match.bin;
        const uint32_t above_count   = smem->match.above_count;
        const uint32_t equal_count   = smem->match.equal_count;
        score_rank -= above_count;
        if (round < 3) {
            total_active = equal_count;
        }
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            if (!active[i]) {
                continue;
            }
            const uint32_t bin = (score_key[i] >> shift) & 0xffu;
            if (bin > threshold_bin) {
                write_pos[i] = atomicAdd(&smem->counter, 1u);
                active[i]    = false;
            } else if (bin < threshold_bin) {
                active[i] = false;
            }
        }
        if (score_rank == 0) {
            break;
        }
    }
    __syncthreads();

    const uint32_t exact_above = smem->counter;
    const uint32_t exact_equal = smem->match.equal_count;
    const uint32_t need_equal  = remaining_topk - exact_above;
    if (need_equal > 0 && exact_equal <= need_equal) {
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            if (active[i]) {
                write_pos[i] = exact_above + atomicAdd(&smem->counter_final, 1u);
            }
        }
    } else if (need_equal > 0) {
        if (tx < kRadixSize) {
            smem->histogram[0][tx] = 0;
        }
        if (tx == kRadixSize) {
            smem->counter_final = 0;
        }
        __syncthreads();

        uint32_t index_rank = need_equal;
        total_active        = exact_equal;
#pragma unroll
        for (uint32_t round = 0; round < 4; ++round) {
            const uint32_t shift     = 24u - round * 8u;
            const uint32_t hist_idx  = round & 1u;
            uint32_t*      histogram = smem->histogram[hist_idx];
#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i) {
                if (active[i]) {
                    const uint32_t index_key = 0xffffffffu - problem.tie_position(idx[i]);
                    atomicAdd(&histogram[(index_key >> shift) & 0xffu], 1u);
                }
            }
            if (round < 3 && tx < kRadixSize) {
                smem->histogram[hist_idx ^ 1u][tx] = 0;
            }
            __syncthreads();

            uint32_t hist_value  = 0;
            uint32_t warp_prefix = 0;
            if (tx < kRadixSize) {
                hist_value  = histogram[tx];
                warp_prefix = base::warp_inclusive_sum(lane_id, hist_value);
                if (lane_id == kWarpSize - 1u) {
                    smem->warp_sum[warp_id] = warp_prefix;
                }
            }
            __syncthreads();
            if (tx < kRadixSize) {
                const uint32_t inter_warp = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0u);
                const uint32_t prefix     = inter_warp + warp_prefix;
                const uint32_t above      = total_active - prefix;
                if (above < index_rank && above + hist_value >= index_rank) {
                    smem->match = {tx, above, hist_value};
                }
            }
            __syncthreads();

            const uint32_t threshold_bin = smem->match.bin;
            const uint32_t above_count   = smem->match.above_count;
            const uint32_t equal_count   = smem->match.equal_count;
            index_rank -= above_count;
            if (round < 3) {
                total_active = equal_count;
            }
#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i) {
                if (!active[i]) {
                    continue;
                }
                const uint32_t index_key = 0xffffffffu - problem.tie_position(idx[i]);
                const uint32_t bin       = (index_key >> shift) & 0xffu;
                if (bin > threshold_bin) {
                    write_pos[i] = exact_above + atomicAdd(&smem->counter_final, 1u);
                    active[i]    = false;
                } else if (bin < threshold_bin) {
                    active[i] = false;
                } else if (round == 3) {
                    write_pos[i] = exact_above + atomicAdd(&smem->counter_final, 1u);
                    active[i]    = false;
                }
            }
            if (index_rank == 0) {
                break;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
        if (write_pos[i] < remaining_topk) {
            problem.emit(base_out + write_pos[i], idx[i]);
        }
    }
}

// Resolve the candidates in the coarse threshold bin. Scores above the exact
// threshold are all selected. If only part of the exact-threshold group fits,
// select its smallest relative indices using a second four-round radix pass on
// index_key = UINT32_MAX - idx.
template<typename Problem>
SGL_DEVICE void stable_handle_tie(const TieValue*                           tie_buffer,
                                  const Problem&                            problem,
                                  uint32_t                                  base_out,
                                  uint32_t                                  num_ties,
                                  uint32_t                                  remaining_topk,
                                  typename base::TopKConfig::TieHandleSmem* smem) {
    const uint32_t tx = threadIdx.x;
    if (remaining_topk == 0) {
        return;
    }

    // If the whole coarse threshold interval is needed, selecting every
    // candidate is already stable: no equal-score group is cut. This is the
    // common continuous-score path and avoids an unnecessary exact radix pass.
    // A correctly classified interval has num_ties == remaining_topk here;
    // retain defensive -1 padding if malformed metadata ever makes it smaller.
    if (num_ties <= remaining_topk) {
        for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
            problem.emit(base_out + i, tie_buffer[i].idx);
        }
        for (uint32_t i = num_ties + tx; i < remaining_topk; i += kBlockSize) {
            problem.out[base_out + i] = -1;
        }
        return;
    }
    if (num_ties <= kBlockSize) {
        stable_handle_tie_impl<1>(tie_buffer, problem, base_out, num_ties, remaining_topk, smem);
    } else {
        stable_handle_tie_impl<base::TopKConfig::kTieItems>(
            tie_buffer, problem, base_out, num_ties, remaining_topk, smem);
    }
}

template<uint32_t kHistBits>
SGL_DEVICE bool in_threshold_bin(float value, uint32_t threshold_bin) {
    // The first pass builds the histogram with extract_coarse_bin(). Reuse the
    // exact same classification here instead of reconstructing it with FP32
    // comparisons. IEEE comparisons consider +0 and -0 equal, whereas their
    // ordered coarse keys are adjacent; using >= here can therefore turn the
    // -0 threshold bin into "above" and bypass stable_handle_tie().
    return base::extract_coarse_bin<kHistBits>(value) == threshold_bin;
}

template<uint32_t kHistBits>
SGL_DEVICE bool is_signed_zero_threshold_bin(uint32_t threshold_bin) {
    constexpr uint32_t kShift           = 16u - kHistBits;
    constexpr uint32_t kNegativeZeroBin = 0x7fffu >> kShift;
    constexpr uint32_t kPositiveZeroBin = 0x8000u >> kShift;
    return threshold_bin - kNegativeZeroBin <= kPositiveZeroBin - kNegativeZeroBin;
}

template<uint32_t kHistBits>
SGL_DEVICE bool needs_exact_coarse_classification(uint32_t threshold_bin) {
    constexpr uint32_t kShift = 16u - kHistBits;
    // The edge bins touch +/-inf or the canonical NaN bin. Numeric comparisons
    // cannot reproduce their coarse-key classification.
    constexpr uint32_t kNegativeInfBin      = 0x03ffu >> kShift;
    constexpr uint32_t kPositiveInfBin      = 0xfc00u >> kShift;
    constexpr uint32_t kFiniteInteriorBegin = kNegativeInfBin + 1u;
    constexpr uint32_t kFiniteInteriorSize  = kPositiveInfBin - kFiniteInteriorBegin;
    // Unsigned wrap makes bins <= kNegativeInfBin land above the interior
    // size, combining the two edge comparisons into one.
    return is_signed_zero_threshold_bin<kHistBits>(threshold_bin)
           || threshold_bin - kFiniteInteriorBegin >= kFiniteInteriorSize;
}

// Candidate-buffer overflow fallback. It performs the same stable score then
// index radix selection while rescanning the coarse threshold interval, or the
// whole row for the non-monotonic canonical NaN bin.
template<uint32_t kHistBits, typename Problem>
SGL_DEVICE void stable_exact_boundary_scan_topk(const Problem&                            problem,
                                                typename base::TopKConfig::TieHandleSmem* smem,
                                                uint32_t                                  threshold_bin,
                                                uint32_t                                  output_base) {
    const uint32_t tx = threadIdx.x;
    if (output_base >= problem.topk) {
        return;
    }
    const uint32_t remaining_topk = problem.topk - output_base;
    const auto     in_interval    = [threshold_bin](float value, uint32_t) {
        // Match per-row radix: FP32->FP16 canonicalizes every NaN sign/payload
        // into the highest coarse bin. If that bin exceeds its 2048-candidate
        // buffer, per-row restarts radix selection from all row inputs so
        // finite values can outrank negative NaNs by their exact score key.
        // Restricting the rescan to the non-monotonic NaN bin would lose them.
        if (threshold_bin == (1u << kHistBits) - 1u) {
            return true;
        }
        return in_threshold_bin<kHistBits>(value, threshold_bin);
    };
    const uint32_t threshold_score_key = radix_select_problem(
        problem, remaining_topk, smem, in_interval, [](float value, uint32_t) { return stable_score_key(value); });

    if (tx == 0) {
        smem->counter       = 0;
        smem->counter_final = 0;
    }
    __syncthreads();
    for (uint32_t idx = tx; idx < problem.seq_len; idx += kBlockSize) {
        const float value = problem.load(idx);
        if (!in_interval(value, idx)) {
            continue;
        }
        const uint32_t key = stable_score_key(value);
        if (key > threshold_score_key) {
            atomicAdd(&smem->counter, 1u);
        } else if (key == threshold_score_key) {
            atomicAdd(&smem->counter_final, 1u);
        }
    }
    __syncthreads();
    const uint32_t exact_above = smem->counter;
    const uint32_t exact_equal = smem->counter_final;
    const uint32_t need_equal  = remaining_topk - exact_above;
    // All warps must snapshot the count before thread 0 reuses it for
    // emission. Otherwise later warps can observe zero and overwrite the
    // strictly-greater outputs when they emit the threshold-score ties.
    __syncthreads();

    if (tx == 0) {
        smem->counter = 0;
    }
    __syncthreads();
    for (uint32_t idx = tx; idx < problem.seq_len; idx += kBlockSize) {
        const float value = problem.load(idx);
        if (in_interval(value, idx) && stable_score_key(value) > threshold_score_key) {
            const uint32_t pos = atomicAdd(&smem->counter, 1u);
            problem.emit(output_base + pos, idx);
        }
    }
    __syncthreads();

    if (exact_equal <= need_equal) {
        if (tx == 0) {
            smem->counter_final = 0;
        }
        __syncthreads();
        for (uint32_t idx = tx; idx < problem.seq_len; idx += kBlockSize) {
            const float value = problem.load(idx);
            if (in_interval(value, idx) && stable_score_key(value) == threshold_score_key) {
                const uint32_t pos = atomicAdd(&smem->counter_final, 1u);
                problem.emit(output_base + exact_above + pos, idx);
            }
        }
        return;
    }

    const uint32_t threshold_index_key = radix_select_problem(
        problem,
        need_equal,
        smem,
        [=](float value, uint32_t idx) {
            return in_interval(value, idx) && stable_score_key(value) == threshold_score_key;
        },
        [&](float, uint32_t idx) { return 0xffffffffu - problem.tie_position(idx); });

    if (tx == 0) {
        smem->counter_final = 0;
    }
    __syncthreads();
    for (uint32_t idx = tx; idx < problem.seq_len; idx += kBlockSize) {
        const float value = problem.load(idx);
        if (in_interval(value, idx) && stable_score_key(value) == threshold_score_key
            && 0xffffffffu - problem.tie_position(idx) >= threshold_index_key) {
            const uint32_t pos = atomicAdd(&smem->counter_final, 1u);
            problem.emit(output_base + exact_above + pos, idx);
        }
    }
}

template<uint32_t kLocalVecs_>
struct TopKRegister: DsaRadixBase<12> {
    static constexpr uint32_t kLocalVecs = kLocalVecs_;
    static constexpr uint32_t kMaxSeqLen = kBlockSize * base::TopKRadixBase<12>::kVecSize * kLocalVecs;
    using Base                           = DsaRadixBase<12>;
    using Smem                           = Base::Smem;
    using vec_t                          = typename Base::vec_t;

    template<bool kAlignedInput, typename Problem>
    SGL_DEVICE static void forward(const Problem problem, void* raw_smem) {
        const uint32_t          tx   = threadIdx.x;
        auto*                   smem = static_cast<Smem*>(raw_smem);
        typename Smem::kHistVec hist_vec;
        hist_vec.fill(0);
        smem->hist_vecs[tx] = hist_vec;
        if (tx == 0) {
            smem->count_eq = 0;
            smem->count_gt = 0;
        }
        __syncthreads();

        constexpr uint32_t kVecSize   = Base::kVecSize;
        const uint32_t     num_full   = problem.seq_len / kVecSize;
        const uint32_t     tail_start = num_full * kVecSize;
        const uint32_t     tail       = problem.seq_len - tail_start;
        const bool         owns_tail  = tx >= kBlockSize - tail;
        float              tail_value = 0.0f;

        vec_t local_vecs[kLocalVecs];
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const uint32_t vi = tx + kBlockSize * i;
            if (vi >= num_full) {
                break;
            }
            if constexpr (kAlignedInput) {
                local_vecs[i].load(problem.score_ptr(vi * kVecSize), 0);
            } else {
                local_vecs[i].load_unaligned(problem.score_ptr(vi * kVecSize), 0);
            }
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j) {
                local_vecs[i][j] = problem.normalize(local_vecs[i][j], vi * kVecSize + j);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const uint32_t vi = tx + kBlockSize * i;
            if (vi >= num_full) {
                break;
            }
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j) {
                atomicAdd(&smem->histogram[base::extract_coarse_bin<Base::kHistBits>(local_vecs[i][j])], 1u);
            }
        }
        if (owns_tail) {
            const uint32_t idx = tail_start + tx - (kBlockSize - tail);
            tail_value         = problem.load(idx);
            atomicAdd(&smem->histogram[base::extract_coarse_bin<Base::kHistBits>(tail_value)], 1u);
        }
        __syncthreads();

        Base::find_threshold(problem.topk, problem.seq_len, smem);
        const uint32_t threshold_bin               = smem->threshold_bin;
        const auto     bounds                      = coarse_bin_bounds<Base::kHistBits>(threshold_bin);
        const float    v_hi                        = bounds.y;
        const float    v_lo                        = bounds.x;
        const bool     exact_coarse_classification = needs_exact_coarse_classification<Base::kHistBits>(threshold_bin);
        const uint32_t topk                        = problem.topk;
        const auto     collect_above               = [&](uint32_t idx) {
            const uint32_t pos = atomicAdd(&smem->count_gt, 1u);
            if (pos < topk) {
                problem.emit(pos, idx);
            }
        };
        const auto collect_equal = [&](float value, uint32_t idx) {
            const uint32_t pos = atomicAdd(&smem->count_eq, 1u);
            if (pos < kMaxNumTie) {
                smem->tie.values[pos] = {value, idx};
            }
        };
        const auto collect_by_bin = [&](float value, uint32_t idx) {
            const uint32_t bin = base::extract_coarse_bin<Base::kHistBits>(value);
            if (bin > threshold_bin) {
                collect_above(idx);
            } else if (bin == threshold_bin) {
                collect_equal(value, idx);
            }
        };
        const auto collect_by_boundary = [&](float value, uint32_t idx) {
            if (value >= v_hi) {
                collect_above(idx);
            } else if (value >= v_lo) {
                collect_equal(value, idx);
            } else if (base::is_nan(value)) {
                const uint32_t bin = base::extract_coarse_bin<Base::kHistBits>(value);
                if (bin > threshold_bin) {
                    collect_above(idx);
                } else if (bin == threshold_bin) {
                    collect_equal(value, idx);
                }
            }
        };
        const auto collect_inputs = [&](const auto& collect) {
#pragma unroll
            for (uint32_t i = 0; i < kLocalVecs; ++i) {
                const uint32_t vi = tx + kBlockSize * i;
                if (vi >= num_full) {
                    break;
                }
                const uint32_t base_idx = vi * kVecSize;
#pragma unroll
                for (uint32_t j = 0; j < kVecSize; ++j) {
                    collect(local_vecs[i][j], base_idx + j);
                }
            }
            if (owns_tail) {
                collect(tail_value, tail_start + tx - (kBlockSize - tail));
            }
        };
        if (__builtin_expect(exact_coarse_classification, false)) {
            collect_inputs(collect_by_bin);
        } else {
            collect_inputs(collect_by_boundary);
        }
        __syncthreads();

        const uint32_t above_count = smem->count_gt;
        const uint32_t equal_count = smem->count_eq;
        if (equal_count > kMaxNumTie) {
            if (above_count < topk) {
                stable_exact_boundary_scan_topk<Base::kHistBits>(
                    problem, &smem->tie.handle, threshold_bin, above_count);
            }
            return;
        }
        stable_handle_tie(smem->tie.values,
                          problem,
                          above_count,
                          equal_count,
                          above_count < topk ? topk - above_count : 0u,
                          &smem->tie.handle);
    }
};

struct TopKStreaming: TopKRegister<2> {
    static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

    template<bool kAlignedInput, typename Problem>
    SGL_DEVICE static void forward(const Problem problem, void* raw_smem) {
        const auto              address = [&](uint32_t index) { return problem.score_ptr(index); };
        const uint32_t          tx      = threadIdx.x;
        auto*                   smem    = static_cast<Smem*>(raw_smem);
        typename Smem::kHistVec hist_vec;
        hist_vec.fill(0);
        smem->hist_vecs[tx] = hist_vec;
        if (tx == 0) {
            smem->count_eq = 0;
            smem->count_gt = 0;
        }
        __syncthreads();

        Base::template for_each_input_address<kAlignedInput>(address, problem.seq_len, [&](float value, uint32_t idx) {
            value = problem.normalize(value, idx);
            atomicAdd(&smem->histogram[base::extract_coarse_bin<Base::kHistBits>(value)], 1u);
        });
        __syncthreads();
        Base::find_threshold(problem.topk, problem.seq_len, smem);

        const uint32_t threshold_bin               = smem->threshold_bin;
        const auto     bounds                      = coarse_bin_bounds<Base::kHistBits>(threshold_bin);
        const float    v_hi                        = bounds.y;
        const float    v_lo                        = bounds.x;
        const bool     exact_coarse_classification = needs_exact_coarse_classification<Base::kHistBits>(threshold_bin);
        const uint32_t topk                        = problem.topk;
        const auto     collect_above               = [&](uint32_t idx) {
            const uint32_t pos = atomicAdd(&smem->count_gt, 1u);
            if (pos < topk) {
                problem.emit(pos, idx);
            }
        };
        const auto collect_equal = [&](float value, uint32_t idx) {
            const uint32_t pos = atomicAdd(&smem->count_eq, 1u);
            if (pos < kMaxNumTie) {
                smem->tie.values[pos] = {value, idx};
            }
        };
        if (__builtin_expect(exact_coarse_classification, false)) {
            Base::template for_each_input_address<kAlignedInput>(
                address, problem.seq_len, [&](float value, uint32_t idx) {
                    value              = problem.normalize(value, idx);
                    const uint32_t bin = base::extract_coarse_bin<Base::kHistBits>(value);
                    if (bin > threshold_bin) {
                        collect_above(idx);
                    } else if (bin == threshold_bin) {
                        collect_equal(value, idx);
                    }
                });
        } else {
            Base::template for_each_input_address<kAlignedInput>(
                address, problem.seq_len, [&](float value, uint32_t idx) {
                    value = problem.normalize(value, idx);
                    if (value >= v_hi) {
                        collect_above(idx);
                    } else if (value >= v_lo) {
                        collect_equal(value, idx);
                    } else if (base::is_nan(value)) {
                        const uint32_t bin = base::extract_coarse_bin<Base::kHistBits>(value);
                        if (bin > threshold_bin) {
                            collect_above(idx);
                        } else if (bin == threshold_bin) {
                            collect_equal(value, idx);
                        }
                    }
                });
        }
        __syncthreads();

        const uint32_t above_count = smem->count_gt;
        const uint32_t equal_count = smem->count_eq;
        if (equal_count > kMaxNumTie) {
            if (above_count < topk) {
                stable_exact_boundary_scan_topk<Base::kHistBits>(
                    problem, &smem->tie.handle, threshold_bin, above_count);
            }
            return;
        }
        stable_handle_tie(smem->tie.values,
                          problem,
                          above_count,
                          equal_count,
                          above_count < topk ? topk - above_count : 0u,
                          &smem->tie.handle);
    }
};

}  // namespace device::k31_range_topk
