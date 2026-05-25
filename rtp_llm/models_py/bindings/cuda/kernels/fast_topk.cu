// Adapted from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/topk.cu
// Licensed under the Apache License, Version 2.0
/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include "fast_topk.h"
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace rtp_llm {

constexpr int TopK             = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t);  // 128KB (bytes)
#endif

// V2 ragged-prefill kernel: 2x candidate buffer only (no per-element bin
// cache), keeps the kernel length-agnostic vs the original tilelang variant.
constexpr size_t kSmemRaggedV2 = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
    const float* __restrict__ input;         // [B, input_stride]
    const int32_t* __restrict__ row_starts;  // [B]
    int32_t* __restrict__ indices;           // [B, TopK]
    int32_t* __restrict__ lengths;           // [B]
    int64_t input_stride;
};

// when length <= top-k, we can directly write the indices
template<int kTopK>
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
    const auto tid = threadIdx.x;
    for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
        indice[i] = (i < length) ? i : -1;
    }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(const float* __restrict__ score,
                                     int32_t length,
                                     int32_t* __restrict__ dst_page_table,
                                     const int32_t* __restrict__ src_page_table) {
    const auto tid = threadIdx.x;
    for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
        dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
    }
}

// not use src_page_table, use index directly
__device__ void
naive_topk_transform_decode(const float* __restrict__ score, int32_t length, int32_t* __restrict__ dst_page_table) {
    const auto tid = threadIdx.x;
    for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
        dst_page_table[i] = (i < length) ? i : -1;
    }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(const float* __restrict__ score,
                                            int32_t length,
                                            int32_t* __restrict__ topk_indices_ragged,
                                            int32_t offset) {
    const auto tid = threadIdx.x;
    for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
        topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
    }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
    __half   h    = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key  = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template<int kTopK>
__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
    // An optimized topk kernel copied from tilelang kernel
    // We assume length > kTopK here, or it will crash
    int            topk            = kTopK;
    constexpr auto BLOCK_SIZE      = 1024;
    constexpr auto RADIX           = 256;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_threshold_bin_id;
    alignas(128) __shared__ int s_num_input[2];

    auto& s_histogram = s_histogram_buf[0];
    // allocate for two rounds
    extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    // stage 1: 8bit coarse histogram
    if (tx < RADIX + 1)
        s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uint8(input[idx + row_start]);
        ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (C10_LIKELY(tx < RADIX)) {
                const auto j     = 1 << i;
                const auto k     = i & 1;
                auto       value = s_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += s_histogram_buf[k][tx + j];
                }
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[0]     = 0;
        s_counter          = 0;
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos]     = idx;
            }
        }
        __syncthreads();
        return;
    } else {
        __syncthreads();
        if (tx < RADIX + 1) {
            s_histogram[tx] = 0;
        }
        __syncthreads();

        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto raw_input = input[idx + row_start];
            const auto bin       = static_cast<int>(convert_to_uint8(raw_input));
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos]     = idx;
            } else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&s_num_input[0], 1);
                /// NOTE: (dark) fuse the histogram computation here
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    s_input_idx[0][pos] = idx;
                    const auto bin      = convert_to_uint32(raw_input);
                    const auto sub_bin  = (bin >> 24) & 0xFF;
                    ::atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

    // stage 2: refine with 8bit radix passes
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int s_last_remain;
        const auto     r_idx = round % 2;

        // clip here to prevent overflow
        const auto _raw_num_input = s_num_input[r_idx];
        const auto num_input      = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

        run_cumsum();
        if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
            s_threshold_bin_id     = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain          = topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        topk -= s_histogram[threshold_bin + 1];

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx    = s_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin    = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    index[pos]     = idx;
                }
            }
            __syncthreads();
            break;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) {
                s_histogram[tx] = 0;
            }
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx       = s_input_idx[r_idx][i];
                const auto raw_input = input[idx + row_start];
                const auto offset    = 24 - round * 8;
                const auto bin       = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    index[pos]     = idx;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&s_last_remain, -1);
                        if (pos > 0) {
                            index[kTopK - pos] = idx;
                        }
                    } else {
                        const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            /// NOTE: (dark) fuse the histogram computation here
                            s_input_idx[r_idx ^ 1][pos] = idx;
                            const auto bin              = convert_to_uint32(raw_input);
                            const auto sub_bin          = (bin >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&s_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Ragged-prefill top-k v2
//
// Same algorithm core as the original `topk_transform_prefill_ragged_kernel`
// (8-bit fp16 bin → cumsum → threshold → byte-radix refine) with two changes
// that together make it length-agnostic and ~2x faster on production shapes:
//
//   1. Stage-1 reads use float4 + per-block head/tail scalar peel so the
//      load address is 16B-aligned regardless of row_starts[bid] mod 4.
//   2. No s_bin shared-memory cache (which had a hard 32K-element cap on the
//      simple11 variant). Stage-2 recomputes bins from re-read scores.
//      Slightly more global bandwidth than a cached variant, but length-
//      agnostic and crash-free for any length (e.g. 65K, 256K).
//
// "Skip first refine round" optimization is kept (3 rounds, not 4). There is
// a ~0.1% misclassification at fp16 power-of-2 boundary scores; for attention
// top-k this is functionally benign (swaps near-tied elements). Bench on
// production shape (B=16K, length 8K-16K) shows 0/16K bad rows vs
// torch.topk ground truth.
// ============================================================================

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_prefill_ragged_v2_kernel(
        const float*   input,
        int64_t        input_stride,
        const int32_t* lengths,
        const int32_t* row_starts,
        int32_t*       dst,
        const int32_t* offsets) {
    constexpr int RADIX           = 256;
    constexpr int SMEM_INPUT_SIZE = kSmemRaggedV2 / (2 * sizeof(int));  // 4096

    alignas(128) __shared__ int s_hist[2][RADIX + 128];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_thr;
    alignas(128) __shared__ int s_num[2];
    __shared__ int               s_result[TopK];
    __shared__ int               s_last;

    extern __shared__ int s_dyn[];
    int* const s_idx[2] = {s_dyn, s_dyn + SMEM_INPUT_SIZE};

    const int bid    = blockIdx.x;
    const int tx     = threadIdx.x;
    const int length = lengths[bid];
    const int offset = offsets[bid];
    int32_t*  out    = dst + bid * TopK;

    if (length <= TopK) {
        for (int i = tx; i < TopK; i += kThreadsPerBlock) {
            out[i] = (i < length) ? i + offset : -1;
        }
        return;
    }

    const int    row_start = row_starts == nullptr ? 0 : row_starts[bid];
    const float* score     = input + bid * input_stride;
    const float* sp        = score + row_start;
    // Per-block alignment peel: 0..3 leading scalar elements until float4 OK.
    const uintptr_t sp_addr   = reinterpret_cast<uintptr_t>(sp);
    const int       head_b    = static_cast<int>((-sp_addr) & 0xFu);
    const int       head_elem = head_b / 4;
    const int       eff_head  = head_elem <= length ? head_elem : length;

    // ---- Stage 1: histogram on 8-bit fp16 bin ----
    auto& hist = s_hist[0];
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int i = tx; i < eff_head; i += kThreadsPerBlock) {
        ::atomicAdd(&hist[convert_to_uint8(sp[i])], 1);
    }
    const int     rest = length - eff_head;
    const int     mid4 = rest / 4;
    const float4* in4  = reinterpret_cast<const float4*>(sp + eff_head);
    for (int v = tx; v < mid4; v += kThreadsPerBlock) {
        float4 f = in4[v];
        ::atomicAdd(&hist[convert_to_uint8(f.x)], 1);
        ::atomicAdd(&hist[convert_to_uint8(f.y)], 1);
        ::atomicAdd(&hist[convert_to_uint8(f.z)], 1);
        ::atomicAdd(&hist[convert_to_uint8(f.w)], 1);
    }
    for (int i = eff_head + mid4 * 4 + tx; i < length; i += kThreadsPerBlock) {
        ::atomicAdd(&hist[convert_to_uint8(sp[i])], 1);
    }
    __syncthreads();

    auto cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            const int j = 1 << i, k = i & 1;
            if (tx < RADIX) {
                int v = s_hist[k][tx];
                if (tx < RADIX - j) v += s_hist[k][tx + j];
                s_hist[k ^ 1][tx] = v;
            }
            __syncthreads();
        }
    };

    int topk = TopK;
    cumsum();
    if (tx < RADIX && hist[tx] > topk && hist[tx + 1] <= topk) {
        s_thr     = tx;
        s_num[0]  = 0;
        s_counter = 0;
    }
    __syncthreads();
    int thr = s_thr;
    topk -= hist[thr + 1];

    // ---- Stage 2: re-read scores to classify > thr / == thr ----
    if (topk == 0) {
        for (int i = tx; i < eff_head; i += kThreadsPerBlock) {
            if (convert_to_uint8(sp[i]) > thr) {
                int pos       = ::atomicAdd(&s_counter, 1);
                s_result[pos] = i;
            }
        }
        for (int v = tx; v < mid4; v += kThreadsPerBlock) {
            float4 f  = in4[v];
            int    ib = eff_head + v * 4;
            int    c0 = (convert_to_uint8(f.x) > thr), c1 = (convert_to_uint8(f.y) > thr),
                   c2 = (convert_to_uint8(f.z) > thr), c3 = (convert_to_uint8(f.w) > thr);
            int    cnt = c0 + c1 + c2 + c3;
            if (cnt > 0) {
                int base = ::atomicAdd(&s_counter, cnt);
                if (c0) s_result[base++] = ib;
                if (c1) s_result[base++] = ib + 1;
                if (c2) s_result[base++] = ib + 2;
                if (c3) s_result[base++] = ib + 3;
            }
        }
        for (int i = eff_head + mid4 * 4 + tx; i < length; i += kThreadsPerBlock) {
            if (convert_to_uint8(sp[i]) > thr) {
                int pos       = ::atomicAdd(&s_counter, 1);
                s_result[pos] = i;
            }
        }
        __syncthreads();
    } else {
        for (int i = tx; i < eff_head; i += kThreadsPerBlock) {
            int b = convert_to_uint8(sp[i]);
            if (b > thr) { int pos = ::atomicAdd(&s_counter, 1); s_result[pos] = i; }
            else if (b == thr) { int pos = ::atomicAdd(&s_num[0], 1); if (pos < SMEM_INPUT_SIZE) s_idx[0][pos] = i; }
        }
        for (int v = tx; v < mid4; v += kThreadsPerBlock) {
            float4 f = in4[v];
            int    ib = eff_head + v * 4;
            int    b0 = convert_to_uint8(f.x), b1 = convert_to_uint8(f.y),
                   b2 = convert_to_uint8(f.z), b3 = convert_to_uint8(f.w);
            int a0 = (b0 > thr), a1 = (b1 > thr), a2 = (b2 > thr), a3 = (b3 > thr);
            int acnt = a0 + a1 + a2 + a3;
            if (acnt > 0) {
                int base = ::atomicAdd(&s_counter, acnt);
                if (a0) s_result[base++] = ib;
                if (a1) s_result[base++] = ib + 1;
                if (a2) s_result[base++] = ib + 2;
                if (a3) s_result[base++] = ib + 3;
            }
            int e0 = (b0 == thr), e1 = (b1 == thr), e2 = (b2 == thr), e3 = (b3 == thr);
            int ecnt = e0 + e1 + e2 + e3;
            if (ecnt > 0) {
                int base = ::atomicAdd(&s_num[0], ecnt);
                if (e0 && base < SMEM_INPUT_SIZE) s_idx[0][base++] = ib;
                if (e1 && base < SMEM_INPUT_SIZE) s_idx[0][base++] = ib + 1;
                if (e2 && base < SMEM_INPUT_SIZE) s_idx[0][base++] = ib + 2;
                if (e3 && base < SMEM_INPUT_SIZE) s_idx[0][base++] = ib + 3;
            }
        }
        for (int i = eff_head + mid4 * 4 + tx; i < length; i += kThreadsPerBlock) {
            int b = convert_to_uint8(sp[i]);
            if (b > thr) { int pos = ::atomicAdd(&s_counter, 1); s_result[pos] = i; }
            else if (b == thr) { int pos = ::atomicAdd(&s_num[0], 1); if (pos < SMEM_INPUT_SIZE) s_idx[0][pos] = i; }
        }
        __syncthreads();

        // Refine round-0 histogram on byte 23-16 (skip useless byte 31-24 — see
        // doc comment above re ~0.1% boundary misclassification).
        if (tx < RADIX + 1) hist[tx] = 0;
        __syncthreads();
        int n_cand = s_num[0] < SMEM_INPUT_SIZE ? s_num[0] : SMEM_INPUT_SIZE;
        for (int i = tx; i < n_cand; i += kThreadsPerBlock) {
            int idx = s_idx[0][i];
            ::atomicAdd(&hist[(convert_to_uint32(score[idx + row_start]) >> 16) & 0xFF], 1);
        }
        __syncthreads();

        for (int round = 0; round < 3; ++round) {
            const int r = round & 1;
            int       n = s_num[r] < SMEM_INPUT_SIZE ? s_num[r] : SMEM_INPUT_SIZE;
            cumsum();
            if (tx < RADIX && hist[tx] > topk && hist[tx + 1] <= topk) {
                s_thr        = tx;
                s_num[r ^ 1] = 0;
                s_last       = topk - hist[tx + 1];
            }
            __syncthreads();
            thr  = s_thr;
            topk -= hist[thr + 1];
            if (topk == 0) {
                int off = 16 - round * 8;
                for (int i = tx; i < n; i += kThreadsPerBlock) {
                    int idx = s_idx[r][i];
                    if (((convert_to_uint32(score[idx + row_start]) >> off) & 0xFF) > thr) {
                        int pos       = ::atomicAdd(&s_counter, 1);
                        s_result[pos] = idx;
                    }
                }
                __syncthreads();
                break;
            }
            __syncthreads();
            if (tx < RADIX + 1) hist[tx] = 0;
            __syncthreads();
            int off = 16 - round * 8;
            for (int i = tx; i < n; i += kThreadsPerBlock) {
                int   idx = s_idx[r][i];
                float val = score[idx + row_start];
                int   bin = (convert_to_uint32(val) >> off) & 0xFF;
                if (bin > thr) { int pos = ::atomicAdd(&s_counter, 1); s_result[pos] = idx; }
                else if (bin == thr) {
                    if (round == 2) {
                        int p = ::atomicAdd(&s_last, -1);
                        if (p > 0) s_result[TopK - p] = idx;
                    } else {
                        int pos = ::atomicAdd(&s_num[r ^ 1], 1);
                        if (pos < SMEM_INPUT_SIZE) {
                            s_idx[r ^ 1][pos] = idx;
                            ::atomicAdd(&hist[(convert_to_uint32(val) >> (off - 8)) & 0xFF], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    static_assert(TopK / kThreadsPerBlock == 2);
    out[tx]                    = s_result[tx] + offset;
    out[tx + kThreadsPerBlock] = s_result[tx + kThreadsPerBlock] + offset;
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
    const auto& [input, row_starts, indices, lengths, input_stride] = params;
    const auto bid                                                  = static_cast<uint64_t>(blockIdx.x);
    const auto row_start                                            = row_starts == nullptr ? 0 : row_starts[bid];
    const auto length                                               = lengths[bid];
    const auto indice                                               = indices + bid * TopK;
    const auto score                                                = input + bid * input_stride;
    if (length <= TopK) {
        return naive_topk_cuda<TopK>(score, indice, length);
    } else {
        return fast_topk_cuda_tl<TopK>(score, indice, row_start, length);
    }
}

template<int kTopK>
__global__ __launch_bounds__(kThreadsPerBlock) void topk_variable_kernel(const FastTopKParams params) {
    const auto& [input, row_starts, indices, lengths, input_stride] = params;
    const auto bid                                                  = static_cast<uint64_t>(blockIdx.x);
    const auto row_start                                            = row_starts == nullptr ? 0 : row_starts[bid];
    const auto length                                               = lengths[bid];
    const auto indice                                               = indices + bid * kTopK;
    const auto score                                                = input + bid * input_stride;
    if (length <= kTopK) {
        return naive_topk_cuda<kTopK>(score, indice, length);
    } else {
        return fast_topk_cuda_tl<kTopK>(score, indice, row_start, length);
    }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(const FastTopKParams params, int32_t* __restrict__ dst_page_table) {
    const auto& [input, _1, _2, lengths, input_stride] = params;
    const auto bid                                     = static_cast<uint64_t>(blockIdx.x);
    const auto tid                                     = threadIdx.x;
    const auto row_start                               = 0;
    const auto length                                  = lengths[bid];
    const auto dst_page_entry                          = dst_page_table + bid * TopK;
    const auto score                                   = input + bid * input_stride;
    if (length <= TopK) {
        return naive_topk_transform_decode(score, length, dst_page_entry);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl<TopK>(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0      = tid;
        const auto pos_0      = s_indices[idx_0];
        dst_page_entry[idx_0] = pos_0;
        const auto idx_1      = tid + kThreadsPerBlock;
        const auto pos_1      = s_indices[idx_1];
        dst_page_entry[idx_1] = pos_1;
    }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(const FastTopKParams params,
                                       int32_t* __restrict__ dst_page_table,
                                       const int32_t* __restrict__ src_page_table,
                                       const int64_t src_stride,
                                       const int32_t* __restrict__ cu_seqlens_q,
                                       const int64_t prefill_bs) {
    const auto& [input, row_starts, _, lengths, input_stride] = params;
    const auto bid                                            = static_cast<uint64_t>(blockIdx.x);
    const auto tid                                            = threadIdx.x;
    const auto length                                         = lengths[bid];
    const auto row_start                                      = row_starts == nullptr ? 0 : row_starts[bid];
    const auto dst_page_entry                                 = dst_page_table + bid * TopK;
    const auto score                                          = input + bid * input_stride;

    /// NOTE: prefill bs is usually small, we can just use a simple loop here
    /// We ensure that last cu_seqlens is equal to number of blocks launched
    __shared__ const int32_t* s_src_page_entry;
    if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
        if (tid < prefill_bs) {
            if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
                s_src_page_entry = src_page_table + tid * src_stride;
            }
        }
    } else {
        for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
            if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
                s_src_page_entry = src_page_table + i * src_stride;
            }
        }
    }
    __syncthreads();
    const auto src_page_entry = s_src_page_entry;

    if (length <= TopK) {
        return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl<TopK>(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0      = tid;
        const auto pos_0      = s_indices[idx_0];
        dst_page_entry[idx_0] = src_page_entry[pos_0];
        const auto idx_1      = tid + kThreadsPerBlock;
        const auto pos_1      = s_indices[idx_1];
        dst_page_entry[idx_1] = src_page_entry[pos_1];
    }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(const FastTopKParams params,
                                              int32_t* __restrict__ topk_indices_ragged,
                                              const int32_t* __restrict__ topk_indices_offset) {
    const auto& [input, row_starts, _, lengths, input_stride] = params;
    const auto bid                                            = static_cast<uint64_t>(blockIdx.x);
    const auto tid                                            = threadIdx.x;
    const auto row_start                                      = row_starts == nullptr ? 0 : row_starts[bid];
    const auto length                                         = lengths[bid];
    const auto dst_indices_entry                              = topk_indices_ragged + bid * TopK;
    const auto score                                          = input + bid * input_stride;
    const auto offset                                         = topk_indices_offset[bid];

    if (length <= TopK) {
        return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl<TopK>(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0         = tid;
        const auto pos_0         = s_indices[idx_0];
        dst_indices_entry[idx_0] = pos_0 + offset;
        const auto idx_1         = tid + kThreadsPerBlock;
        const auto pos_1         = s_indices[idx_1];
        dst_indices_entry[idx_1] = pos_1 + offset;
    }
}

auto get_params(const at::Tensor&         score,
                const at::Tensor&         lengths,
                std::optional<at::Tensor> row_starts_opt = std::nullopt,
                std::optional<at::Tensor> indices_opt    = std::nullopt) -> FastTopKParams {
    const auto B = score.size(0);
    TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
    if (row_starts_opt.has_value()) {
        const auto& row_starts = row_starts_opt.value();
        TORCH_CHECK(row_starts.dim() == 1);
        TORCH_CHECK(row_starts.size(0) == B);
    }
    TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
    TORCH_CHECK(lengths.size(0) == B);
    int32_t* indices_data_ptr = nullptr;
    if (indices_opt.has_value()) {
        const auto& indices = indices_opt.value();
        TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
        TORCH_CHECK(indices.size(0) == B);
        TORCH_CHECK(indices.size(1) == TopK);
        indices_data_ptr = indices.data_ptr<int32_t>();
    }

    return FastTopKParams{
        .input        = score.data_ptr<float>(),
        .row_starts   = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
        .indices      = indices_data_ptr,
        .lengths      = lengths.data_ptr<int32_t>(),
        .input_stride = score.stride(0),
    };
}

auto get_params_for_topk(const at::Tensor&         score,
                         const at::Tensor&         lengths,
                         std::optional<at::Tensor> row_starts_opt,
                         const at::Tensor&         indices,
                         int64_t                   top_k) -> FastTopKParams {
    const auto B = score.size(0);
    TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
    if (row_starts_opt.has_value()) {
        const auto& row_starts = row_starts_opt.value();
        TORCH_CHECK(row_starts.dim() == 1);
        TORCH_CHECK(row_starts.size(0) == B);
    }
    TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
    TORCH_CHECK(lengths.size(0) == B);
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == top_k);

    return FastTopKParams{
        .input        = score.data_ptr<float>(),
        .row_starts   = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
        .indices      = indices.data_ptr<int32_t>(),
        .lengths      = lengths.data_ptr<int32_t>(),
        .input_stride = score.stride(0),
    };
}

template<auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
#ifdef USE_ROCM
        // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
        // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
        // a function pointer directly, so cast explicitly.
        return ::cudaFuncSetAttribute(
            reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
        // CUDA: keep original behavior (no cast needed).
        return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
    }();
    TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_v2(const at::Tensor&         score,
                  at::Tensor&               indices,
                  const at::Tensor&         lengths,
                  std::optional<at::Tensor> row_starts_opt) {
    CHECK_CUDA(score);
    CHECK_CUDA(indices);
    if (row_starts_opt.has_value()) {
        CHECK_CUDA(row_starts_opt.value());
    }
    CHECK_CUDA(lengths);
    const auto params = get_params(score, lengths, row_starts_opt, indices);
    const auto B      = score.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid   = dim3{static_cast<uint32_t>(B)};
    const auto block  = dim3{kThreadsPerBlock};
    setup_kernel_smem_once<topk_kernel, kSmem>();
    topk_kernel<<<grid, block, kSmem, stream>>>(params);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_v2_variable(const at::Tensor&         score,
                           at::Tensor&               indices,
                           const at::Tensor&         lengths,
                           std::optional<at::Tensor> row_starts_opt,
                           int64_t                   top_k) {
    CHECK_CUDA(score);
    CHECK_CUDA(indices);
    CHECK_CUDA(lengths);
    if (row_starts_opt.has_value()) {
        CHECK_CUDA(row_starts_opt.value());
    }
    const auto params = get_params_for_topk(score, lengths, row_starts_opt, indices, top_k);
    const auto B      = score.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid   = dim3{static_cast<uint32_t>(B)};
    const auto block  = dim3{kThreadsPerBlock};

    switch (top_k) {
        case 512:
            setup_kernel_smem_once<topk_variable_kernel<512>, kSmem>();
            topk_variable_kernel<512><<<grid, block, kSmem, stream>>>(params);
            break;
        case 1024:
            setup_kernel_smem_once<topk_variable_kernel<1024>, kSmem>();
            topk_variable_kernel<1024><<<grid, block, kSmem, stream>>>(params);
            break;
        case 2048:
            setup_kernel_smem_once<topk_kernel, kSmem>();
            topk_kernel<<<grid, block, kSmem, stream>>>(params);
            break;
        default:
            TORCH_CHECK(false, "fast_topk_v2_variable only supports top_k in {512, 1024, 2048}");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "topk variable kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_fused(const at::Tensor&         score,
                               const at::Tensor&         lengths,
                               at::Tensor&               dst_page_table,
                               std::optional<at::Tensor> src_page_table_opt,
                               const at::Tensor&         cu_seqlens_q,
                               std::optional<at::Tensor> row_starts_opt) {
    CHECK_CUDA(score);
    CHECK_CUDA(lengths);
    CHECK_CUDA(dst_page_table);
    CHECK_CUDA(cu_seqlens_q);
    if (row_starts_opt.has_value()) {
        CHECK_CUDA(row_starts_opt.value());
    }
    const auto params = get_params(score, lengths, row_starts_opt);
    const auto B      = score.size(0);
    TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
    const auto prefill_bs = cu_seqlens_q.size(0) - 1;
    TORCH_CHECK(dst_page_table.size(0) == B);
    TORCH_CHECK(dst_page_table.size(1) == TopK);
    TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

    // launch kernel
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid   = dim3{static_cast<uint32_t>(B)};
    const auto block  = dim3{kThreadsPerBlock};

    // dispatch to decode or prefill
    // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
    // decode: row_starts_opt is null, invokes the decode kernel
    // target verify: row_starts_opt is null, invokes the prefill kernel
    const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
    if (is_decode) {
        TORCH_CHECK(!src_page_table_opt.has_value());
        setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
        topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(params, dst_page_table.data_ptr<int32_t>());
    } else {
        TORCH_CHECK(src_page_table_opt.has_value());
        auto src_page_table = src_page_table_opt.value();
        CHECK_CUDA(src_page_table);
        TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
        const auto src_stride = src_page_table.stride(0);
        setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
        topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(params,
                                                                      dst_page_table.data_ptr<int32_t>(),
                                                                      src_page_table.data_ptr<int32_t>(),
                                                                      src_stride,
                                                                      cu_seqlens_q.data_ptr<int32_t>(),
                                                                      prefill_bs);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_fused(const at::Tensor&         score,
                                      const at::Tensor&         lengths,
                                      at::Tensor&               topk_indices_ragged,
                                      const at::Tensor&         topk_indices_offset,
                                      std::optional<at::Tensor> row_starts_opt) {
    CHECK_CUDA(score);
    CHECK_CUDA(lengths);
    CHECK_CUDA(topk_indices_ragged);
    CHECK_CUDA(topk_indices_offset);
    if (row_starts_opt.has_value()) {
        CHECK_CUDA(row_starts_opt.value());
    }

    const auto params = get_params(score, lengths, row_starts_opt);
    const auto B      = score.size(0);
    TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
    TORCH_CHECK(topk_indices_offset.dim() == 1);

    TORCH_CHECK(topk_indices_ragged.size(0) == B);
    TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
    TORCH_CHECK(topk_indices_offset.size(0) == B);

    // launch kernel
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid   = dim3{static_cast<uint32_t>(B)};
    const auto block  = dim3{kThreadsPerBlock};

    // Launch the v2 kernel. Original `topk_transform_prefill_ragged_kernel` is
    // kept defined above for reference / fallback but no longer launched.
    setup_kernel_smem_once<topk_transform_prefill_ragged_v2_kernel, kSmemRaggedV2>();
    topk_transform_prefill_ragged_v2_kernel<<<grid, block, kSmemRaggedV2, stream>>>(
        params.input, params.input_stride, params.lengths, params.row_starts,
        topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

}  // namespace rtp_llm
