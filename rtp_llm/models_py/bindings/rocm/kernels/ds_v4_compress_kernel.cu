/*
 * DeepSeek-V4 FlashCompress4/128 HIP kernels for AMD ROCm
 * Adapted from sglang jit_kernel/csrc/deepseek_v4/c4.cuh and c128.cuh
 *
 * Key algorithm:
 *   C4: 4:1 KV compression via online softmax over 8-token ring buffer
 *   C128: 128:1 KV compression via online softmax over 128-token ring buffer
 *
 * Both kernels use 32-thread sub-warp reduction within AMD's 64-thread wavefront.
 */

#include "rtp_llm/models_py/bindings/rocm/kernels/ds_v4_compress_kernel.h"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"

namespace rtp_llm {

// ============================================================================
// Common constants
// ============================================================================

static constexpr int kC4TileElems = 4;
static constexpr int kC4BlockSize = 128;
static constexpr int kC4WarpThreads = 32;

static constexpr int kC128TileElems = 4;
static constexpr int kC128BlockSize = 128;
static constexpr int kC128WarpThreads = 32;

static inline __device__ float warpReduceMax32(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor(val, mask));
    }
    return val;
}

static inline __device__ float warpReduceSum32(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor(val, mask);
    }
    return val;
}

static inline uint32_t div_ceil_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// ============================================================================
// C4 Decode kernel
// ============================================================================

template <int64_t kHeadDim>
__global__ __launch_bounds__(kC4BlockSize, 4)
void flash_c4_decode_kernel(
    float* __restrict__ kv_score_buffer,   // [num_indices, 8, kHeadDim * 4]
    const float* __restrict__ kv_score_input,  // [batch_size, kHeadDim * 4]
    float* __restrict__ kv_compressed_output,  // [batch_size, kHeadDim]
    const float* __restrict__ score_bias,      // [8, kHeadDim] (ape)
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seq_lens,
    uint32_t batch_size) {

    constexpr int64_t kTileDim = kC4TileElems * kC4WarpThreads;  // 128
    constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
    constexpr int64_t kElementSize = kHeadDim * 4;

    // For head_dim < kTileDim, kNumSplit==0; the entire head_dim fits in one warp.
    constexpr uint32_t kEffectiveSplit = kNumSplit > 0 ? kNumSplit : 1;

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_wid = global_tid / kC4WarpThreads;
    const uint32_t global_bid = global_wid / kEffectiveSplit;
    const uint32_t global_sid = global_wid % kEffectiveSplit;

    if (global_bid >= batch_size) return;

    const int32_t index = indices[global_bid];
    const int32_t seq_len = seq_lens[global_bid];
    const int64_t split_offset = global_sid * kTileDim;
    const int64_t base = threadIdx.x * kC4TileElems;

    float* kv_buf_base = kv_score_buffer + index * (kElementSize * 8) + split_offset;
    const float* kv_src = kv_score_input + global_bid * kElementSize + split_offset;
    float* kv_out = kv_compressed_output + global_bid * kHeadDim + split_offset;
    const float* score_bias_split = score_bias + split_offset;

    // ---- Write: c4_write ----
    const int32_t write_pos = (seq_len + 7) % 8;
    float* write_dst = kv_buf_base + write_pos * kElementSize;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < kC4TileElems; ++j) {
            write_dst[kHeadDim * i + base + j] = kv_src[kHeadDim * i + base + j];
        }
    }

    // ---- Forward: c4_forward (only when seq_len % 4 == 0) ----
    if (seq_len % 4 == 0) {
        constexpr int64_t element_size = kHeadDim * 4;
        const int64_t score_offset = kHeadDim * 2;

        float kv_data[8][kC4TileElems];
        float score_data[8][kC4TileElems];
        float bias_data[kC4TileElems];

        // Load bias
        for (int j = 0; j < kC4TileElems; ++j) {
            bias_data[j] = score_bias_split[base + j];
        }

        // Load KV + score from ring buffer
        for (int i = 0; i < 8; ++i) {
            const bool is_overlap = i < 4;
            const int32_t k = (seq_len + i) % 8;
            const float* src = kv_buf_base + k * element_size + (is_overlap ? 0 : kHeadDim);
            for (int j = 0; j < kC4TileElems; ++j) {
                kv_data[i][j] = src[base + j];
                score_data[i][j] = src[base + j + score_offset];
            }
        }

        // Online softmax + weighted sum
        for (int j = 0; j < kC4TileElems; ++j) {
            float score_fp32[8];
            for (int i = 0; i < 8; ++i) {
                score_fp32[i] = score_data[i][j] + bias_data[j];
            }

            float max_val = score_fp32[0];
            for (int i = 1; i < 8; ++i) max_val = fmaxf(max_val, score_fp32[i]);

            float sum_product = 0.0f, sum_exp = 0.0f;
            for (int i = 0; i < 8; ++i) {
                float e = expf(score_fp32[i] - max_val);
                sum_product += kv_data[i][j] * e;
                sum_exp += e;
            }

            kv_out[base + j] = sum_product / sum_exp;
        }
    }
}

// ============================================================================
// C4 Prefill kernel
// ============================================================================

template <int64_t kHeadDim, bool kWrite>
__global__ __launch_bounds__(kC4BlockSize, 4)
void flash_c4_prefill_kernel(
    float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    float* __restrict__ kv_compressed_output,
    const float* __restrict__ score_bias,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ plan,  // [num_plans, 4]: ragged_id, batch_id, position, window_len
    uint32_t num_plans) {

    constexpr int64_t kTileDim = kC4TileElems * kC4WarpThreads;
    constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
    constexpr int64_t kElementSize = kHeadDim * 4;
    constexpr uint32_t kEffectiveSplit = kNumSplit > 0 ? kNumSplit : 1;

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_wid = global_tid / kC4WarpThreads;
    const uint32_t global_pid = global_wid / kEffectiveSplit;
    const uint32_t global_sid = global_wid % kEffectiveSplit;

    if (global_pid >= num_plans) return;

    const uint32_t ragged_id = plan[global_pid * 4 + 0];
    if (ragged_id == 0xFFFFFFFF) return;

    const int64_t split_offset = global_sid * kTileDim;
    const int64_t base = threadIdx.x * kC4TileElems;

    const float* kv_src = kv_score_input + ragged_id * kElementSize + split_offset;
    float* kv_out = kv_compressed_output + ragged_id * kHeadDim + split_offset;
    const float* score_bias_split = score_bias + split_offset;

    const uint32_t batch_id = plan[global_pid * 4 + 1];
    const uint32_t position = plan[global_pid * 4 + 2];
    const int32_t index = indices[batch_id];
    const int32_t seq_len = position + 1;

    float* kv_buf_base = kv_score_buffer + index * (kElementSize * 8) + split_offset;

    if constexpr (kWrite) {
        const int32_t write_pos = position % 8;
        float* write_dst = kv_buf_base + write_pos * kElementSize;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < kC4TileElems; ++j) {
                write_dst[kHeadDim * i + base + j] = kv_src[kHeadDim * i + base + j];
            }
        }
    } else {
        constexpr int64_t element_size = kHeadDim * 4;
        const int64_t score_offset = kHeadDim * 2;

        float kv_data[8][kC4TileElems];
        float score_data[8][kC4TileElems];
        float bias_data[kC4TileElems];

        for (int j = 0; j < kC4TileElems; ++j) {
            bias_data[j] = score_bias_split[base + j];
        }

        for (int i = 0; i < 8; ++i) {
            const int32_t k = (seq_len + i) % 8;
            const float* src = kv_buf_base + k * element_size;
            for (int j = 0; j < kC4TileElems; ++j) {
                kv_data[i][j] = src[base + j];
                score_data[i][j] = src[base + j + score_offset];
            }
        }

        for (int j = 0; j < kC4TileElems; ++j) {
            float score_fp32[8];
            for (int i = 0; i < 8; ++i) {
                score_fp32[i] = score_data[i][j] + bias_data[j];
            }

            float max_val = score_fp32[0];
            for (int i = 1; i < 8; ++i) max_val = fmaxf(max_val, score_fp32[i]);

            float sum_product = 0.0f, sum_exp = 0.0f;
            for (int i = 0; i < 8; ++i) {
                float e = expf(score_fp32[i] - max_val);
                sum_product += kv_data[i][j] * e;
                sum_exp += e;
            }

            kv_out[base + j] = sum_product / sum_exp;
        }
    }
}

// ============================================================================
// C128 Decode kernel
// ============================================================================

template <int64_t kHeadDim>
__global__ __launch_bounds__(kC128BlockSize, 2)
void flash_c128_decode_kernel(
    float* __restrict__ kv_score_buffer,   // [num_indices, 128, kHeadDim * 2]
    const float* __restrict__ kv_score_input,  // [batch_size, kHeadDim * 2]
    float* __restrict__ kv_compressed_output,  // [batch_size, kHeadDim]
    const float* __restrict__ score_bias,      // [128, kHeadDim]
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seq_lens,
    uint32_t batch_size) {

    constexpr int64_t kTileDim = kC128TileElems * kC128WarpThreads;
    constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
    constexpr int64_t kElementSize = kHeadDim * 2;
    constexpr uint32_t kEffectiveSplit = kNumSplit > 0 ? kNumSplit : 1;

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_wid = global_tid / kC128WarpThreads;
    const uint32_t global_bid = global_wid / kEffectiveSplit;
    const uint32_t global_sid = global_wid % kEffectiveSplit;

    if (global_bid >= batch_size) return;

    const int32_t index = indices[global_bid];
    const int32_t seq_len = seq_lens[global_bid];
    const int64_t split_offset = global_sid * kTileDim;
    const int64_t base = threadIdx.x * kC128TileElems;

    float* kv_buf_base = kv_score_buffer + index * (kElementSize * 128) + split_offset;
    const float* kv_src = kv_score_input + global_bid * kElementSize + split_offset;
    float* kv_out = kv_compressed_output + global_bid * kHeadDim + split_offset;
    const float* score_bias_split = score_bias + split_offset;

    // ---- Write ----
    const int32_t write_pos = (seq_len + 127) % 128;
    float* write_dst = kv_buf_base + write_pos * kElementSize;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < kC128TileElems; ++j) {
            write_dst[kHeadDim * i + base + j] = kv_src[kHeadDim * i + base + j];
        }
    }

    // ---- Forward (only when seq_len % 4 == 0) ----
    if (seq_len % 4 == 0) {
        constexpr int64_t element_size = kHeadDim * 2;
        constexpr int kC128NumWarps = kC128BlockSize / kC128WarpThreads;
        __shared__ float warp_max[kC128NumWarps];
        __shared__ float warp_exp_sum[kC128NumWarps];
        __shared__ float warp_product[kC128NumWarps];
        float local_max = -1e9f;
        for (int i = 0; i < 128; ++i) {
            const int32_t k = (seq_len + i) % 128;
            const float* src = kv_buf_base + k * element_size;
            float s = src[base] + score_bias_split[base];
            for (int j = 1; j < kC128TileElems; ++j) {
                s = fmaxf(s, src[base + j] + score_bias_split[base + j]);
            }
            local_max = fmaxf(local_max, s);
        }
        local_max = warpReduceMax32(local_max);

        // CTA-level max reduction via warp leaders
        warp_max[global_wid % (kC128BlockSize / kC128WarpThreads)] = local_max;
        __syncthreads();

        float global_max = -1e9f;
        if (threadIdx.x < kC128NumWarps) {
            for (int w = 0; w < kC128NumWarps; ++w) {
                global_max = fmaxf(global_max, warp_max[w]);
            }
        }
        __syncthreads();
        // Broadcast: every thread reads from warp 0's thread
        global_max = __shfl(global_max, 0);

        // Phase 2: per-warp local sums
        float local_exp_sum = 0.0f;
        float local_product = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const int32_t k = (seq_len + i) % 128;
            const float* src = kv_buf_base + k * element_size;
            for (int j = 0; j < kC128TileElems; ++j) {
                float score = src[base + j] + score_bias_split[base + j] - global_max;
                float e = expf(score);
                local_exp_sum += e;
                local_product += src[base + j] * e;
            }
        }

        local_exp_sum = warpReduceSum32(local_exp_sum);
        local_product = warpReduceSum32(local_product);

        // CTA-level sum reduction
        int warp_idx = global_wid % kC128NumWarps;
        warp_exp_sum[warp_idx] = local_exp_sum;
        warp_product[warp_idx] = local_product;
        __syncthreads();

        float total_exp = 0.0f, total_prod = 0.0f;
        if (threadIdx.x < kC128NumWarps) {
            for (int w = 0; w < kC128NumWarps; ++w) {
                total_exp += warp_exp_sum[w];
                total_prod += warp_product[w];
            }
        }
        // Broadcast
        total_exp = __shfl(total_exp, 0);
        total_prod = __shfl(total_prod, 0);

        for (int j = 0; j < kC128TileElems; ++j) {
            kv_out[base + j] = total_prod / total_exp;
        }
    }
}

// ============================================================================
// C128 Prefill kernel (reuses C4 prefill structure with 128-tile params)
// ============================================================================

template <int64_t kHeadDim, bool kWrite>
__global__ __launch_bounds__(kC128BlockSize, 2)
void flash_c128_prefill_kernel(
    float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    float* __restrict__ kv_compressed_output,
    const float* __restrict__ score_bias,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ plan,
    uint32_t num_plans) {

    constexpr int64_t kTileDim = kC128TileElems * kC128WarpThreads;
    constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
    constexpr int64_t kElementSize = kHeadDim * 2;
    constexpr uint32_t kEffectiveSplit = kNumSplit > 0 ? kNumSplit : 1;

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_wid = global_tid / kC128WarpThreads;
    const uint32_t global_pid = global_wid / kEffectiveSplit;
    const uint32_t global_sid = global_wid % kEffectiveSplit;

    if (global_pid >= num_plans) return;

    const uint32_t ragged_id = plan[global_pid * 4 + 0];
    if (ragged_id == 0xFFFFFFFF) return;

    const int64_t split_offset = global_sid * kTileDim;
    const int64_t base = threadIdx.x * kC128TileElems;

    const float* kv_src = kv_score_input + ragged_id * kElementSize + split_offset;
    float* kv_out = kv_compressed_output + ragged_id * kHeadDim + split_offset;
    const float* score_bias_split = score_bias + split_offset;

    const uint32_t batch_id = plan[global_pid * 4 + 1];
    const uint32_t position = plan[global_pid * 4 + 2];
    const int32_t index = indices[batch_id];
    const int32_t seq_len = position + 1;

    float* kv_buf_base = kv_score_buffer + index * (kElementSize * 128) + split_offset;

    if constexpr (kWrite) {
        const int32_t write_pos = position % 128;
        float* write_dst = kv_buf_base + write_pos * kElementSize;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < kC128TileElems; ++j) {
                write_dst[kHeadDim * i + base + j] = kv_src[kHeadDim * i + base + j];
            }
        }
    } else {
        constexpr int kC128NumWarps = kC128BlockSize / kC128WarpThreads;
        __shared__ float warp_max[kC128NumWarps];
        __shared__ float warp_exp_sum[kC128NumWarps];
        __shared__ float warp_product[kC128NumWarps];

        constexpr int64_t element_size = kHeadDim * 2;

        float local_max = -1e9f;
        for (int i = 0; i < 128; ++i) {
            const int32_t k = (seq_len + i) % 128;
            const float* src = kv_buf_base + k * element_size;
            float s = src[base] + score_bias_split[base];
            for (int j = 1; j < kC128TileElems; ++j) {
                s = fmaxf(s, src[base + j] + score_bias_split[base + j]);
            }
            local_max = fmaxf(local_max, s);
        }
        local_max = warpReduceMax32(local_max);

        int warp_idx = global_wid % kC128NumWarps;
        warp_max[warp_idx] = local_max;
        __syncthreads();

        float global_max = -1e9f;
        if (threadIdx.x < kC128NumWarps) {
            for (int w = 0; w < kC128NumWarps; ++w) {
                global_max = fmaxf(global_max, warp_max[w]);
            }
        }
        global_max = __shfl(global_max, 0);

        float local_exp_sum = 0.0f, local_product = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const int32_t k = (seq_len + i) % 128;
            const float* src = kv_buf_base + k * element_size;
            for (int j = 0; j < kC128TileElems; ++j) {
                float score = src[base + j] + score_bias_split[base + j] - global_max;
                float e = expf(score);
                local_exp_sum += e;
                local_product += src[base + j] * e;
            }
        }

        local_exp_sum = warpReduceSum32(local_exp_sum);
        local_product = warpReduceSum32(local_product);

        warp_exp_sum[warp_idx] = local_exp_sum;
        warp_product[warp_idx] = local_product;
        __syncthreads();

        float total_exp = 0.0f, total_prod = 0.0f;
        if (threadIdx.x < kC128NumWarps) {
            for (int w = 0; w < kC128NumWarps; ++w) {
                total_exp += warp_exp_sum[w];
                total_prod += warp_product[w];
            }
        }
        total_exp = __shfl(total_exp, 0);
        total_prod = __shfl(total_prod, 0);

        for (int j = 0; j < kC128TileElems; ++j) {
            kv_out[base + j] = total_prod / total_exp;
        }
    }
}

// ============================================================================
// Host-side launch wrappers (BF16 variant)
// ============================================================================

void invokeFlashCompress4Decode(
    void* kv_score_buffer,
    const void* kv_score_input,
    void* kv_compressed_output,
    const void* score_bias,
    const int32_t* indices,
    const int32_t* seq_lens,
    const int32_t* extra,
    uint32_t batch_size,
    int64_t head_dim,
    hipStream_t stream) {

    (void)extra;  // ring buffer mode only for now
    const uint32_t kNumSplit = head_dim / (kC4TileElems * kC4WarpThreads);
    const uint32_t kWarpsPerBlock = kC4BlockSize / kC4WarpThreads;
    const uint32_t num_blocks = (kNumSplit > 0) ? div_ceil_u32(batch_size * kNumSplit, kWarpsPerBlock) : div_ceil_u32(batch_size, kWarpsPerBlock);

#define LAUNCH_C4_DECODE_KERNEL(HDIM) \
    hipLaunchKernelGGL( \
        (flash_c4_decode_kernel<HDIM>), \
        dim3(num_blocks), dim3(kC4BlockSize), 0, stream, \
        static_cast<float*>(kv_score_buffer), \
        static_cast<const float*>(kv_score_input), \
        static_cast<float*>(kv_compressed_output), \
        static_cast<const float*>(score_bias), \
        indices, seq_lens, batch_size);

    switch (head_dim) {
        case 32:  LAUNCH_C4_DECODE_KERNEL(32) break;
        case 64:  LAUNCH_C4_DECODE_KERNEL(64) break;
        case 128: LAUNCH_C4_DECODE_KERNEL(128) break;
        case 256: LAUNCH_C4_DECODE_KERNEL(256) break;
        default: break;
    }
#undef LAUNCH_C4_DECODE_KERNEL
}

void invokeFlashCompress4Prefill(
    void* kv_score_buffer,
    const void* kv_score_input,
    void* kv_compressed_output,
    const void* score_bias,
    const int32_t* indices,
    const int32_t* compress_plan,
    const int32_t* write_plan,
    const int32_t* extra,
    uint32_t num_compress,
    uint32_t num_write,
    int64_t head_dim,
    hipStream_t stream) {

    (void)extra;
    const uint32_t kNumSplit = head_dim / (kC4TileElems * kC4WarpThreads);
    const uint32_t kWarpsPerBlock = kC4BlockSize / kC4WarpThreads;

#define LAUNCH_C4_PREFILL_KERNEL(HDIM, WRITE, PLAN_PTR, NUM) \
    { \
        const uint32_t num_wb = (kNumSplit > 0) ? div_ceil_u32((NUM) * kNumSplit, kWarpsPerBlock) : div_ceil_u32(NUM, kWarpsPerBlock); \
        hipLaunchKernelGGL( \
            (flash_c4_prefill_kernel<HDIM, WRITE>), \
            dim3(num_wb), dim3(kC4BlockSize), 0, stream, \
            static_cast<float*>(kv_score_buffer), \
            static_cast<const float*>(kv_score_input), \
            static_cast<float*>(kv_compressed_output), \
            static_cast<const float*>(score_bias), \
            indices, (const uint32_t*)(PLAN_PTR), (NUM)); \
    }

    if (num_compress > 0) {
        switch (head_dim) {
            case 32:  LAUNCH_C4_PREFILL_KERNEL(32, false, compress_plan, num_compress) break;
            case 64:  LAUNCH_C4_PREFILL_KERNEL(64, false, compress_plan, num_compress) break;
            case 128: LAUNCH_C4_PREFILL_KERNEL(128, false, compress_plan, num_compress) break;
            case 256: LAUNCH_C4_PREFILL_KERNEL(256, false, compress_plan, num_compress) break;
            default: break;
        }
    }

    if (num_write > 0) {
        switch (head_dim) {
            case 32:  LAUNCH_C4_PREFILL_KERNEL(32, true, write_plan, num_write) break;
            case 64:  LAUNCH_C4_PREFILL_KERNEL(64, true, write_plan, num_write) break;
            case 128: LAUNCH_C4_PREFILL_KERNEL(128, true, write_plan, num_write) break;
            case 256: LAUNCH_C4_PREFILL_KERNEL(256, true, write_plan, num_write) break;
            default: break;
        }
    }
#undef LAUNCH_C4_PREFILL_KERNEL
}

void invokeFlashCompress128Decode(
    void* kv_score_buffer,
    const void* kv_score_input,
    void* kv_compressed_output,
    const void* score_bias,
    const int32_t* indices,
    const int32_t* seq_lens,
    uint32_t batch_size,
    int64_t head_dim,
    hipStream_t stream) {

    const uint32_t kNumSplit = head_dim / (kC128TileElems * kC128WarpThreads);
    const uint32_t kWarpsPerBlock = kC128BlockSize / kC128WarpThreads;
    const uint32_t num_blocks = div_ceil_u32(batch_size * kNumSplit, kWarpsPerBlock);

    hipLaunchKernelGGL(
        (flash_c128_decode_kernel<128>),
        dim3(num_blocks), dim3(kC128BlockSize), 0, stream,
        static_cast<float*>(kv_score_buffer),
        static_cast<const float*>(kv_score_input),
        static_cast<float*>(kv_compressed_output),
        static_cast<const float*>(score_bias),
        indices, seq_lens, batch_size);
}

void invokeFlashCompress128Prefill(
    void* kv_score_buffer,
    const void* kv_score_input,
    void* kv_compressed_output,
    const void* score_bias,
    const int32_t* indices,
    const int32_t* compress_plan,
    const int32_t* write_plan,
    uint32_t num_compress,
    uint32_t num_write,
    int64_t head_dim,
    hipStream_t stream) {

    const uint32_t kNumSplit = head_dim / (kC128TileElems * kC128WarpThreads);
    const uint32_t kWarpsPerBlock = kC128BlockSize / kC128WarpThreads;

    if (num_compress > 0) {
        const uint32_t num_c_blocks = div_ceil_u32(num_compress * kNumSplit, kWarpsPerBlock);
        hipLaunchKernelGGL(
            (flash_c128_prefill_kernel<128, false>),
            dim3(num_c_blocks), dim3(kC128BlockSize), 0, stream,
            static_cast<float*>(kv_score_buffer),
            static_cast<const float*>(kv_score_input),
            static_cast<float*>(kv_compressed_output),
            static_cast<const float*>(score_bias),
            indices, (const uint32_t*)compress_plan, num_compress);
    }

    if (num_write > 0) {
        const uint32_t num_w_blocks = div_ceil_u32(num_write * kNumSplit, kWarpsPerBlock);
        hipLaunchKernelGGL(
            (flash_c128_prefill_kernel<128, true>),
            dim3(num_w_blocks), dim3(kC128BlockSize), 0, stream,
            static_cast<float*>(kv_score_buffer),
            static_cast<const float*>(kv_score_input),
            static_cast<float*>(kv_compressed_output),
            static_cast<const float*>(score_bias),
            indices, (const uint32_t*)write_plan, num_write);
    }
}

}  // namespace rtp_llm
