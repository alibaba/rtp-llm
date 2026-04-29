#include "ds_v4_topk512_kernel.h"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <hip/hip_fp16.h>
#include <cstdint>

namespace rtp_llm {

namespace {

constexpr uint32_t kTopK = 512;
constexpr uint32_t kTopKBlockSize = 512;
constexpr uint32_t kRadix = 256;
// Shared memory for input indices in stage 2 refinement:
// Each entry is int32_t, need enough space for threshold-bin elements.
// 8192 * sizeof(int32_t) = 32KB
constexpr uint32_t kSmemIdxCapacity = 8192;

// Convert float to uint8 for radix sort (sign-aware ordering)
__device__ uint8_t convertToUint8(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

// Convert float to uint32 for fine-grained radix sort
__device__ uint32_t convertToUint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Page table lookup: convert logical offset to physical index
__device__ int32_t pageToIndices(const int32_t* page_table, uint32_t i, uint32_t page_bits) {
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

// In-place cumulative sum on shared histogram (256 elements, 8 parallel scan steps)
// Result overwrites hist[0..255]. hist[256] should be 0 (sentinel).
__device__ void cumSumInPlace(uint32_t* hist) {
    const uint32_t tx = threadIdx.x;
    for (int32_t i = 0; i < 8; ++i) {
        const uint32_t stride = 1u << i;
        if (tx < kRadix) {
            uint32_t val = hist[tx];
            if (tx + stride < kRadix) {
                val += hist[tx + stride];
            }
            hist[tx] = val;
        }
        __syncthreads();
    }
}

// Naive transform for short sequences (seq_len <= kTopK)
__device__ void naiveTransform(
    const int32_t* page_table,
    int32_t* indices,
    int32_t* raw_indices,
    uint32_t length,
    uint32_t page_bits) {

    const uint32_t tx = threadIdx.x;
    if (tx < length) {
        indices[tx] = pageToIndices(page_table, tx, page_bits);
        if (raw_indices != nullptr) {
            raw_indices[tx] = static_cast<int32_t>(tx);
        }
    } else if (tx < kTopK) {
        indices[tx] = -1;
        if (raw_indices != nullptr) {
            raw_indices[tx] = -1;
        }
    }
}

// Radix-based top-K selection using multi-stage histogram refinement
// input:  [length] scores
// output: [kTopK] indices into input (sorted by score descending)
__device__ void radixTopK(
    const float* input,
    int32_t* output,
    uint32_t length) {

    const uint32_t tx = threadIdx.x;

    // Shared memory layout:
    //   hist[kRadix + 32]  - histogram + padding for bank conflicts
    //   counter            - atomic counter for output position
    //   threshold_bin_id   - the boundary bin found during scan
    //   num_input          - number of elements at threshold bin (for refinement)
    //   idx_buf[kSmemIdxCapacity * 2] - double-buffered input indices for stage 2
    alignas(128) __shared__ uint32_t s_hist[kRadix + 32];
    alignas(128) __shared__ uint32_t s_counter;
    alignas(128) __shared__ uint32_t s_threshold_bin_id;
    alignas(128) __shared__ uint32_t s_num_input;
    alignas(128) __shared__ int32_t s_last_remain;
    alignas(128) __shared__ int32_t s_idx_buf[kSmemIdxCapacity * 2];

    uint32_t remain_topk = kTopK;
    uint32_t buf_idx = 0;  // which buffer is currently being populated

    // ===== Stage 1: 8-bit coarse histogram =====
    if (tx < kRadix + 1) {
        s_hist[tx] = 0;
    }
    __syncthreads();

    // Accumulate histogram
    for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
        const uint32_t bin = convertToUint8(input[idx]);
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    // Cumulative sum
    cumSumInPlace(s_hist);

    // Find threshold bin: highest bin where cumsum exceeds remain_topk
    if (tx < kRadix && s_hist[tx] > remain_topk && s_hist[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
        s_num_input = 0;
        s_counter = 0;
    }
    __syncthreads();

    const uint32_t threshold_bin = s_threshold_bin_id;
    remain_topk -= s_hist[threshold_bin + 1];

    if (remain_topk == 0) {
        // All top-K elements are strictly above threshold bin
        for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
            const uint32_t bin = convertToUint8(input[idx]);
            if (bin > threshold_bin) {
                const uint32_t pos = atomicAdd(&s_counter, 1);
                output[pos] = static_cast<int32_t>(idx);
            }
        }
        __syncthreads();
        return;
    }

    // ===== Stage 2: refine with additional radix passes =====
    // Collect elements above threshold to output directly.
    // Collect elements AT threshold into shared memory for further refinement.
    s_counter = 0;
    s_num_input = 0;

    for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
        const float raw_val = input[idx];
        const uint32_t bin = convertToUint8(raw_val);
        if (bin > threshold_bin) {
            const uint32_t pos = atomicAdd(&s_counter, 1);
            output[pos] = static_cast<int32_t>(idx);
        } else if (bin == threshold_bin) {
            const uint32_t pos = atomicAdd(&s_num_input, 1);
            if (pos < kSmemIdxCapacity) {
                s_idx_buf[buf_idx * kSmemIdxCapacity + pos] = static_cast<int32_t>(idx);
                // Build histogram for next refinement pass (byte 0, i.e., bits 31-24)
                const uint32_t bin32 = (convertToUint32(raw_val) >> 24) & 0xFF;
                atomicAdd(&s_hist[bin32], 1);
            }
        }
    }
    __syncthreads();

    // 4 refinement rounds, each processing 8 more bits
    for (int round = 0; round < 4; ++round) {
        const uint32_t num_input = s_num_input < kSmemIdxCapacity ? s_num_input : kSmemIdxCapacity;
        const uint32_t offset = 24 - round * 8;

        // Cumsum on the refinement histogram
        cumSumInPlace(s_hist);

        // Find new threshold within this refinement bin
        if (tx < kRadix && s_hist[tx] > remain_topk && s_hist[tx + 1] <= remain_topk) {
            s_threshold_bin_id = tx;
            s_num_input = 0;
            s_last_remain = remain_topk - s_hist[tx + 1];
        }
        __syncthreads();

        const uint32_t thr = s_threshold_bin_id;
        remain_topk -= s_hist[thr + 1];

        if (remain_topk == 0) {
            // All remaining elements above threshold
            for (uint32_t i = tx; i < num_input; i += kTopKBlockSize) {
                const int32_t idx = s_idx_buf[buf_idx * kSmemIdxCapacity + i];
                const uint32_t bin = (convertToUint32(input[idx]) >> offset) & 0xFF;
                if (bin > thr) {
                    const uint32_t pos = atomicAdd(&s_counter, 1);
                    output[pos] = idx;
                }
            }
            __syncthreads();
            break;
        }

        // Reset histogram for next round
        __syncthreads();
        if (tx < kRadix + 1) {
            s_hist[tx] = 0;
        }
        __syncthreads();

        const uint32_t next_buf = buf_idx ^ 1;

        for (uint32_t i = tx; i < num_input; i += kTopKBlockSize) {
            const int32_t idx = s_idx_buf[buf_idx * kSmemIdxCapacity + i];
            const float raw_val = input[idx];
            const uint32_t bin = (convertToUint32(raw_val) >> offset) & 0xFF;

            if (bin > thr) {
                const uint32_t pos = atomicAdd(&s_counter, 1);
                output[pos] = idx;
            } else if (bin == thr) {
                if (round == 3) {
                    // Last round: take remaining s_last_remain elements at threshold
                    const int32_t pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        output[kTopK - pos] = idx;
                    }
                } else {
                    const uint32_t pos = atomicAdd(&s_num_input, 1);
                    if (pos < kSmemIdxCapacity) {
                        s_idx_buf[next_buf * kSmemIdxCapacity + pos] = idx;
                        // Build histogram for next round
                        const uint32_t sub_bin = (convertToUint32(raw_val) >> (offset - 8)) & 0xFF;
                        atomicAdd(&s_hist[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
        buf_idx = next_buf;
    }
}

}  // namespace

__global__ void topk512TransformKernel(
    const float* scores,
    const int32_t* seq_lens,
    const int32_t* page_table,
    int32_t* page_indices,
    int32_t* raw_indices,
    uint32_t batch_size,
    int64_t score_stride,
    int64_t page_table_stride,
    uint32_t page_bits) {

    const uint32_t work_id = blockIdx.x;
    if (work_id >= batch_size) return;

    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[work_id]);
    const auto score_ptr = scores + work_id * score_stride;
    const auto page_ptr = page_table + work_id * page_table_stride;
    const auto indices_ptr = page_indices + work_id * kTopK;
    const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * kTopK : nullptr;

    if (seq_len <= kTopK) {
        naiveTransform(page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
    } else {
        __shared__ int32_t s_topk_indices[kTopK];
        radixTopK(score_ptr, s_topk_indices, seq_len);

        const auto tx = threadIdx.x;
        if (tx < kTopK) {
            indices_ptr[tx] = pageToIndices(page_ptr, s_topk_indices[tx], page_bits);
            if (raw_indices_ptr != nullptr) {
                raw_indices_ptr[tx] = s_topk_indices[tx];
            }
        }
    }
}

void invokeTopK512(
    const float* scores,
    const int32_t* seq_lens,
    const int32_t* page_table,
    int32_t* page_indices,
    int32_t* raw_indices,
    uint32_t batch_size,
    int64_t score_stride,
    int64_t page_table_stride,
    uint32_t page_size,
    hipStream_t stream) {

    if (batch_size == 0) return;

    // page_size must be power of 2
    uint32_t page_bits = 0;
    while ((1u << page_bits) < page_size) page_bits++;

    const uint32_t block_size = kTopKBlockSize;

    hipLaunchKernelGGL(
        topk512TransformKernel,
        dim3(batch_size),
        dim3(block_size),
        0,
        stream,
        scores,
        seq_lens,
        page_table,
        page_indices,
        raw_indices,
        batch_size,
        score_stride,
        page_table_stride,
        page_bits);
}

}  // namespace rtp_llm
