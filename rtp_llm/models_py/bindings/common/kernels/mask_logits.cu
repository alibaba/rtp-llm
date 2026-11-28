#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"

#include <cfloat>

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half
#include <cuda_bf16.h>  // For __nv_bfloat16
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

#if USING_CUDA
#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

#if USING_ROCM
#ifndef HIP_INF_FP16
#define HIP_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef HIP_INF_BF16
#define HIP_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#define CUDART_INF_FP16 HIP_INF_FP16
#define CUDART_INF_BF16 HIP_INF_BF16
#endif

template<typename T>
__device__ T NegativeInfinity() {
    return -INFINITY;
}

template<>
__device__ __half NegativeInfinity<__half>() {
    return -CUDART_INF_FP16;
}

template<>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

// Match BaseLogitsProcessor::neg_inf after conversion to the logits dtype.
template<typename T>
__device__ T MaskedLogitValue() {
    return -FLT_MAX;
}

template<>
__device__ __half MaskedLogitValue<__half>() {
    return NegativeInfinity<__half>();
}

template<>
__device__ __nv_bfloat16 MaskedLogitValue<__nv_bfloat16>() {
    return NegativeInfinity<__nv_bfloat16>();
}

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void
mask_logits(const int batch_size, const int vocab_size, T* logits_batch, const uint8_t* __restrict__ mask_batch) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size && vocab_idx < vocab_size) {
        int global_idx = batch_idx * vocab_size + vocab_idx;
        if (mask_batch[global_idx]) {
            logits_batch[global_idx] = NegativeInfinity<T>();
        }
    }
}

#if USING_CUDA
template<typename T>
__global__ void packed_mask_logits(T* __restrict__ logits_batch,
                                   const int32_t* __restrict__ packed_allow_mask,
                                   const int32_t* __restrict__ row_indices,
                                   const int mask_rows,
                                   const int logits_rows,
                                   const int logits_row_stride,
                                   const int vocab_size,
                                   const int bitmask_row_stride,
                                   const int bitmask_words) {
    const int compact_row = blockIdx.y;
    const int vocab_idx   = threadIdx.x + blockIdx.x * blockDim.x;
    if (compact_row >= mask_rows || vocab_idx >= vocab_size) {
        return;
    }

    const int logits_row = row_indices == nullptr ? compact_row : row_indices[compact_row];
    if (logits_row < 0 || logits_row >= logits_rows) {
        return;
    }

    const int word_idx = vocab_idx / 32;
    bool      allowed  = false;
    if (word_idx < bitmask_words) {
        const int64_t mask_offset =
            static_cast<int64_t>(compact_row) * static_cast<int64_t>(bitmask_row_stride) + word_idx;
        const uint32_t word = static_cast<uint32_t>(packed_allow_mask[mask_offset]);
        allowed             = (word & (1u << (vocab_idx % 32))) != 0u;
    }
    if (!allowed) {
        const int64_t logits_offset =
            static_cast<int64_t>(logits_row) * static_cast<int64_t>(logits_row_stride) + vocab_idx;
        logits_batch[logits_offset] = MaskedLogitValue<T>();
    }
}
#endif

template<typename T>
void invokeMaskLogits(T* logits_batch,
                      const uint8_t* __restrict__ mask_batch,
                      const int    batch_size,
                      const int    vocab_size,
                      cudaStream_t stream) {
    dim3 block, grid;

    block.x = 64;
    block.y = 1;
    block.z = 1;
    grid.x  = (vocab_size + block.x - 1) / block.x;
    grid.y  = batch_size;
    grid.z  = 1;

    mask_logits<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch, mask_batch);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

#if USING_CUDA
template<typename T>
void invokePackedMaskLogits(T* logits_batch,
                            const int32_t* __restrict__ packed_allow_mask,
                            const int32_t* __restrict__ row_indices,
                            const int    mask_rows,
                            const int    logits_rows,
                            const int    logits_row_stride,
                            const int    vocab_size,
                            const int    bitmask_row_stride,
                            const int    bitmask_words,
                            cudaStream_t stream) {
    const dim3 block(256);
    const dim3 grid((vocab_size + block.x - 1) / block.x, mask_rows);
    packed_mask_logits<<<grid, block, 0, stream>>>(logits_batch,
                                                   packed_allow_mask,
                                                   row_indices,
                                                   mask_rows,
                                                   logits_rows,
                                                   logits_row_stride,
                                                   vocab_size,
                                                   bitmask_row_stride,
                                                   bitmask_words);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}
#endif

template void invokeMaskLogits<float>(float* logits_batch,
                                      const uint8_t* __restrict__ mask_batch,
                                      const int    batch_size,
                                      const int    vocab_size,
                                      cudaStream_t stream);
template void invokeMaskLogits<half>(half* logits_batch,
                                     const uint8_t* __restrict__ mask_batch,
                                     const int    batch_size,
                                     const int    vocab_size,
                                     cudaStream_t stream);
template void invokeMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                              const uint8_t* __restrict__ mask_batch,
                                              const int    batch_size,
                                              const int    vocab_size,
                                              cudaStream_t stream);

#if USING_CUDA
template void invokePackedMaskLogits<float>(float* logits_batch,
                                            const int32_t* __restrict__ packed_allow_mask,
                                            const int32_t* __restrict__ row_indices,
                                            const int    mask_rows,
                                            const int    logits_rows,
                                            const int    logits_row_stride,
                                            const int    vocab_size,
                                            const int    bitmask_row_stride,
                                            const int    bitmask_words,
                                            cudaStream_t stream);
template void invokePackedMaskLogits<half>(half* logits_batch,
                                           const int32_t* __restrict__ packed_allow_mask,
                                           const int32_t* __restrict__ row_indices,
                                           const int    mask_rows,
                                           const int    logits_rows,
                                           const int    logits_row_stride,
                                           const int    vocab_size,
                                           const int    bitmask_row_stride,
                                           const int    bitmask_words,
                                           cudaStream_t stream);
template void invokePackedMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                    const int32_t* __restrict__ packed_allow_mask,
                                                    const int32_t* __restrict__ row_indices,
                                                    const int    mask_rows,
                                                    const int    logits_rows,
                                                    const int    logits_row_stride,
                                                    const int    vocab_size,
                                                    const int    bitmask_row_stride,
                                                    const int    bitmask_words,
                                                    cudaStream_t stream);
template<typename T>
__device__ T ZeroLogitValue() {
    return 0.0f;
}

template<>
__device__ __half ZeroLogitValue<__half>() {
    return __float2half(0.0f);
}

template<>
__device__ __nv_bfloat16 ZeroLogitValue<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template<typename T>
__global__ void batch_grammar_mask_logits(T* __restrict__ logits_batch,
                                          const int32_t* __restrict__ packed_buf,
                                          const int bitmask_header_words,
                                          const int active_rows,
                                          const int logits_rows,
                                          const int logits_row_stride,
                                          const int vocab_size,
                                          const int bitmask_row_stride) {
    const int compact_row = static_cast<int>(blockIdx.y);
    if (compact_row >= active_rows) {
        return;
    }

    const int logits_row = __ldg(packed_buf + compact_row);
    if (logits_row < 0 || logits_row >= logits_rows) {
        return;
    }

    const int mode        = __ldg(packed_buf + active_rows + compact_row);
    const int vocab_limit = __ldg(packed_buf + 2 * active_rows + compact_row);
    const int block_base  = static_cast<int>(blockIdx.x) * 1024;
    T* __restrict__ logits_row_ptr =
        logits_batch + static_cast<int64_t>(logits_row) * static_cast<int64_t>(logits_row_stride);
    const T neg_val = MaskedLogitValue<T>();

    if (mode == -1) {
        if (block_base + 1024 <= vocab_limit) {
            return;
        }
#pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int col = block_base + iter * 256 + static_cast<int>(threadIdx.x);
            if (col >= vocab_limit && col < vocab_size) {
                logits_row_ptr[col] = neg_val;
            }
        }
        return;
    }

    if (mode >= 0) {
        const T zero_val = ZeroLogitValue<T>();
#pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int col = block_base + iter * 256 + static_cast<int>(threadIdx.x);
            if (col < vocab_size) {
                logits_row_ptr[col] = (col == mode) ? zero_val : neg_val;
            }
        }
        return;
    }

    const int mask_row = -2 - mode;
    const int32_t* __restrict__ mask_row_ptr =
        packed_buf + bitmask_header_words + static_cast<int64_t>(mask_row) * static_cast<int64_t>(bitmask_row_stride);
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int lane    = static_cast<int>(threadIdx.x) & 31;

#pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        const int word_base_col = block_base + iter * 256 + warp_id * 32;
        if (word_base_col >= vocab_size) {
            break;
        }
        const int word_idx = word_base_col >> 5;
        uint32_t  word     = 0u;
        if (word_base_col < vocab_limit && word_idx < bitmask_row_stride) {
            word = static_cast<uint32_t>(__ldg(mask_row_ptr + word_idx));
        }
        if (word == 0xFFFFFFFFu && word_base_col + 31 < vocab_limit && word_base_col + 31 < vocab_size) {
            continue;
        }
        const int col = word_base_col + lane;
        if (col < vocab_size) {
            const bool allowed = (col < vocab_limit) && ((word & (1u << lane)) != 0u);
            if (!allowed) {
                logits_row_ptr[col] = neg_val;
            }
        }
    }
}

template<typename T>
void invokeBatchGrammarMaskLogits(T* logits_batch,
                                  const int32_t* __restrict__ packed_buf,
                                  const int    bitmask_header_words,
                                  const int    active_rows,
                                  const int    logits_rows,
                                  const int    logits_row_stride,
                                  const int    vocab_size,
                                  const int    bitmask_row_stride,
                                  cudaStream_t stream) {
    if (active_rows <= 0 || vocab_size <= 0) {
        return;
    }
    const dim3 block(256, 1);
    const dim3 grid(static_cast<unsigned int>((vocab_size + 1023) / 1024), static_cast<unsigned int>(active_rows));
    batch_grammar_mask_logits<T><<<grid, block, 0, stream>>>(
        logits_batch, packed_buf, bitmask_header_words, active_rows, logits_rows, logits_row_stride, vocab_size, bitmask_row_stride);
    check_cuda_error();
}

template void invokeBatchGrammarMaskLogits<float>(float* logits_batch,
                                                  const int32_t* __restrict__ packed_buf,
                                                  const int    bitmask_header_words,
                                                  const int    active_rows,
                                                  const int    logits_rows,
                                                  const int    logits_row_stride,
                                                  const int    vocab_size,
                                                  const int    bitmask_row_stride,
                                                  cudaStream_t stream);
template void invokeBatchGrammarMaskLogits<half>(half* logits_batch,
                                                 const int32_t* __restrict__ packed_buf,
                                                 const int    bitmask_header_words,
                                                 const int    active_rows,
                                                 const int    logits_rows,
                                                 const int    logits_row_stride,
                                                 const int    vocab_size,
                                                 const int    bitmask_row_stride,
                                                 cudaStream_t stream);
template void invokeBatchGrammarMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                          const int32_t* __restrict__ packed_buf,
                                                          const int    bitmask_header_words,
                                                          const int    active_rows,
                                                          const int    logits_rows,
                                                          const int    logits_row_stride,
                                                          const int    vocab_size,
                                                          const int    bitmask_row_stride,
                                                          cudaStream_t stream);
#endif

}  // namespace rtp_llm
