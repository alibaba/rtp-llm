#include "rtp_llm/cpp/kernels/weight_logits.h"
// #include "rtp_llm/cpp/kernels/logits_util.h"
#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
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
__device__ T NegativeInfinity2() {
    return -INFINITY;
}

template<>
__device__ __half NegativeInfinity2<__half>() {
    return -CUDART_INF_FP16;
}

template<>
__device__ __nv_bfloat16 NegativeInfinity2<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

template<typename T>
__device__ T addWeight(T logits, float weight) {
    return logits + weight;
}

template<>
__device__ __half addWeight<__half>(__half logits, float weight) {
    __half weight_half = __float2half(weight);
    return logits + weight_half;
}

template<>
__device__ __nv_bfloat16 addWeight<__nv_bfloat16>(__nv_bfloat16 logits, float weight) {
    __nv_bfloat16 weight_bf16 = __float2bfloat16(weight);
    return logits + weight_bf16;
}

template<typename T>
__device__ float toFloat(T v) {
    return v;
}

template<>
__device__ float toFloat<__half>(__half v) {
    return __half2float(v);
}

template<>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename T>
__device__ T getWeight(float v) {
    return v;
}

template<>
__device__ __half getWeight<__half>(float v) {
    return __float2half(v);
}

template<>
__device__ __nv_bfloat16 getWeight<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename T>
__global__ void extract_valid_scores(const int batch_size,
                                     const int vocab_size,
                                     const int weight_size,
                                     T*        logits_batch,
                                     const int* __restrict__ batch_idx,
                                     const int* __restrict__ vocab_idx,
                                     T* valid_score) {
    int score_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int b_idx      = 0;
    int total_size = batch_size * 2;
    for (int i = 0; i < total_size; i += 2) {
        if (score_idx < batch_idx[i]) {
            b_idx = batch_idx[i + 1];
            break;
        }
    }

    if (score_idx < weight_size) {
        int v_idx = vocab_idx[score_idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx         = b_idx * vocab_size + v_idx;
            valid_score[score_idx] = logits_batch[global_idx];
        }
    }
}

template<typename T>
__global__ void fill_logits_with_neg_inf(const int batch_size, const int vocab_size, T* logits_batch) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size && vocab_idx < vocab_size) {
        int global_idx           = batch_idx * vocab_size + vocab_idx;
        logits_batch[global_idx] = NegativeInfinity2<T>();
    }
}

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void weight_logits(const int batch_size,
                              const int vocab_size,
                              const int weight_size,
                              T*        logits_batch,
                              const int* __restrict__ batch_idx,
                              const int* __restrict__ vocab_idx,
                              float* __restrict__ weight_batch,
                              T* valid_score) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int b_idx      = 0;
    int total_size = batch_size * 2;
    for (int i = 0; i < total_size; i += 2) {
        if (weight_idx < batch_idx[i]) {
            b_idx = batch_idx[i + 1];
            break;
        }
    }
    if (weight_idx < weight_size) {
        int v_idx = vocab_idx[weight_idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx           = b_idx * vocab_size + v_idx;
            logits_batch[global_idx] = addWeight<T>(valid_score[weight_idx], weight_batch[weight_idx]);
        }
    }
}

template<typename T>
void invokeWeightLogits(T* logits_batch,
                        const int* __restrict__ batch_idx,
                        const int* __restrict__ vocab_idx,
                        float* __restrict__ weight_batch,
                        const int    batch_size,
                        const int    vocab_size,
                        const int    weight_size,
                        cudaStream_t stream) {
    dim3 block, grid;

    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.y  = 1;
    grid.z  = 1;

    // first store valid scores
    T* valid_scores;
    cudaMalloc(&valid_scores, weight_size * sizeof(T));
    check_cuda_error();
    grid.x = (weight_size + block.x - 1) / block.x;
    extract_valid_scores<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, weight_size, logits_batch, batch_idx, vocab_idx, valid_scores);

    // fill logits with -INF
    grid.y = batch_size;
    grid.x = (vocab_size + block.x - 1) / block.x;
    fill_logits_with_neg_inf<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch);

    // add weight to valid scores
    grid.y = 1;
    grid.x = (weight_size + block.x - 1) / block.x;

    weight_logits<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, weight_size, logits_batch, batch_idx, vocab_idx, weight_batch, valid_scores);
    cudaFree(valid_scores);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeWeightLogits<float>(float* logits_batch,
                                        const int* __restrict__ batch_idx,
                                        const int* __restrict__ vocab_idx,
                                        float* __restrict__ weight_batch,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    weight_size,
                                        cudaStream_t stream);
template void invokeWeightLogits<half>(half* logits_batch,
                                       const int* __restrict__ batch_idx,
                                       const int* __restrict__ vocab_idx,
                                       float* __restrict__ weight_batch,
                                       const int    batch_size,
                                       const int    vocab_size,
                                       const int    weight_size,
                                       cudaStream_t stream);
template void invokeWeightLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                const int* __restrict__ batch_idx,
                                                const int* __restrict__ vocab_idx,
                                                float* __restrict__ weight_batch,
                                                const int    batch_size,
                                                const int    vocab_size,
                                                const int    weight_size,
                                                cudaStream_t stream);

}  // namespace rtp_llm