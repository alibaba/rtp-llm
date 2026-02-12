#pragma once

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
__device__ T NegativeInfinity() {
    return -INFINITY;
}

template<>
__device__ inline __half NegativeInfinity<__half>() {
    return -CUDART_INF_FP16;
}

template<>
__device__ inline __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

template<typename T>
__device__ T addWeight(T logits, float weight) {
    return logits + weight;
}

template<>
__device__ inline __half addWeight<__half>(__half logits, float weight) {
    __half weight_half = __float2half(weight);
    return logits + weight_half;
}

template<>
__device__ inline __nv_bfloat16 addWeight<__nv_bfloat16>(__nv_bfloat16 logits, float weight) {
    __nv_bfloat16 weight_bf16 = __float2bfloat16(weight);
    return logits + weight_bf16;
}

template<typename T>
__device__ inline float to_float(T val) {
    return val;
}

template<>
__device__ inline float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<>
__device__ inline float to_float(__half val) {
    return __half2float(val);
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
        logits_batch[global_idx] = NegativeInfinity<T>();
    }
}

template<typename T>
__global__ void extract_valid_scores_to_weights(const int batch_size,
                                                const int vocab_size,
                                                const int weight_size,
                                                T*        logits_batch,
                                                const int* __restrict__ batch_idx,
                                                const int* __restrict__ vocab_idx,
                                                float* __restrict__ vocab_weights) {
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
            int global_idx = b_idx * vocab_size + v_idx;
            vocab_weights[score_idx] += to_float<T>(logits_batch[global_idx]);
        }
    }
}

}  // namespace rtp_llm