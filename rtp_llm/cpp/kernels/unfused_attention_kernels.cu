/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/kernels/rotary_position_embedding.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#include <cstdlib>

namespace rtp_llm {

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4) {
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

__global__ void getSkipLength(int* skip_length, int* prefix_lengths, int batch_size) {
    int min_skip_length = prefix_lengths[0];
    for (int i = 1; i < batch_size; i++) {
        if (min_skip_length > prefix_lengths[i]) {
            min_skip_length = prefix_lengths[i];
        }
    }
    *skip_length = min_skip_length;
}

__global__ void float_to_half_kernel(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void float_to_half(const void* input, void* output, int size) {
    const float*  float_input       = reinterpret_cast<const float*>(input);
    half*         half_output       = reinterpret_cast<half*>(output);
    constexpr int THREADS_PER_BLOCK = 256;
    int           n_blocks          = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float_to_half_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(float_input, half_output, size);
    cudaDeviceSynchronize();
}

__global__ void
half_to_float_kernel(const __half* __restrict__ input, float* __restrict__ output, const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = __half2float(input[idx]);
    }
}

void half_to_float(const void* input, void* output, const int num_elements) {

    const half* half_input   = reinterpret_cast<const half*>(input);
    float*      float_output = reinterpret_cast<float*>(output);

    const int blockSize = 256;
    const int gridSize  = (num_elements + blockSize - 1) / blockSize;

    half_to_float_kernel<<<gridSize, blockSize>>>(half_input, float_output, num_elements);
    cudaDeviceSynchronize();
}

template<typename T, typename T_IN, int ITEMS_PER_THREAD>
__global__ void softmax_kernel(T*           attn_score,
                               const T_IN*  qk,
                               const T*     attn_mask,
                               const float* linear_bias_slopes,
                               const int    batch_size,
                               const int    head_num,
                               const int    q_length,
                               const int    k_length,
                               const float  qk_scale) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    const int bi = blockIdx.y;  // Batch index.
    const int hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float)linear_bias_slopes[hi] : 0.0f;

    // Loop along with Q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {

        float   data[ITEMS_PER_THREAD];
        int64_t qk_offset;
        float   local_max = -1e20f;

        // Loop along with K dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            int ki    = blockDim.x * i + threadIdx.x;  // Index of K dimension.
            qk_offset = ((bi * head_num + hi) * q_length + qi) * static_cast<int64_t>(k_length) + ki;

            float qk_val  = static_cast<float>(qk[qk_offset]);
            float qk_bias = 0.0f;

            if (linear_bias_slopes != nullptr) {
                // We don't handle the upper diagonal (ki > qi) separately, whose values
                // are negligible due to the negative infinity mask. And it matches with
                // the HF's implementation.
                qk_bias -= static_cast<float>(abs(linear_bias_slope * (ki - qi)));
            }

            int   mask_offset = (bi * q_length + qi) * k_length + ki;
            float mask_val    = static_cast<float>(ldg(&attn_mask[mask_offset]));
            qk_bias += (1.0f - mask_val) * -10000.0f;

            data[i]   = qk_scale * qk_val + qk_bias;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            qk_offset             = ((bi * head_num + hi) * q_length + qi) * k_length + blockDim.x * i + threadIdx.x;
            attn_score[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2(T*           attn_score,
                                  const T*     qk_buf,
                                  const T*     attn_mask,
                                  const float* linear_bias_slopes,
                                  const int    batch_size,
                                  const int    head_num,
                                  const int    q_length,
                                  const int    k_length,
                                  const T      qk_scale) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = reinterpret_cast<const T2*>(attn_mask);

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale_h2 = cuda_cast<T2>(qk_scale);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {
        T2      data[ITEMS_PER_THREAD];
        int64_t qk_offset;
        float   local_max = -1e20f;

        // Loop over k dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki          = blockDim.x * i + threadIdx.x;
            qk_offset       = ((bi * head_num + hi) * q_length + qi) * static_cast<int64_t>(k_length / 2) + ki;
            int mask_offset = (bi * q_length + qi) * (k_length / 2) + ki;

            // The value of QK^T matrix at (qi, ki).
            T2 qk = qk_buf_h2[qk_offset];
            // The bias value to the position (qi, ki) including both mask and positional bias.
            T2 qk_bias = ZERO;

            if (linear_bias_slopes != nullptr) {
                // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                // separately, whose values are negligible due to the negative infinity mask.
                T2 dist(2.0f * ki - qi, 2.0f * ki + 1 - qi);
                qk_bias = hadd2<T2>(qk_bias, -cuda_abs(hmul2<T2>(linear_bias_slope, dist)));
            }

            T2 mask_val = ldg(&attn_mask_h2[mask_offset]);
            qk_bias     = hadd2<T2>(qk_bias, hmul2<T2>(hsub2<T2>(ONE, mask_val), NEG_INFTY));

            data[i]   = hadd2<T2>(hmul2<T2>(qk, qk_scale_h2), qk_bias);
            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + blockDim.x * i + threadIdx.x;
            attn_score_h2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

template<typename T, int K_ITEMS_PER_THREAD, int Q_ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2_v2(T*           attn_score,
                                     const T*     qk_buf,
                                     const T*     attn_mask,
                                     const float* linear_bias_slopes,
                                     const int    batch_size,
                                     const int    head_num,
                                     const int    q_length,
                                     const int    k_length,
                                     const T      scalar) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    // QK^T matrix of shape (batch_size, head_num, q_length, k_length / 2)
    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = reinterpret_cast<const T2*>(attn_mask);

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale = cuda_cast<T2>(scalar);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    __shared__ float s_sum[Q_ITEMS_PER_THREAD], s_max[Q_ITEMS_PER_THREAD];

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x * Q_ITEMS_PER_THREAD) {
        T2 data[Q_ITEMS_PER_THREAD][K_ITEMS_PER_THREAD];

        int64_t qk_offset[Q_ITEMS_PER_THREAD];

        float local_max[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_max[j] = -1e20f;
        }

        // Loop over k dimension.
        const int Q_ITEMS = min((q_length - qi + gridDim.x - 1) / gridDim.x, Q_ITEMS_PER_THREAD);
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki = blockDim.x * i + threadIdx.x;

            int mask_offset[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] =
                    ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * static_cast<int64_t>(k_length / 2) + ki;
                mask_offset[j] = (bi * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
            }

            T2 mask_val[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = ldg(&attn_mask_h2[mask_offset[j]]);
            }

            T2 qk[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk[j] = qk_buf_h2[qk_offset[j]];
            }

            T2 pos_bias[Q_ITEMS_PER_THREAD];
            if (linear_bias_slopes != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                    // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                    // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                    // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                    // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                    // separately, whose values are negligible due to the negative infinity mask.
                    int qidx = qi + j * gridDim.x;
                    T2  dist(2.0f * ki - qidx, 2.0f * ki + 1 - qidx);
                    pos_bias[j] = -cuda_abs(hmul2<T2>(linear_bias_slope, dist));
                }
            }
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(ONE, mask_val[j]), NEG_INFTY);
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                T2 val = hadd2<T2>(hmul2<T2>(qk_scale, qk[j]), mask_val[j]);
                if (linear_bias_slopes != nullptr) {
                    val = hadd2<T2>(val, pos_bias[j]);
                }
                data[j][i]   = val;
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        } else {
            blockReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; ++j) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        } else {
            blockReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + blockDim.x * i
                               + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                attn_score_h2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

#define LAUNCH_MAKSED_SOFTMAX_(T_, ITEMS_PER_THREAD)                                                                   \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    block.x = (block.x + 31) / 32 * 32;                                                                                \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_h2_v2<T_, ITEMS_PER_THREAD, 4>                                                              \
                <<<grid, block, 0, stream>>>((T_*)param.attention_score,                                               \
                                             (const T_*)param.qk,                                                      \
                                             (const T_*)param.attention_mask,                                          \
                                             (const float*)param.linear_bias_slopes,                                   \
                                             param.batch_size,                                                         \
                                             param.num_heads,                                                          \
                                             param.q_length,                                                           \
                                             param.k_length,                                                           \
                                             (const T_)param.qk_scale);                                                \
        } else {                                                                                                       \
            softmax_kernel_h2<T_, ITEMS_PER_THREAD>                                                                    \
                <<<grid, block, 0, stream>>>((T_*)param.attention_score,                                               \
                                             (const T_*)param.qk,                                                      \
                                             (const T_*)param.attention_mask,                                          \
                                             (const float*)param.linear_bias_slopes,                                   \
                                             param.batch_size,                                                         \
                                             param.num_heads,                                                          \
                                             param.q_length,                                                           \
                                             param.k_length,                                                           \
                                             (const T_)param.qk_scale);                                                \
        }                                                                                                              \
    } else {                                                                                                           \
        softmax_kernel<T, T_IN, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(param.attention_score,                   \
                                                                              param.qk,                                \
                                                                              param.attention_mask,                    \
                                                                              param.linear_bias_slopes,                \
                                                                              param.batch_size,                        \
                                                                              param.num_heads,                         \
                                                                              param.q_length,                          \
                                                                              param.k_length,                          \
                                                                              param.qk_scale);                         \
    }

#define LAUNCH_MAKSED_SOFTMAX(ITEMS_PER_THREAD) LAUNCH_MAKSED_SOFTMAX_(half, ITEMS_PER_THREAD)

template<typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream) {
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);

    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    RTP_LLM_CHECK(param.k_length <= 32768);
    if (block.x > 16384 && block.x <= 32768) {
        LAUNCH_MAKSED_SOFTMAX(32)
    } else if (block.x > 8192) {
        LAUNCH_MAKSED_SOFTMAX(16)
    } else if (block.x > 4096) {
        LAUNCH_MAKSED_SOFTMAX(8)
    } else if (block.x > 2048) {
        LAUNCH_MAKSED_SOFTMAX(4)
    } else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX(2)
    } else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX(1)
    }
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, half>& param, cudaStream_t stream);

#ifdef ENABLE_BF16
template<>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, float>& param, cudaStream_t stream) {
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T    = __nv_bfloat16;
    using T_IN = float;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    RTP_LLM_CHECK(param.k_length <= 8192);
    if (block.x > 4096 && block.x <= 8192) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 8);
    } else if (block.x > 2048) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 4);
    } else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 2);
    } else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 1);
    }
}
template<>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, __nv_bfloat16>& param, cudaStream_t stream) {
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T    = __nv_bfloat16;
    using T_IN = __nv_bfloat16;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    RTP_LLM_CHECK(param.k_length <= 8192);
    if (block.x > 4096 && block.x <= 8192) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 8);
    } else if (block.x > 2048) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 4);
    } else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 2);
    } else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 1);
    }
}

#endif

#undef LAUNCH_MAKSED_SOFTMAX
#undef LAUNCH_MAKSED_SOFTMAX_

template<typename T>
__global__ void transpose(const T*     src,
                          T*           dst,
                          const int    batch_size,
                          const int    seq_len,
                          const int    head_num,
                          const int    size_per_head,
                          const float* scale,
                          int          int8_mode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    if (int8_mode == 2) {
        using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
        using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;

        const Float_Packed_T scale_val = cuda_cast<Float_Packed_T>(*scale);
        reinterpret_cast<Int8_Packed_T*>(dst)[target_id] =
            cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src[tid]) * scale_val);
    } else {
        dst[target_id] = src[tid];
    }
}

template<>
__global__ void transpose(const float* src,
                          float*       dst,
                          const int    batch_size,
                          const int    seq_len,
                          const int    head_num,
                          const int    size_per_head,
                          const float* scale,
                          int          int8_mode) {
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id   = blockIdx.x % seq_len;
    int head_id  = (blockIdx.x % (head_num * seq_len)) / seq_len;

    const int target_id = batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
                          + head_id * size_per_head + threadIdx.x;
    const int src_id = blockIdx.x * size_per_head + threadIdx.x;

    if (int8_mode == 2) {
        const float scale_val                     = *scale;
        reinterpret_cast<int8_t*>(dst)[target_id] = cuda_cast<int8_t>(src[src_id] * scale_val);
    } else {
        dst[target_id] = src[src_id];
    }
}

template<typename T>
void invokeTransposeQKV(T*           dst,
                        T*           src,
                        const int    batch_size,
                        const int    seq_len,
                        const int    head_num,
                        const int    size_per_head,
                        const float* scale,
                        const int    int8_mode,
                        cudaStream_t stream) {
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x            = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        RTP_LLM_CHECK(grid.x * seq_per_block == (size_t)batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2, scale, int8_mode);
            }

#ifdef ENABLE_BF16
            else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                transpose<__nv_bfloat162><<<grid, block, 0, stream>>>((__nv_bfloat162*)src,
                                                                      (__nv_bfloat162*)dst,
                                                                      batch_size,
                                                                      seq_len,
                                                                      head_num,
                                                                      size_per_head / 2,
                                                                      scale,
                                                                      int8_mode);
            }
#endif
        } else {
            block.x = seq_per_block * size_per_head;
            transpose<T>
                <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
        }
    } else {
        const int seq_per_block = 1;
        grid.x                  = batch_size * head_num * seq_len / seq_per_block;
        block.x                 = seq_per_block * size_per_head;
        transpose<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
    }
}

#define INSTANTIATETRANSPOSEQKV(T)                                                                                     \
    template void invokeTransposeQKV(T*           src,                                                                 \
                                     T*           dst,                                                                 \
                                     const int    batch_size,                                                          \
                                     const int    seq_len,                                                             \
                                     const int    head_num,                                                            \
                                     const int    size_per_head,                                                       \
                                     const float* scale,                                                               \
                                     const int    int8_mode,                                                           \
                                     cudaStream_t stream)
INSTANTIATETRANSPOSEQKV(float);
INSTANTIATETRANSPOSEQKV(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEQKV(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEQKV

template<typename T>
__global__ void transpose_remove_padding(const T*     src,
                                         T*           dst,
                                         const int    batch_size,
                                         const int    seq_len,
                                         const int    head_num,
                                         const int    size_per_head,
                                         const int*   mask_offset,
                                         const float* scale,
                                         const int    int8_mode) {
    // TODO: optimize this kernel?
    // do remove_sequence_length_padding
    const int bid = blockIdx.x;  // batch * seq_len or valid_word_num

    const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int src_seq_id   = (bid + mask_offset[bid]) % seq_len;

    const int dst_seq_id = bid;

    const int src_offset_base = src_batch_id * seq_len * head_num * size_per_head + src_seq_id * size_per_head;
    const int dst_offset_base = dst_seq_id * head_num * size_per_head;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    const Float_Packed_T scale_val =
        int8_mode == 2 ? cuda_cast<Float_Packed_T>(*scale) : cuda_cast<Float_Packed_T>(0.0f);

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x) {
        const int head_id   = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        const T   src_elem  = ldg(&src[src_offset_base + head_id * seq_len * size_per_head + hidden_id]);
        if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(dst)[dst_offset_base + idx] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src_elem) * scale_val);
        } else {
            dst[dst_offset_base + idx] = src_elem;
        }
    }
}

// clang-format off
template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream)
{
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2       = typename TypeConverter<T>::Type;  // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2   = false;
                block_size = std::min(block_size, 1024);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 1024);
    }

    if (is_half2) {
        transpose_remove_padding<T2><<<valid_word_num, block_size, 0, stream>>>(
            (T2*)src, (T2*)dst, batch_size, seq_len, head_num, size_per_head / 2, mask_offset, scale, int8_mode);
    }
    else {
        transpose_remove_padding<<<valid_word_num, block_size, 0, stream>>>(
            src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset, scale, int8_mode);
    }
}
// clang-format on

#define INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(T)                                                               \
    template void invokeTransposeAttentionOutRemovePadding(T*           src,                                           \
                                                           T*           dst,                                           \
                                                           const int    valid_word_num,                                \
                                                           const int    batch_size,                                    \
                                                           const int    seq_len,                                       \
                                                           const int    head_num,                                      \
                                                           const int    size_per_head,                                 \
                                                           const int*   mask_offset,                                   \
                                                           const float* scale,                                         \
                                                           const int    int8_mode,                                     \
                                                           cudaStream_t stream)
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(float);
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING

template<typename T>
struct Vec_t {
    static constexpr int size = 0;
};

template<>
struct Vec_t<float> {
    using Type                = float2;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};

template<>
struct Vec_t<half> {
    using Type                = uint32_t;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};

#ifdef ENABLE_BF16
template<>
struct Vec_t<__nv_bfloat16> {
    using Type                = __nv_bfloat162;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};
#endif

// Multiple calls of reinterpret_cast.
template<typename type_in, typename type_out>
inline __device__ type_out* reinterpret_ptr(void* ptr, size_t offset) {
    return reinterpret_cast<type_out*>(reinterpret_cast<type_in*>(ptr) + offset);
}

#include <cstdio>

// Helper function to convert to float (specialized for each type)
__device__ float convert_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ float convert_to_float(__half val) {
    return __half2float(val);
}

__device__ float convert_to_float(float val) {
    return val;
}

__device__ float convert_to_float(int val) {
    return float(val);
}

template<typename T>
__global__ void debug_kernel2(T* data, int start_row, int start_col, int m, int n, int row_len, int info_id) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("debug_kernel2 start: %d\n", info_id);
        for (int i = start_row; i < start_row + m; i++) {
            for (int j = start_col; j < start_col + n; j++) {
                int   index = i * row_len + j;
                float value = convert_to_float(data[index]);
                printf("%f ", value);
            }
            printf("\n");
        }
        printf("debug_kernel2 end: %d\n", info_id);
    }
}

template<typename T>
void invoke_debug_kernel2(
    T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream) {
    debug_kernel2<<<1, 1, 0, stream>>>(data, start_row, start_col, m, n, row_len, info_id);
}

#define INSTANTIATEDEBUGKERNEL2(T)                                                                                     \
    template void invoke_debug_kernel2(                                                                                \
        T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream)
INSTANTIATEDEBUGKERNEL2(float);
INSTANTIATEDEBUGKERNEL2(half);
INSTANTIATEDEBUGKERNEL2(int);
#ifdef ENABLE_BF16
INSTANTIATEDEBUGKERNEL2(__nv_bfloat16);
#endif
#undef INSTANTIATEDEBUGKERNEL2

// Bandwidth-bound kernel by reading cos/sin coefficients from global memory (pre-computed and saved as weights).

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_kernel(T*                            q_no_transpose_buf,
                                                   T*                            q_buf,
                                                   T*                            k_buf,
                                                   T*                            v_buf,
                                                   PrefixPromptBatchWeightsParam param,
                                                   T*                            QKV,
                                                   void*                         QuantizedQKV,
                                                   const int*                    position_ids,
                                                   const T* __restrict qkv_bias,
                                                   const int* padding_offset,
                                                   const int* cu_seqlens,
                                                   const int  batch_size,
                                                   const int  seq_len,
                                                   const int  head_num,
                                                   const int  head_num_kv,
                                                   const int  size_per_head,
                                                   RopeConfig rope_config,
                                                   const bool use_logn_attn,
                                                   bool       store_qkv,
                                                   bool       store_q_no_transpose,
                                                   bool       store_q,
                                                   bool       store_kv,
                                                   bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }
    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }
    const int pre_len   = cu_seqlens[batch_idx];
    const int input_len = cu_seqlens[batch_idx + 1] - pre_len;
    context_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                       q,
                                       k,
                                       reinterpret_cast<T*>(smem_),
                                       tidx,
                                       seq_idx,
                                       position_id,
                                       seq_len,
                                       input_len,
                                       PREFIX_PROMPT,
                                       prefix_prompt_length,
                                       param.count_length);

    if (use_logn_attn) {
        logn_attention(q, seq_idx, rope_config.max_pos);
    }

    __syncthreads();

    if (store_qkv) {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < head_num_kv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
#endif
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }
#ifdef ENABLE_FP8
        if (QuantizedQKV != nullptr) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
#endif
    }

    if (store_q_no_transpose) {
        size_t dest_q_no_transpose_idx =
            (pre_len + seq_idx) * head_num * size_per_head + head_idx * size_per_head + tidx * vec_size;

        *reinterpret_cast<Vec_t*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) = q;
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
        }

#ifdef ENABLE_FP8
        if (QuantizedQKV != nullptr) {
            // fp8 paged fmha
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        } else {
            // paged fmha
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
#else
        // paged fmha
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
#endif
    }

    if (store_kv) {
        const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                + tidx * vec_size;

        if (head_idx < head_num_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inBlockIdx =
                    kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max[2];
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max[2];
                    local_max[0] = vector_abs_max(k);
                    local_max[1] = vector_abs_max(v);
                    blockReduceMaxV2<float, 2>(local_max);
                    if (threadIdx.x == 0) {
                        s_max[0] = local_max[0];
                        s_max[1] = local_max[1];
                    }
                } else {
                    s_max[0] = float(1 << (8 - 1));
                    s_max[1] = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max[0]);
                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max[1]);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
                const int inBlockIdx =
                    kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*                             q_no_transpose_buf,
                                    T*                             q_buf,
                                    T*                             k_buf,
                                    T*                             v_buf,
                                    PrefixPromptBatchWeightsParam* param_ptr,
                                    T*                             QKV,
                                    void*                          QuantizedQKV,
                                    const int*                     position_ids,
                                    const T*                       qkv_bias,
                                    const int*                     padding_offset,
                                    const int*                     cu_seqlens,
                                    const int                      batch_size,
                                    const int                      seq_len,
                                    const int                      token_num,
                                    const int                      head_num,
                                    const int                      head_num_kv,
                                    const int                      size_per_head,
                                    const RopeConfig               rope_config,
                                    const bool                     use_logn_attn,
                                    const float*                   scale,
                                    const int                      int8_mode,
                                    const bool                     use_paged_fmha,
                                    const bool                     store_qkv,
                                    const bool                     store_q_no_transpose,
                                    const bool                     store_q,
                                    const bool                     store_kv,
                                    const bool                     store_cache,
                                    cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_kernel<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_no_transpose_buf,
                                                             q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q_no_transpose,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        });
    });
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T, typename Tcache, RopeStyle ROPE_STYLE>
__global__ void decode_add_fusedQKV_bias_transpose_with_rope_cache_kernel(T*           q_buf,
                                                                          T*           k_buf,
                                                                          T*           v_buf,
                                                                          KVBlockArray kv_block_array,
                                                                          T*           QKV,
                                                                          const int*   position_ids,
                                                                          const T* __restrict qkv_bias,
                                                                          const float* rope_cache,
                                                                          const int    batch_size,
                                                                          const int    head_num,
                                                                          const int    head_num_kv,
                                                                          const int    size_per_head,
                                                                          RopeConfig   rope_config,
                                                                          const bool   use_logn_attn,
                                                                          bool         store_q,
                                                                          bool         store_kv,
                                                                          bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     total_seq_len = seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int    position_id = position_ids[token_idx * rope_config.index_factor];
    bool   work        = false;
    float2 coef;
    if (bidy < head_num + head_num_kv) {
        constexpr int vec_size = vector_size<T, Vec_t>::size;
        const int     rope_idx = tidx * vec_size;
        work                   = (rope_idx >= 0 && rope_idx < rope_config.dim);
        if (work) {
            coef =
                *(reinterpret_cast<float2*>(const_cast<float*>(&rope_cache[position_id * rope_config.dim + tidx * 2])));
        }
    }

    if (bidy < head_num) {
        const int head_idx   = bidy;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q            = add(q, q_bias);
        }

        apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(q, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (use_logn_attn) {
            logn_attention(q, seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
    } else if (bidy < head_num + head_num_kv) {
        const int head_idx   = bidy - head_num;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k            = add(k, k_bias);
        }

        apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(k, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        __syncthreads();

        if (store_kv) {
            const int dst_kv_seq_idx = seq_idx;
            const int dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                    + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(k);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
            }
        }
    } else {
        const int head_idx   = bidy - head_num - head_num_kv;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v            = add(v, v_bias);
        }

        __syncthreads();

        if (store_kv) {
            const int dst_kv_seq_idx = seq_idx;
            const int dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                    + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(v);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T, typename Tcache, RopeStyle ROPE_STYLE>
__global__ void decode_add_fusedQKV_bias_transpose_kernel(T*           q_buf,
                                                          T*           k_buf,
                                                          T*           v_buf,
                                                          KVBlockArray kv_block_array,
                                                          T*           QKV,
                                                          const int*   position_ids,
                                                          const T* __restrict qkv_bias,
                                                          const int  batch_size,
                                                          const int  head_num,
                                                          const int  head_num_kv,
                                                          const int  size_per_head,
                                                          RopeConfig rope_config,
                                                          const bool use_logn_attn,
                                                          bool       store_q,
                                                          bool       store_kv,
                                                          bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     total_seq_len = seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }

    if (bidy < head_num) {
        const int head_idx   = bidy;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q            = add(q, q_bias);
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(rope_config, q, reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        if (use_logn_attn) {
            logn_attention(q, seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
    } else if (bidy < head_num + head_num_kv) {
        const int head_idx   = bidy - head_num;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k            = add(k, k_bias);
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(rope_config, k, reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        __syncthreads();

        if (store_kv) {
            const int dst_kv_seq_idx = seq_idx;
            const int dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                    + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(k);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
            }
        }
    } else {
        const int head_idx   = bidy - head_num - head_num_kv;
        const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const int src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v            = add(v, v_bias);
        }

        __syncthreads();

        if (store_kv) {
            const int dst_kv_seq_idx = seq_idx;
            const int dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                    + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(v);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T,
         typename Tcache,
         RopeStyle ROPE_STYLE,
         int       HEAD_Q_BLOCK_NUM,
         int       HEAD_K_BLOCK_NUM,
         int       HEAD_V_BLOCK_NUM>
__global__ void decode_add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel(T*           q_buf,
                                                                                   T*           k_buf,
                                                                                   T*           v_buf,
                                                                                   KVBlockArray kv_block_array,
                                                                                   T*           QKV,
                                                                                   const int*   position_ids,
                                                                                   const T* __restrict qkv_bias,
                                                                                   const float* rope_cache,
                                                                                   const int    batch_size,
                                                                                   const int    head_num,
                                                                                   const int    head_num_kv,
                                                                                   const int    size_per_head,
                                                                                   RopeConfig   rope_config,
                                                                                   const bool   use_logn_attn,
                                                                                   bool         store_q,
                                                                                   bool         store_kv,
                                                                                   bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     max_q_bidy    = head_num / HEAD_Q_BLOCK_NUM;
    const int     max_k_bidy    = max_q_bidy + head_num_kv / HEAD_K_BLOCK_NUM;
    const int     max_v_bidy    = max_k_bidy + head_num_kv / HEAD_V_BLOCK_NUM;
    const int     total_seq_len = seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int    position_id = position_ids[token_idx * rope_config.index_factor];
    bool   work        = false;
    float2 coef;
    if (bidy < max_k_bidy) {
        constexpr int vec_size = vector_size<T, Vec_t>::size;
        const int     rope_idx = tidx * vec_size;
        work                   = (rope_idx >= 0 && rope_idx < rope_config.dim);
        if (work) {
            coef =
                *(reinterpret_cast<float2*>(const_cast<float*>(&rope_cache[position_id * rope_config.dim + tidx * 2])));
        }
    }

    if (bidy < max_q_bidy) {
        Vec_t q[2];
        int   q_load_idx  = 0;
        int   q_store_idx = 0;
        int   q_idx_off   = 1;

        int head_idx   = bidy * HEAD_Q_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        int dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                         + seq_idx * size_per_head + tidx * vec_size;

        q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q[q_load_idx] = add(q[q_load_idx], q_bias);
        }

#pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            q_load_idx ^= q_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_q_idx += size_per_head;

            q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

            if (qkv_bias) {
                Vec_t q_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
                q[q_load_idx] = add(q[q_load_idx], q_bias);
            }

            apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(
                q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

            if (use_logn_attn) {
                logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
            }

            __syncthreads();

            if (store_q) {
                *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
                dest_q_idx += (size_per_head * seq_len);
            }

            q_store_idx ^= q_idx_off;
        }

        apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(
            q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (use_logn_attn) {
            logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        }
    } else if (bidy < max_k_bidy) {
        Vec_t k[2];
        int   k_load_idx  = 0;
        int   k_store_idx = 0;
        int   k_idx_off   = 1;

        int head_idx   = (bidy - max_q_bidy) * HEAD_K_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        const int dst_kv_seq_idx = seq_idx;
        int       dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                          + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     k_cache        = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k[k_load_idx] = add(k[k_load_idx], k_bias);
        }

#pragma unroll
        for (int h = 1; h < HEAD_K_BLOCK_NUM; ++h) {
            k_load_idx ^= k_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_k_idx += size_per_head;

            k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

            if (qkv_bias) {
                Vec_t k_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
                k[k_load_idx] = add(k[k_load_idx], k_bias);
            }

            apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(
                k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

            __syncthreads();

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            k_store_idx ^= k_idx_off;
        }

        apply_rope_with_cache<Vec_t, T, ROPE_STYLE>(
            k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        __syncthreads();

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
            }
        }
    } else {
        Vec_t v[2];
        int   v_load_idx  = 0;
        int   v_store_idx = 0;
        int   v_idx_off   = 1;

        int head_idx   = (bidy - max_k_bidy) * HEAD_V_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        const int dst_kv_seq_idx = seq_idx;
        int       dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                          + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     v_cache        = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v[v_load_idx] = add(v[v_load_idx], v_bias);
        }

        __syncthreads();

#pragma unroll
        for (int h = 1; h < HEAD_V_BLOCK_NUM; ++h) {
            v_load_idx ^= v_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_v_idx += size_per_head;

            v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

            if (qkv_bias) {
                Vec_t v_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
                v[v_load_idx] = add(v[v_load_idx], v_bias);
            }

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            v_store_idx ^= v_idx_off;

            __syncthreads();
        }

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
            }
        }
    }
}

template<typename T,
         typename Tcache,
         RopeStyle ROPE_STYLE,
         int       HEAD_Q_BLOCK_NUM,
         int       HEAD_K_BLOCK_NUM,
         int       HEAD_V_BLOCK_NUM>
__global__ void decode_add_fusedQKV_bias_transpose_non_int8_kernel(T*           q_buf,
                                                                   T*           k_buf,
                                                                   T*           v_buf,
                                                                   KVBlockArray kv_block_array,
                                                                   T*           QKV,
                                                                   const int*   position_ids,
                                                                   const T* __restrict qkv_bias,
                                                                   const int  batch_size,
                                                                   const int  head_num,
                                                                   const int  head_num_kv,
                                                                   const int  size_per_head,
                                                                   RopeConfig rope_config,
                                                                   const bool use_logn_attn,
                                                                   bool       store_q,
                                                                   bool       store_kv,
                                                                   bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     max_q_bidy    = head_num / HEAD_Q_BLOCK_NUM;
    const int     max_k_bidy    = max_q_bidy + head_num_kv / HEAD_K_BLOCK_NUM;
    const int     max_v_bidy    = max_k_bidy + head_num_kv / HEAD_V_BLOCK_NUM;
    const int     total_seq_len = seq_len;
    if (bidy >= max_v_bidy) {
        return;
    }

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }

    if (bidy < max_q_bidy) {
        Vec_t q[2];
        int   q_load_idx  = 0;
        int   q_store_idx = 0;
        int   q_idx_off   = 1;

        int head_idx   = bidy * HEAD_Q_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        int dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                         + seq_idx * size_per_head + tidx * vec_size;

        q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q[q_load_idx] = add(q[q_load_idx], q_bias);
        }

#pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            q_load_idx ^= q_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_q_idx += size_per_head;

            q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

            if (qkv_bias) {
                Vec_t q_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
                q[q_load_idx] = add(q[q_load_idx], q_bias);
            }

            apply_rope<T, Vec_t, ROPE_STYLE>(
                rope_config, q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

            if (use_logn_attn) {
                logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
            }

            __syncthreads();

            if (store_q) {
                *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
                dest_q_idx += (size_per_head * seq_len);
            }

            q_store_idx ^= q_idx_off;
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(
            rope_config, q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        if (use_logn_attn) {
            logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        }
    } else if (bidy < max_k_bidy) {
        Vec_t k[2];
        int   k_load_idx  = 0;
        int   k_store_idx = 0;
        int   k_idx_off   = 1;

        int head_idx   = (bidy - max_q_bidy) * HEAD_K_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        const int dst_kv_seq_idx = seq_idx;
        int       dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                          + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     k_cache        = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k[k_load_idx] = add(k[k_load_idx], k_bias);
        }

#pragma unroll
        for (int h = 1; h < HEAD_K_BLOCK_NUM; ++h) {
            k_load_idx ^= k_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_k_idx += size_per_head;

            k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

            if (qkv_bias) {
                Vec_t k_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
                k[k_load_idx] = add(k[k_load_idx], k_bias);
            }

            apply_rope<T, Vec_t, ROPE_STYLE>(
                rope_config, k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

            __syncthreads();

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            k_store_idx ^= k_idx_off;
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(
            rope_config, k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        __syncthreads();

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
            }
        }
    } else {
        Vec_t v[2];
        int   v_load_idx  = 0;
        int   v_store_idx = 0;
        int   v_idx_off   = 1;

        int head_idx   = (bidy - max_k_bidy) * HEAD_V_BLOCK_NUM;
        int hidden_idx = head_idx * size_per_head + tidx * vec_size;
        int src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        const int dst_kv_seq_idx = seq_idx;
        int       dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                          + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     v_cache        = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v[v_load_idx] = add(v[v_load_idx], v_bias);
        }

        __syncthreads();

#pragma unroll
        for (int h = 1; h < HEAD_V_BLOCK_NUM; ++h) {
            v_load_idx ^= v_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_v_idx += size_per_head;

            v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

            if (qkv_bias) {
                Vec_t v_bias  = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
                v[v_load_idx] = add(v[v_load_idx], v_bias);
            }

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            v_store_idx ^= v_idx_off;

            __syncthreads();
        }

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
            }
        }
    }
}

template<typename T>
void invokeDecodeAddFusedQKVBiasTranspose(T*               q_buf,
                                          T*               k_buf,
                                          T*               v_buf,
                                          KVBlockArray     kv_block_array,
                                          T*               QKV,
                                          const int*       position_ids,
                                          const T*         qkv_bias,
                                          const float*     rope_cache,
                                          const int        batch_size,
                                          const int        head_num,
                                          const int        head_num_kv,
                                          const int        size_per_head,
                                          const RopeConfig rope_config,
                                          const bool       use_logn_attn,
                                          const bool       store_q,
                                          const bool       store_kv,
                                          const bool       store_cache,
                                          cudaStream_t     stream) {
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);
    if ((rope_config.style == RopeStyle::Base || rope_config.style == RopeStyle::Yarn) && rope_cache) {
        constexpr int head_q_block_num = 4;
        constexpr int head_k_block_num = 4;
        constexpr int head_v_block_num = 4;
        if (batch_size <= 16 || head_num % head_q_block_num != 0 || head_num_kv % head_k_block_num != 0
            || head_num_kv % head_v_block_num != 0 || kv_block_array.cache_type == KvCacheDataType::INT8) {
            dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
            dim3 grid(batch_size, head_num + head_num_kv * 2);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_with_rope_cache_kernel<T, Tcache, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             rope_cache,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        } else {
            dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
            dim3 grid(batch_size,
                      head_num / head_q_block_num + head_num_kv / head_k_block_num + head_num_kv / head_v_block_num);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel<T,
                                                                                       Tcache,
                                                                                       ROPE_STYLE,
                                                                                       head_q_block_num,
                                                                                       head_k_block_num,
                                                                                       head_v_block_num>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             rope_cache,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        }
    } else {
        constexpr int head_q_block_num = 2;
        constexpr int head_k_block_num = 2;
        constexpr int head_v_block_num = 4;
        if (batch_size <= 16 || head_num % head_q_block_num != 0 || head_num_kv % head_k_block_num != 0
            || head_num_kv % head_v_block_num != 0 || kv_block_array.cache_type == KvCacheDataType::INT8) {
            dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
            dim3 grid(batch_size, head_num + head_num_kv * 2);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_kernel<T, Tcache, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        } else {
            dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
            dim3 grid(batch_size,
                      head_num / head_q_block_num + head_num_kv / head_k_block_num + head_num_kv / head_v_block_num);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_non_int8_kernel<T,
                                                                       Tcache,
                                                                       ROPE_STYLE,
                                                                       head_q_block_num,
                                                                       head_k_block_num,
                                                                       head_v_block_num>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        }
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#if USING_ROCM
inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const amd_bfloat162 u) {
    __hip_bfloat162_raw   raw_bf16  = *reinterpret_cast<const __hip_bfloat162_raw*>(&u);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_bfloat16raw2_to_fp8x2(raw_bf16, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<__hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const float2 u) {
    __half2               h2        = __float22half2_rn(u);
    __half2_raw           raw_h2    = *reinterpret_cast<const __half2_raw*>(&h2);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_halfraw2_to_fp8x2(raw_h2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<const __hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const uint32_t u) {
    __half2_raw           raw_h2    = *reinterpret_cast<const __half2_raw*>(&u);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_halfraw2_to_fp8x2(raw_h2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<const __hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_prefill_kernel_v1(T*                            q_buf,
                                                              T*                            k_buf,
                                                              T*                            v_buf,
                                                              PrefixPromptBatchWeightsParam param,
                                                              T*                            QKV,
                                                              void*                         QuantizedQKV,
                                                              const int*                    position_ids,
                                                              const T* __restrict qkv_bias,
                                                              const int*    padding_offset,
                                                              const int*    cu_seqlens,
                                                              const int     batch_size,
                                                              const int     seq_len,
                                                              const int     head_num,
                                                              const int     head_num_kv,
                                                              const int     size_per_head,
                                                              RopeConfig    rope_config,
                                                              const bool    use_logn_attn,
                                                              bool          store_qkv,
                                                              bool          store_q,
                                                              bool          store_kv,
                                                              bool          store_cache,
                                                              const float2* cos_sin_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3,
    // head_num, size_per_head], and QKV split to 3 split buffer q, k, v and
    // transpose them to [batch_size, head_num, seq_len, size_per_head]. For q and
    // k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt
    // and key/value along seq_len dimension like [prompt, key/value]. So, the
    // final shape of q is same ([batch_size, head_num, seq_len, size_per_head]),
    // but the shapes of key and values become [batch_size, head_num,
    // max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }
    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }
    const int pre_len   = cu_seqlens[batch_idx];
    const int input_len = cu_seqlens[batch_idx + 1] - pre_len;
    context_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                       q,
                                       k,
                                       reinterpret_cast<T*>(smem_),
                                       tidx,
                                       seq_idx,
                                       position_id,
                                       seq_len,
                                       input_len,
                                       PREFIX_PROMPT,
                                       prefix_prompt_length,
                                       param.count_length,
                                       cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, seq_idx, rope_config.max_pos);
    }

    __syncthreads();

    if (store_qkv) {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < head_num_kv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
#endif
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }
#ifdef ENABLE_FP8
        if (QuantizedQKV != nullptr) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
#endif
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
        }
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }

    if (store_kv) {
        const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                + tidx * vec_size;

        if (head_idx < head_num_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));

#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposePrefillV1(T*                             q_buf,
                                             T*                             k_buf,
                                             T*                             v_buf,
                                             PrefixPromptBatchWeightsParam* param_ptr,
                                             T*                             QKV,
                                             void*                          QuantizedQKV,
                                             const int*                     position_ids,
                                             const T*                       qkv_bias,
                                             const int*                     padding_offset,
                                             const int*                     cu_seqlens,
                                             const int                      batch_size,
                                             const int                      seq_len,
                                             const int                      token_num,
                                             const int                      head_num,
                                             const int                      head_num_kv,
                                             const int                      size_per_head,
                                             const RopeConfig               rope_config,
                                             const bool                     use_logn_attn,
                                             const float*                   scale,
                                             const int                      int8_mode,
                                             const bool                     use_paged_fmha,
                                             const bool                     store_qkv,
                                             const bool                     store_q,
                                             const bool                     store_kv,
                                             const bool                     store_cache,
                                             const float2*                  cos_sin_cache,
                                             cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_prefill_kernel_v1<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_prefill_kernel(T*                            q_buf,
                                                           T*                            k_buf,
                                                           T*                            v_buf,
                                                           PrefixPromptBatchWeightsParam param,
                                                           T*                            QKV,
                                                           void*                         QuantizedQKV,
                                                           const int*                    position_ids,
                                                           const T* __restrict qkv_bias,
                                                           const int*    padding_offset,
                                                           const int*    cu_seqlens,
                                                           const int     batch_size,
                                                           const int     seq_len,
                                                           const int     head_num,
                                                           const int     head_num_kv,
                                                           const int     size_per_head,
                                                           RopeConfig    rope_config,
                                                           const bool    use_logn_attn,
                                                           bool          store_qkv,
                                                           bool          store_q,
                                                           bool          store_kv,
                                                           bool          store_cache,
                                                           const float2* cos_sin_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3,
    // head_num, size_per_head], and QKV split to 3 split buffer q, k, v and
    // transpose them to [batch_size, head_num, seq_len, size_per_head]. For q and
    // k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt
    // and key/value along seq_len dimension like [prompt, key/value]. So, the
    // final shape of q is same ([batch_size, head_num, seq_len, size_per_head]),
    // but the shapes of key and values become [batch_size, head_num,
    // max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __hip_fp8_e4m3_fnuz;
    using QuantizedVecType = __hip_fp8x2_e4m3_fnuz;

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }
    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }
    const int pre_len   = cu_seqlens[batch_idx];
    const int input_len = cu_seqlens[batch_idx + 1] - pre_len;
    context_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                       q,
                                       k,
                                       reinterpret_cast<T*>(smem_),
                                       tidx,
                                       seq_idx,
                                       position_id,
                                       seq_len,
                                       input_len,
                                       PREFIX_PROMPT,
                                       prefix_prompt_length,
                                       param.count_length,
                                       cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, seq_idx, rope_config.max_pos);
    }

    __syncthreads();

    if (store_qkv) {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < head_num_kv) {
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }

        if (QuantizedQKV != nullptr) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
        }
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }

    if (store_kv) {
        const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                + tidx * vec_size;

        if (head_idx < head_num_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));

#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    // convert_to_fp8(reinterpret_cast<__nv_fp8_e4m3*>(k_cache) + inKBlockIdx,
                    // float(reinterpret_cast<T*>(&k)[vec_i]) * float(1 << (8 - 1)) / s_max[0]);
                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposePrefill(T*                             q_buf,
                                           T*                             k_buf,
                                           T*                             v_buf,
                                           PrefixPromptBatchWeightsParam* param_ptr,
                                           T*                             QKV,
                                           void*                          QuantizedQKV,
                                           const int*                     position_ids,
                                           const T*                       qkv_bias,
                                           const int*                     padding_offset,
                                           const int*                     cu_seqlens,
                                           const int                      batch_size,
                                           const int                      seq_len,
                                           const int                      token_num,
                                           const int                      head_num,
                                           const int                      head_num_kv,
                                           const int                      size_per_head,
                                           const RopeConfig               rope_config,
                                           const bool                     use_logn_attn,
                                           const float*                   scale,
                                           const int                      int8_mode,
                                           const bool                     use_paged_fmha,
                                           const bool                     store_qkv,
                                           const bool                     store_q,
                                           const bool                     store_kv,
                                           const bool                     store_cache,
                                           const float2*                  cos_sin_cache,
                                           cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_prefill_kernel<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_decode_kernel_v1(T*                            q_buf,
                                                             T*                            k_buf,
                                                             T*                            v_buf,
                                                             PrefixPromptBatchWeightsParam param,
                                                             const int*                    input_lengths,
                                                             T*                            QKV,
                                                             void*                         QuantizedQKV,
                                                             const int*                    position_ids,
                                                             const T* __restrict qkv_bias,
                                                             const int*    padding_offset,
                                                             const int*    cu_seqlens,
                                                             const int*    sequence_lengths,
                                                             const int     batch_size,
                                                             const int     seq_len,
                                                             const int     head_num,
                                                             const int     head_num_kv,
                                                             const int     size_per_head,
                                                             RopeConfig    rope_config,
                                                             const bool    use_logn_attn,
                                                             bool          store_qkv,
                                                             bool          store_q,
                                                             bool          store_kv,
                                                             bool          store_cache,
                                                             const float2* cos_sin_cache) {
    extern __shared__ __align__(sizeof(float2)) char smem_[];

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx = blockIdx.y;
    const int tidx     = threadIdx.x;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int sequence_length      = sequence_lengths[batch_idx];
    const int tlength              = sequence_length + param.max_prefix_prompt_length;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + tlength;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }

    // refer to the implementation of hipify decode attention
    const auto batch_beam_idx = blockIdx.y;
    const int  position_id    = position_ids == nullptr ? -1 : position_ids[token_idx * rope_config.index_factor];

    const int input_len = (input_lengths == nullptr) ? 0 : input_lengths[batch_beam_idx];
    const int timestep  = tlength;
    attention_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                         q,
                                         k,
                                         reinterpret_cast<T*>(smem_),
                                         tidx,
                                         tlength,
                                         tlength,  // timestep,
                                         sequence_length,
                                         position_id,
                                         input_len,
                                         prefix_prompt_length,
                                         true /*count_prefix_length*/,
                                         true /*HANDLE_KV*/,
                                         cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, tlength, rope_config.max_pos);
    }

    __syncthreads();

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_decode_kernel(T*                            q_buf,
                                                          T*                            k_buf,
                                                          T*                            v_buf,
                                                          PrefixPromptBatchWeightsParam param,
                                                          const int*                    input_lengths,
                                                          T*                            QKV,
                                                          void*                         QuantizedQKV,
                                                          const int*                    position_ids,
                                                          const T* __restrict qkv_bias,
                                                          const int*    padding_offset,
                                                          const int*    cu_seqlens,
                                                          const int*    sequence_lengths,
                                                          const int     batch_size,
                                                          const int     seq_len,
                                                          const int     head_num,
                                                          const int     head_num_kv,
                                                          const int     size_per_head,
                                                          RopeConfig    rope_config,
                                                          const bool    use_logn_attn,
                                                          bool          store_qkv,
                                                          bool          store_q,
                                                          bool          store_kv,
                                                          bool          store_cache,
                                                          const float2* cos_sin_cache) {
    extern __shared__ __align__(sizeof(float2)) char smem_[];

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int             batch_idx          = tgt_token_idx / seq_len;
    const int             seq_idx            = tgt_token_idx % seq_len;
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    const int head_idx = blockIdx.y;
    const int tidx     = threadIdx.x;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int sequence_length      = sequence_lengths[batch_idx];
    const int tlength              = sequence_length + param.max_prefix_prompt_length;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + tlength;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }

    // refer to the implementation of hipify decode attention
    const auto batch_beam_idx = blockIdx.y;
    const int  position_id    = position_ids == nullptr ? -1 : position_ids[token_idx * rope_config.index_factor];

    const int input_len = (input_lengths == nullptr) ? 0 : input_lengths[batch_beam_idx];
    const int timestep  = tlength;
    attention_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                         q,
                                         k,
                                         reinterpret_cast<T*>(smem_),
                                         tidx,
                                         tlength,
                                         tlength,  // timestep,
                                         sequence_length,
                                         position_id,
                                         input_len,
                                         prefix_prompt_length,
                                         true /*count_prefix_length*/,
                                         true /*HANDLE_KV*/,
                                         cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, tlength, rope_config.max_pos);
    }

    __syncthreads();

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }
    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposeDecodeV1(T*                             q_buf,
                                            T*                             k_buf,
                                            T*                             v_buf,
                                            PrefixPromptBatchWeightsParam* param_ptr,
                                            const int*                     input_lengths,
                                            T*                             QKV,
                                            void*                          QuantizedQKV,
                                            const int*                     position_ids,
                                            const T*                       qkv_bias,
                                            const int*                     padding_offset,
                                            const int*                     cu_seqlens,
                                            const int*                     sequence_lengths,
                                            const int                      batch_size,
                                            const int                      seq_len,
                                            const int                      token_num,
                                            const int                      head_num,
                                            const int                      head_num_kv,
                                            const int                      size_per_head,
                                            const RopeConfig               rope_config,
                                            const bool                     use_logn_attn,
                                            const float*                   scale,
                                            const int                      int8_mode,
                                            const bool                     use_paged_fmha,
                                            const bool                     store_qkv,
                                            const bool                     store_q,
                                            const bool                     store_kv,
                                            const bool                     store_cache,
                                            const float2*                  cos_sin_cache,
                                            cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_decode_kernel_v1<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             input_lengths,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             sequence_lengths,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

template<typename T>
void invokeAddFusedQKVBiasTransposeDecode(T*                             q_buf,
                                          T*                             k_buf,
                                          T*                             v_buf,
                                          PrefixPromptBatchWeightsParam* param_ptr,
                                          const int*                     input_lengths,
                                          T*                             QKV,
                                          void*                          QuantizedQKV,
                                          const int*                     position_ids,
                                          const T*                       qkv_bias,
                                          const int*                     padding_offset,
                                          const int*                     cu_seqlens,
                                          const int*                     sequence_lengths,
                                          const int                      batch_size,
                                          const int                      seq_len,
                                          const int                      token_num,
                                          const int                      head_num,
                                          const int                      head_num_kv,
                                          const int                      size_per_head,
                                          const RopeConfig               rope_config,
                                          const bool                     use_logn_attn,
                                          const float*                   scale,
                                          const int                      int8_mode,
                                          const bool                     use_paged_fmha,
                                          const bool                     store_qkv,
                                          const bool                     store_q,
                                          const bool                     store_kv,
                                          const bool                     store_cache,
                                          const float2*                  cos_sin_cache,
                                          cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_decode_kernel<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             input_lengths,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             sequence_lengths,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}
#endif

template<typename T, typename Tcache>
__global__ void load_prefix_KVCache_kernel(T*                            q_buf,
                                           T*                            k_buf,
                                           T*                            v_buf,
                                           PrefixPromptBatchWeightsParam param,
                                           const int                     seq_len,
                                           const int                     head_num,
                                           const int                     head_num_kv,
                                           const int                     size_per_head) {
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t            = typename Vec_t<T>::Type;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }
    // NOTE: blockIdx.x < batch_size * param.max_prefix_prompt_length really handles prefix prompts

    if (head_idx < head_num_kv) {
        const int prompt_batch_idx = blockIdx.x / param.max_prefix_prompt_length;
        const int prompt_seq_idx   = blockIdx.x % param.max_prefix_prompt_length;
        const int prompt_length    = param.d_prefix_prompt_lengths[prompt_batch_idx];

        if (prompt_seq_idx < prompt_length) {
            const int dest_kv_idx = prompt_batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + prompt_seq_idx * size_per_head
                                    + tidx * vec_size;
            if (param.kv_block_array.mMaxSeqs > 0) {
                Tcache* k_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getKBlockPtr(prompt_batch_idx, prompt_seq_idx));
                Tcache* v_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getVBlockPtr(prompt_batch_idx, prompt_seq_idx));
                const int inBlockIdx =
                    param.kv_block_array.getKVLocalIdx(prompt_seq_idx, head_idx, size_per_head, tidx * vec_size);

                if constexpr (ENABLE_8BITS_CACHE) {
                    float* k_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getKScalePtr(prompt_batch_idx, prompt_seq_idx));
                    float* v_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getVScalePtr(prompt_batch_idx, prompt_seq_idx));
                    int inScaleIdx = param.kv_block_array.getKVScaleLocalIdx(prompt_seq_idx, head_idx);
                    load_8bits_kv_cache_vec(
                        reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]), k_cache, inBlockIdx, k_scale_ptr[inScaleIdx]);
                    load_8bits_kv_cache_vec(
                        reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]), v_cache, inBlockIdx, v_scale_ptr[inScaleIdx]);
                } else {
                    *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) =
                        *reinterpret_cast<const Vec_t*>(&k_cache[inBlockIdx]);
                    *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) =
                        *reinterpret_cast<const Vec_t*>(&v_cache[inBlockIdx]);
                }
            }
        }
    }
}

#if USING_ROCM
template<typename T>
__global__ void gather_sequences_kernel_combined_v2(T*         output_q,
                                                    T*         output_k,
                                                    T*         output_v,
                                                    const T*   input_q,
                                                    const T*   input_k,
                                                    const T*   input_v,
                                                    const int* cu_seqlens,
                                                    const int* cu_kv_seqlens,
                                                    int        batch_size,
                                                    int        seq_len,
                                                    int        seq_len_with_prefix,
                                                    int        head_num_q,
                                                    int        head_num_k,
                                                    int        head_num_v,
                                                    int        size_per_head) {
    int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int head_idx  = blockIdx.y;

    bool is_q = head_idx < head_num_q;
    bool is_k = head_idx >= head_num_q && head_idx < head_num_q + head_num_k;
    bool is_v = head_idx >= head_num_q + head_num_k;

    if (!is_q && !is_k && !is_v)
        return;

    int real_head_idx;
    int total_tokens;
    int max_seq_len;

    if (is_q) {
        real_head_idx = head_idx;
        total_tokens  = cu_seqlens[batch_size];
        max_seq_len   = seq_len;
    } else if (is_k) {
        real_head_idx = head_idx - head_num_q;
        total_tokens  = cu_kv_seqlens[batch_size];
        max_seq_len   = seq_len_with_prefix;
    } else {
        real_head_idx = head_idx - head_num_q - head_num_k;
        total_tokens  = cu_kv_seqlens[batch_size];
        max_seq_len   = seq_len_with_prefix;
    }

    if (token_idx >= total_tokens)
        return;

    // caculate sample_id 和 pos_in_sample
    int sample_id     = 0;
    int pos_in_sample = token_idx;

    const int* current_seqlens = is_q ? cu_seqlens : cu_kv_seqlens;

    while (sample_id < batch_size && pos_in_sample >= (current_seqlens[sample_id + 1] - current_seqlens[sample_id])) {
        pos_in_sample -= (current_seqlens[sample_id + 1] - current_seqlens[sample_id]);
        sample_id++;
    }
    if (sample_id >= batch_size)
        return;

    // caculate input and output index
    if (is_q) {
        int input_idx = sample_id * head_num_q * max_seq_len * size_per_head
                        + real_head_idx * max_seq_len * size_per_head + pos_in_sample * size_per_head;

        int output_idx = real_head_idx * max_seq_len * batch_size * size_per_head + token_idx * size_per_head;

        for (int s = 0; s < size_per_head; ++s) {
            output_q[output_idx + s] = input_q[input_idx + s];
        }
    } else if (is_k) {
        int input_idx = sample_id * head_num_k * max_seq_len * size_per_head
                        + real_head_idx * max_seq_len * size_per_head + pos_in_sample * size_per_head;

        int output_idx = real_head_idx * max_seq_len * batch_size * size_per_head + token_idx * size_per_head;

        for (int s = 0; s < size_per_head; ++s) {
            output_k[output_idx + s] = input_k[input_idx + s];
        }
    } else if (is_v) {
        int input_idx = sample_id * head_num_v * max_seq_len * size_per_head
                        + real_head_idx * max_seq_len * size_per_head + pos_in_sample * size_per_head;

        int output_idx = real_head_idx * max_seq_len * batch_size * size_per_head + token_idx * size_per_head;

        for (int s = 0; s < size_per_head; ++s) {
            output_v[output_idx + s] = input_v[input_idx + s];
        }
    }
}

template<typename T, typename Tcache>
__global__ void load_prefix_KVCache_kernel_aiter_v1(T*                            q_buf,
                                                    T*                            k_buf,
                                                    T*                            v_buf,
                                                    PrefixPromptBatchWeightsParam param,
                                                    const int                     seq_len,
                                                    const int                     head_num,
                                                    const int                     head_num_kv,
                                                    const int                     size_per_head) {
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t            = typename Vec_t<T>::Type;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }
    // NOTE: blockIdx.x < batch_size * param.max_prefix_prompt_length really handles prefix prompts

    if (head_idx < head_num_kv) {
        const int prompt_batch_idx = blockIdx.x / param.max_prefix_prompt_length;
        const int prompt_seq_idx   = blockIdx.x % param.max_prefix_prompt_length;
        const int prompt_length    = param.d_prefix_prompt_lengths[prompt_batch_idx];

        if (prompt_seq_idx < prompt_length) {
            const int dest_kv_idx = prompt_batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + prompt_seq_idx * size_per_head
                                    + tidx * vec_size;
            if (param.kv_block_array.mMaxSeqs > 0) {
                Tcache* k_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getKBlockPtr(prompt_batch_idx, prompt_seq_idx));
                Tcache* v_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getVBlockPtr(prompt_batch_idx, prompt_seq_idx));
                const int inKBlockIdx = param.kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                    prompt_seq_idx, head_idx, size_per_head, tidx * vec_size);

                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inVBlockIdx = param.kv_block_array.getVLocalIdx(
                        prompt_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_buf[dest_kv_idx + vec_i] = *reinterpret_cast<const T*>(&v_cache[inVBlockIdx]);
                }

                if constexpr (ENABLE_8BITS_CACHE) {
                    float* k_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getKScalePtr(prompt_batch_idx, prompt_seq_idx));
                    float* v_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getVScalePtr(prompt_batch_idx, prompt_seq_idx));
                    int inScaleIdx = param.kv_block_array.getKVScaleLocalIdx(prompt_seq_idx, head_idx);
                    for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                        const int inVBlockIdx = param.kv_block_array.getVLocalIdx(
                            prompt_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                        load_8bits_kv_cache_vec(reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]),
                                                v_cache,
                                                inVBlockIdx,
                                                v_scale_ptr[inScaleIdx]);
                    }
                    load_8bits_kv_cache_vec(
                        reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]), k_cache, inKBlockIdx, k_scale_ptr[inScaleIdx]);
                } else {
                    *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) =
                        *reinterpret_cast<const Vec_t*>(&k_cache[inKBlockIdx]);
                }
            }
        }
    }
}

template<typename T, typename Tcache>
__global__ void load_prefix_KVCache_kernel_aiter(T*                            q_buf,
                                                 T*                            k_buf,
                                                 T*                            v_buf,
                                                 PrefixPromptBatchWeightsParam param,
                                                 const int                     seq_len,
                                                 const int                     head_num,
                                                 const int                     head_num_kv,
                                                 const int                     size_per_head) {
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t            = typename Vec_t<T>::Type;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }
    // NOTE: blockIdx.x < batch_size * param.max_prefix_prompt_length really handles prefix prompts

    if (head_idx < head_num_kv) {
        const int prompt_batch_idx = blockIdx.x / param.max_prefix_prompt_length;
        const int prompt_seq_idx   = blockIdx.x % param.max_prefix_prompt_length;
        const int prompt_length    = param.d_prefix_prompt_lengths[prompt_batch_idx];

        if (prompt_seq_idx < prompt_length) {
            const int dest_kv_idx = prompt_batch_idx * size_per_head * total_seq_len * head_num_kv
                                    + head_idx * size_per_head * total_seq_len + prompt_seq_idx * size_per_head
                                    + tidx * vec_size;
            if (param.kv_block_array.mMaxSeqs > 0) {
                Tcache* k_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getKBlockPtr(prompt_batch_idx, prompt_seq_idx));
                Tcache* v_cache =
                    reinterpret_cast<Tcache*>(param.kv_block_array.getVBlockPtr(prompt_batch_idx, prompt_seq_idx));
                const int inKBlockIdx = param.kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                    prompt_seq_idx, head_idx, size_per_head, tidx * vec_size);

                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inVBlockIdx = param.kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                        prompt_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_buf[dest_kv_idx + vec_i] = *reinterpret_cast<const T*>(&v_cache[inVBlockIdx]);
                }

                if constexpr (ENABLE_8BITS_CACHE) {
                    float* k_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getKScalePtr(prompt_batch_idx, prompt_seq_idx));
                    float* v_scale_ptr =
                        reinterpret_cast<float*>(param.kv_block_array.getVScalePtr(prompt_batch_idx, prompt_seq_idx));
                    int inScaleIdx = param.kv_block_array.getKVScaleLocalIdx(prompt_seq_idx, head_idx);
                    for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                        const int inVBlockIdx = param.kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                            prompt_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                        load_8bits_kv_cache_vec(reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]),
                                                v_cache,
                                                inVBlockIdx,
                                                v_scale_ptr[inScaleIdx]);
                    }
                    load_8bits_kv_cache_vec(
                        reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]), k_cache, inKBlockIdx, k_scale_ptr[inScaleIdx]);
                } else {
                    *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) =
                        *reinterpret_cast<const Vec_t*>(&k_cache[inKBlockIdx]);
                }
            }
        }
    }
}
#endif

template<typename T>
void invokeLoadPrefixKVCache(T*                             q_buf,
                             T*                             k_buf,
                             T*                             v_buf,
                             PrefixPromptBatchWeightsParam* param_ptr,
                             const int                      batch_size,
                             const int                      seq_len,
                             const int                      head_num,
                             const int                      head_num_kv,
                             const int                      size_per_head,
                             const float*                   scale,
                             const int                      int8_mode,
                             cudaStream_t                   stream) {
    auto& param = *param_ptr;
    dim3  block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3  grid(batch_size * param.max_prefix_prompt_length, head_num);

    FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
        load_prefix_KVCache_kernel<T, Tcache>
            <<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, param, seq_len, head_num, head_num_kv, size_per_head);
    });
}

#if USING_ROCM
template<typename T>
void invokeGatherSequencesCombined(T*           output_q,
                                   T*           output_k,
                                   T*           output_v,
                                   const T*     input_q,
                                   const T*     input_k,
                                   const T*     input_v,
                                   const int*   cu_seqlens,
                                   const int*   cu_kv_seqlens,
                                   int          batch_size,
                                   int          seq_len,
                                   int          seq_len_with_prefix,
                                   int          head_num_q,
                                   int          head_num_kv,
                                   int          size_per_head,
                                   cudaStream_t stream) {
    int total_heads = head_num_q + 2 * head_num_kv;  // q heads + k heads + v heads

    // 计算最大token数用于网格配置
    int max_tokens        = max(cu_seqlens[batch_size], cu_kv_seqlens[batch_size]);
    int threads_per_block = 256;

    dim3 block_size(threads_per_block, 1);
    dim3 grid_size((max_tokens + threads_per_block - 1) / threads_per_block, total_heads);

    gather_sequences_kernel_combined_v2<T><<<grid_size, block_size, 0, stream>>>(output_q,
                                                                                 output_k,
                                                                                 output_v,
                                                                                 input_q,
                                                                                 input_k,
                                                                                 input_v,
                                                                                 cu_seqlens,
                                                                                 cu_kv_seqlens,
                                                                                 batch_size,
                                                                                 seq_len,
                                                                                 seq_len_with_prefix,
                                                                                 head_num_q,
                                                                                 head_num_kv,  // k heads
                                                                                 head_num_kv,  // v heads
                                                                                 size_per_head);
}

template<typename T>
void invokeLoadPrefixKVCacheAiterV1(T*                             q_buf,
                                    T*                             k_buf,
                                    T*                             v_buf,
                                    PrefixPromptBatchWeightsParam* param_ptr,
                                    const int                      batch_size,
                                    const int                      seq_len,
                                    const int                      head_num,
                                    const int                      head_num_kv,
                                    const int                      size_per_head,
                                    const float*                   scale,
                                    const int                      int8_mode,
                                    cudaStream_t                   stream) {
    auto& param = *param_ptr;
    dim3  block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3  grid(batch_size * param.max_prefix_prompt_length, head_num);

    FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
        load_prefix_KVCache_kernel_aiter_v1<T, Tcache>
            <<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, param, seq_len, head_num, head_num_kv, size_per_head);
    });
}

template<typename T>
void invokeLoadPrefixKVCacheAiter(T*                             q_buf,
                                  T*                             k_buf,
                                  T*                             v_buf,
                                  PrefixPromptBatchWeightsParam* param_ptr,
                                  const int                      batch_size,
                                  const int                      seq_len,
                                  const int                      head_num,
                                  const int                      head_num_kv,
                                  const int                      size_per_head,
                                  const float*                   scale,
                                  const int                      int8_mode,
                                  cudaStream_t                   stream) {
    auto& param = *param_ptr;
    dim3  block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3  grid(batch_size * param.max_prefix_prompt_length, head_num);

    FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
        load_prefix_KVCache_kernel_aiter<T, Tcache>
            <<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, param, seq_len, head_num, head_num_kv, size_per_head);
    });
}
#endif

template<typename T>
__global__ void SplitQKV_kernel(T*        q_buf,
                                T*        k_buf,
                                T*        v_buf,
                                T*        QKV,
                                const int token_num,
                                const int head_num,
                                const int head_num_kv,
                                const int size_per_head) {
    // QKV: [token_num, 3, n]
    // q_buf, k_buf, v_buf: [token_num, head_num, size_per_head] [token_num, head_num_kv, size_per_head] * 2
    // grid(token_num, head_num + 2 * head_num_kv)
    // block(size_per_head)
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int size_id   = threadIdx.x;
    if (size_id >= size_per_head) {
        return;
    }
    const int qkv_offset =
        token_idx * (head_num + head_num_kv * 2) * size_per_head + head_idx * size_per_head + size_id;

    T val = ldg(&QKV[qkv_offset]);

    if (head_idx < head_num) {
        q_buf[token_idx * head_num * size_per_head + head_idx * size_per_head + size_id] = val;
    } else if (head_idx < head_num + head_num_kv) {
        k_buf[token_idx * head_num_kv * size_per_head + (head_idx - head_num) * size_per_head + size_id] = val;
    } else {
        v_buf[token_idx * head_num_kv * size_per_head + (head_idx - head_num - head_num_kv) * size_per_head + size_id] =
            val;
    }
}

template<typename T>
void invokeSplitQKV(T*           q_buf,
                    T*           k_buf,
                    T*           v_buf,
                    T*           QKV,
                    const int    token_num,
                    const int    head_num,
                    const int    head_num_kv,
                    const int    size_per_head,
                    cudaStream_t stream) {
    dim3 block(size_per_head);
    dim3 grid(token_num, head_num + 2 * head_num_kv);
    SplitQKV_kernel<<<grid, block, 0, stream>>>(
        q_buf, k_buf, v_buf, QKV, token_num, head_num, head_num_kv, size_per_head);
}

#define INSTANTIATESPLITQKV(T)                                                                                         \
    template void invokeSplitQKV(T*           q_buf,                                                                   \
                                 T*           k_buf,                                                                   \
                                 T*           v_buf,                                                                   \
                                 T*           QKV,                                                                     \
                                 const int    token_num,                                                               \
                                 const int    head_num,                                                                \
                                 const int    head_num_kv,                                                             \
                                 const int    size_per_head,                                                           \
                                 cudaStream_t stream)

INSTANTIATESPLITQKV(float);
INSTANTIATESPLITQKV(half);
#ifdef ENABLE_BF16
INSTANTIATESPLITQKV(__nv_bfloat16);
#endif
#undef INSTANTIATESPLITQKV

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(T)                                                                         \
    template void invokeAddFusedQKVBiasTranspose(T*                             q_no_transpose_buf,                    \
                                                 T*                             q_buf,                                 \
                                                 T*                             k_buf,                                 \
                                                 T*                             v_buf,                                 \
                                                 PrefixPromptBatchWeightsParam* param,                                 \
                                                 T*                             QKV,                                   \
                                                 void*                          QuantizedQKV,                          \
                                                 const int*                     position_ids,                          \
                                                 const T*                       qkv_bias,                              \
                                                 const int*                     padding_offset,                        \
                                                 const int*                     cu_seqlens,                            \
                                                 const int                      batch_size,                            \
                                                 const int                      seq_len,                               \
                                                 const int                      token_num,                             \
                                                 const int                      head_num,                              \
                                                 const int                      head_num_kv,                           \
                                                 const int                      size_per_head,                         \
                                                 const RopeConfig               rope_config,                           \
                                                 const bool                     use_logn_attn,                         \
                                                 const float*                   scale,                                 \
                                                 const int                      int8_mode,                             \
                                                 const bool                     use_paged_fmha,                        \
                                                 const bool                     store_qkv,                             \
                                                 const bool                     store_q_no_transpose,                  \
                                                 const bool                     store_q,                               \
                                                 const bool                     store_kv,                              \
                                                 const bool                     store_cache,                           \
                                                 cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSE

#define INSTANTIATEDECODEADDFUSEDQKVBIASTRANSPOSE(T)                                                                   \
    template void invokeDecodeAddFusedQKVBiasTranspose(T*               q_buf,                                         \
                                                       T*               k_buf,                                         \
                                                       T*               v_buf,                                         \
                                                       KVBlockArray     kv_block_array,                                \
                                                       T*               QKV,                                           \
                                                       const int*       position_ids,                                  \
                                                       const T*         qkv_bias,                                      \
                                                       const float*     rope_cache,                                    \
                                                       const int        batch_size,                                    \
                                                       const int        head_num,                                      \
                                                       const int        head_num_kv,                                   \
                                                       const int        size_per_head,                                 \
                                                       const RopeConfig rope_config,                                   \
                                                       const bool       use_logn_attn,                                 \
                                                       const bool       store_q,                                       \
                                                       const bool       store_kv,                                      \
                                                       const bool       store_cache,                                   \
                                                       cudaStream_t     stream)
INSTANTIATEDECODEADDFUSEDQKVBIASTRANSPOSE(float);
INSTANTIATEDECODEADDFUSEDQKVBIASTRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATEDECODEADDFUSEDQKVBIASTRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATEDECODEADDFUSEDQKVBIASTRANSPOSE
#if USING_ROCM

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(T)                                                                \
    template void invokeAddFusedQKVBiasTransposePrefillV1(T*                             q_buf,                        \
                                                          T*                             k_buf,                        \
                                                          T*                             v_buf,                        \
                                                          PrefixPromptBatchWeightsParam* param,                        \
                                                          T*                             QKV,                          \
                                                          void*                          QuantizedQKV,                 \
                                                          const int*                     position_ids,                 \
                                                          const T*                       qkv_bias,                     \
                                                          const int*                     padding_offset,               \
                                                          const int*                     cu_seqlens,                   \
                                                          const int                      batch_size,                   \
                                                          const int                      seq_len,                      \
                                                          const int                      token_num,                    \
                                                          const int                      head_num,                     \
                                                          const int                      head_num_kv,                  \
                                                          const int                      size_per_head,                \
                                                          const RopeConfig               rope_config,                  \
                                                          const bool                     use_logn_attn,                \
                                                          const float*                   scale,                        \
                                                          const int                      int8_mode,                    \
                                                          const bool                     use_paged_fmha,               \
                                                          const bool                     store_qkv,                    \
                                                          const bool                     store_q,                      \
                                                          const bool                     store_kv,                     \
                                                          const bool                     store_cache,                  \
                                                          const float2*                  cos_sin_cache,                \
                                                          cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(T)                                                                  \
    template void invokeAddFusedQKVBiasTransposePrefill(T*                             q_buf,                          \
                                                        T*                             k_buf,                          \
                                                        T*                             v_buf,                          \
                                                        PrefixPromptBatchWeightsParam* param,                          \
                                                        T*                             QKV,                            \
                                                        void*                          QuantizedQKV,                   \
                                                        const int*                     position_ids,                   \
                                                        const T*                       qkv_bias,                       \
                                                        const int*                     padding_offset,                 \
                                                        const int*                     cu_seqlens,                     \
                                                        const int                      batch_size,                     \
                                                        const int                      seq_len,                        \
                                                        const int                      token_num,                      \
                                                        const int                      head_num,                       \
                                                        const int                      head_num_kv,                    \
                                                        const int                      size_per_head,                  \
                                                        const RopeConfig               rope_config,                    \
                                                        const bool                     use_logn_attn,                  \
                                                        const float*                   scale,                          \
                                                        const int                      int8_mode,                      \
                                                        const bool                     use_paged_fmha,                 \
                                                        const bool                     store_qkv,                      \
                                                        const bool                     store_q,                        \
                                                        const bool                     store_kv,                       \
                                                        const bool                     store_cache,                    \
                                                        const float2*                  cos_sin_cache,                  \
                                                        cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(T)                                                                 \
    template void invokeAddFusedQKVBiasTransposeDecodeV1(T*                             q_buf,                         \
                                                         T*                             k_buf,                         \
                                                         T*                             v_buf,                         \
                                                         PrefixPromptBatchWeightsParam* param,                         \
                                                         const int*                     input_lengths,                 \
                                                         T*                             QKV,                           \
                                                         void*                          QuantizedQKV,                  \
                                                         const int*                     position_ids,                  \
                                                         const T*                       qkv_bias,                      \
                                                         const int*                     padding_offset,                \
                                                         const int*                     cu_seqlens,                    \
                                                         const int*                     sequence_lengths,              \
                                                         const int                      batch_size,                    \
                                                         const int                      seq_len,                       \
                                                         const int                      token_num,                     \
                                                         const int                      head_num,                      \
                                                         const int                      head_num_kv,                   \
                                                         const int                      size_per_head,                 \
                                                         const RopeConfig               rope_config,                   \
                                                         const bool                     use_logn_attn,                 \
                                                         const float*                   scale,                         \
                                                         const int                      int8_mode,                     \
                                                         const bool                     use_paged_fmha,                \
                                                         const bool                     store_qkv,                     \
                                                         const bool                     store_q,                       \
                                                         const bool                     store_kv,                      \
                                                         const bool                     store_cache,                   \
                                                         const float2*                  cos_sin_cache,                 \
                                                         cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(T)                                                                   \
    template void invokeAddFusedQKVBiasTransposeDecode(T*                             q_buf,                           \
                                                       T*                             k_buf,                           \
                                                       T*                             v_buf,                           \
                                                       PrefixPromptBatchWeightsParam* param,                           \
                                                       const int*                     input_lengths,                   \
                                                       T*                             QKV,                             \
                                                       void*                          QuantizedQKV,                    \
                                                       const int*                     position_ids,                    \
                                                       const T*                       qkv_bias,                        \
                                                       const int*                     padding_offset,                  \
                                                       const int*                     cu_seqlens,                      \
                                                       const int*                     sequence_lengths,                \
                                                       const int                      batch_size,                      \
                                                       const int                      seq_len,                         \
                                                       const int                      token_num,                       \
                                                       const int                      head_num,                        \
                                                       const int                      head_num_kv,                     \
                                                       const int                      size_per_head,                   \
                                                       const RopeConfig               rope_config,                     \
                                                       const bool                     use_logn_attn,                   \
                                                       const float*                   scale,                           \
                                                       const int                      int8_mode,                       \
                                                       const bool                     use_paged_fmha,                  \
                                                       const bool                     store_qkv,                       \
                                                       const bool                     store_q,                         \
                                                       const bool                     store_kv,                        \
                                                       const bool                     store_cache,                     \
                                                       const float2*                  cos_sin_cache,                   \
                                                       cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE
#endif

#if USING_ROCM
#define INSTANTIATEINVOKELOADPREFIXKVCACHEAITERV1(T)                                                                   \
    template void invokeLoadPrefixKVCacheAiterV1(T*                             q_buf,                                 \
                                                 T*                             k_buf,                                 \
                                                 T*                             v_buf,                                 \
                                                 PrefixPromptBatchWeightsParam* param,                                 \
                                                 const int                      batch_size,                            \
                                                 const int                      seq_len,                               \
                                                 const int                      head_num,                              \
                                                 const int                      head_num_kv,                           \
                                                 const int                      size_per_head,                         \
                                                 const float*                   scale,                                 \
                                                 const int                      int8_mode,                             \
                                                 cudaStream_t                   stream)
INSTANTIATEINVOKELOADPREFIXKVCACHEAITERV1(float);
INSTANTIATEINVOKELOADPREFIXKVCACHEAITERV1(half);
#ifdef ENABLE_BF16
INSTANTIATEINVOKELOADPREFIXKVCACHEAITERV1(__nv_bfloat16);
#endif
#undef INSTANTIATEINVOKELOADPREFIXKVCACHEAITERV1

#define INSTANTIATEINVOKELOADPREFIXKVCACHEAITER(T)                                                                     \
    template void invokeLoadPrefixKVCacheAiter(T*                             q_buf,                                   \
                                               T*                             k_buf,                                   \
                                               T*                             v_buf,                                   \
                                               PrefixPromptBatchWeightsParam* param,                                   \
                                               const int                      batch_size,                              \
                                               const int                      seq_len,                                 \
                                               const int                      head_num,                                \
                                               const int                      head_num_kv,                             \
                                               const int                      size_per_head,                           \
                                               const float*                   scale,                                   \
                                               const int                      int8_mode,                               \
                                               cudaStream_t                   stream)
INSTANTIATEINVOKELOADPREFIXKVCACHEAITER(float);
INSTANTIATEINVOKELOADPREFIXKVCACHEAITER(half);
#ifdef ENABLE_BF16
INSTANTIATEINVOKELOADPREFIXKVCACHEAITER(__nv_bfloat16);
#endif
#undef INSTANTIATEINVOKELOADPREFIXKVCACHEAITER

#define INSTANTIATEINVOKEGATHERSEQUENCESCOMBIED(T)                                                                     \
    template void invokeGatherSequencesCombined<T>(                                                                    \
        T*, T*, T*, const T*, const T*, const T*, const int*, const int*, int, int, int, int, int, int, cudaStream_t);

INSTANTIATEINVOKEGATHERSEQUENCESCOMBIED(float);
INSTANTIATEINVOKEGATHERSEQUENCESCOMBIED(half);

#ifdef ENABLE_BF16
INSTANTIATEINVOKEGATHERSEQUENCESCOMBIED(__nv_bfloat16);
#endif

#undef INSTANTIATEINVOKEGATHERSEQUENCESCOMBIED

#endif

#define INSTANTIATEINVOKELOADPREFIXKVCACHE(T)                                                                          \
    template void invokeLoadPrefixKVCache(T*                             q_buf,                                        \
                                          T*                             k_buf,                                        \
                                          T*                             v_buf,                                        \
                                          PrefixPromptBatchWeightsParam* param,                                        \
                                          const int                      batch_size,                                   \
                                          const int                      seq_len,                                      \
                                          const int                      head_num,                                     \
                                          const int                      head_num_kv,                                  \
                                          const int                      size_per_head,                                \
                                          const float*                   scale,                                        \
                                          const int                      int8_mode,                                    \
                                          cudaStream_t                   stream)
INSTANTIATEINVOKELOADPREFIXKVCACHE(float);
INSTANTIATEINVOKELOADPREFIXKVCACHE(half);
#ifdef ENABLE_BF16
INSTANTIATEINVOKELOADPREFIXKVCACHE(__nv_bfloat16);
#endif
#undef INSTANTIATEINVOKELOADPREFIXKVCACHE

}  // namespace rtp_llm
