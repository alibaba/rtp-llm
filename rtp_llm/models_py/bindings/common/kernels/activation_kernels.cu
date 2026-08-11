/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "rtp_llm/models_py/bindings/common/kernels/activation_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <cuda_fp8.h>
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

template<typename T>
__global__ void addBiasGelu(T* output, const T* bias, size_t numel, size_t hidden_size) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const float x = static_cast<float>(output[index]) + static_cast<float>(bias[index % hidden_size]);
    output[index] = static_cast<T>(0.5f * x * (1.0f + erff(x * 0.7071067811865475f)));
}

#if USING_CUDA
__device__ __forceinline__ float exactGelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

template<typename T>
__global__ void addBiasGeluQuantFp8Kernel(const T* __restrict__ input,
                                          const T* __restrict__ bias,
                                          __nv_fp8_e4m3* __restrict__ output,
                                          uint32_t* __restrict__ scales,
                                          size_t hidden_size,
                                          size_t scale_stride) {
    constexpr int    group_size = 128;
    constexpr int    threads    = 32;
    __shared__ float values[group_size];

    const size_t groups_per_row = hidden_size / group_size;
    const size_t group_id       = blockIdx.x;
    const size_t row            = group_id / groups_per_row;
    const size_t group_col      = group_id % groups_per_row;
    const size_t offset         = row * hidden_size + group_col * group_size;

    float local_absmax = 1e-4f;
#pragma unroll
    for (int i = threadIdx.x; i < group_size; i += threads) {
        const float value =
            exactGelu(static_cast<float>(input[offset + i]) + static_cast<float>(bias[group_col * group_size + i]));
        const T rounded_value = static_cast<T>(value);
        values[i]             = static_cast<float>(rounded_value);
        local_absmax          = fmaxf(local_absmax, fabsf(values[i]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, delta));
    }

    const float scale = exp2f(ceilf(log2f(fmaxf(local_absmax / 448.0f, 1e-10f))));
    if (threadIdx.x == 0) {
        const size_t packed_col = group_col / 4;
        const size_t pack_idx   = group_col % 4;
        reinterpret_cast<uint8_t*>(scales)[packed_col * scale_stride * 4 + row * 4 + pack_idx] =
            static_cast<uint8_t>(static_cast<int>(log2f(scale)) + 127);
    }
    __syncthreads();

#pragma unroll
    for (int i = threadIdx.x; i < group_size; i += threads) {
        output[offset + i] = __nv_fp8_e4m3(fminf(fmaxf(values[i] / scale, -448.0f), 448.0f));
    }
}

template<typename T>
void invokeAddBiasGeluQuantFp8(const T*     input,
                               const T*     bias,
                               void*        output,
                               uint32_t*    scales,
                               size_t       rows,
                               size_t       hidden_size,
                               size_t       scale_stride,
                               cudaStream_t stream) {
    constexpr size_t group_size = 128;
    if (rows == 0) {
        return;
    }
    const size_t groups_per_row = hidden_size / group_size;
    addBiasGeluQuantFp8Kernel<<<rows * groups_per_row, 32, 0, stream>>>(
        input, bias, static_cast<__nv_fp8_e4m3*>(output), scales, hidden_size, scale_stride);
}

template<typename T>
__global__ void addBiasGeluVector2(T* output, const T* bias, size_t pair_count, size_t hidden_pairs) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pair_count) {
        return;
    }
    if constexpr (std::is_same_v<T, half>) {
        half2  value_pair = reinterpret_cast<half2*>(output)[index];
        half2  bias_pair  = reinterpret_cast<const half2*>(bias)[index % hidden_pairs];
        float2 value      = __half22float2(value_pair);
        float2 bias_value = __half22float2(bias_pair);
        reinterpret_cast<half2*>(output)[index] =
            __floats2half2_rn(exactGelu(value.x + bias_value.x), exactGelu(value.y + bias_value.y));
    } else {
        __nv_bfloat162 value_pair = reinterpret_cast<__nv_bfloat162*>(output)[index];
        __nv_bfloat162 bias_pair  = reinterpret_cast<const __nv_bfloat162*>(bias)[index % hidden_pairs];
        float2         value      = __bfloat1622float2(value_pair);
        float2         bias_value = __bfloat1622float2(bias_pair);
        reinterpret_cast<__nv_bfloat162*>(output)[index] =
            __floats2bfloat162_rn(exactGelu(value.x + bias_value.x), exactGelu(value.y + bias_value.y));
    }
}
#endif

template<typename T>
void invokeAddBiasGelu(T* output, const T* bias, size_t numel, size_t hidden_size, cudaStream_t stream) {
    if (numel == 0) {
        return;
    }
    constexpr int block_size = 256;
#if USING_CUDA
    constexpr uintptr_t vector_alignment = 2 * sizeof(T);
    const bool          pointers_aligned = (reinterpret_cast<uintptr_t>(output) % vector_alignment == 0)
                                  && (reinterpret_cast<uintptr_t>(bias) % vector_alignment == 0);
    if (numel % 2 == 0 && hidden_size % 2 == 0 && pointers_aligned) {
        const size_t pair_count = numel / 2;
        const int    grid_size  = static_cast<int>((pair_count + block_size - 1) / block_size);
        addBiasGeluVector2<<<grid_size, block_size, 0, stream>>>(output, bias, pair_count, hidden_size / 2);
        return;
    }
#endif
    const int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    addBiasGelu<<<grid_size, block_size, 0, stream>>>(output, bias, numel, hidden_size);
}

template<typename T>
__global__ void
addBiasSoftMax(T* logits, const T* bias, const int* end_ids, const bool* finished, const int n_padded, const int n) {
    int  bid    = blockIdx.x;
    bool finish = (finished != nullptr) ? finished[bid] : false;
    int  offset = bid * n_padded;

    float            max_val   = -1 * FLT_MAX;
    const bool       IS_FP16   = std::is_same<T, half>::value;
    const T          MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        if (tid < n) {
            if (finish) {
                logits[offset + tid] = (tid == end_ids[bid]) ? static_cast<T>(MAX_T_VAL) : static_cast<T>(-MAX_T_VAL);
            } else {
                T bias_val = (bias != nullptr) ? bias[tid] : static_cast<T>(0.0f);
                logits[offset + tid] += bias_val;
            }
        } else {
            logits[offset + tid] = static_cast<T>(-MAX_T_VAL);
        }
        max_val = max(max_val, (float)logits[offset + tid]);
    }

    max_val = blockReduceMax<float>((float)max_val);
    if (threadIdx.x == 0) {
        s_max_val = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;
    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
        sum_val += (float)logits[offset + tid];
    }

    sum_val = blockReduceSum<float>(sum_val);
    if (threadIdx.x == 0) {
        s_sum_val = sum_val;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        logits[offset + tid] = ((float)logits[offset + tid] / (s_sum_val + 1e-6f));
    }
}

template<typename T>
void invokeAddBiasSoftMax(T*           logits,
                          const T*     bias,
                          const int*   end_ids,
                          const bool*  finished,
                          const int    m,
                          const int    n_padded,
                          const int    n,
                          cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, bias, end_ids, finished, n_padded, n);
}

template void invokeAddBiasSoftMax(float*       logits,
                                   const float* bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    m,
                                   const int    n_padded,
                                   const int    n,
                                   cudaStream_t stream);

template void invokeAddBiasGelu(half* output, const half* bias, size_t numel, size_t hidden_size, cudaStream_t stream);

template void invokeAddBiasGelu(
    __nv_bfloat16* output, const __nv_bfloat16* bias, size_t numel, size_t hidden_size, cudaStream_t stream);

#if USING_CUDA
template void invokeAddBiasGeluQuantFp8(const half*  input,
                                        const half*  bias,
                                        void*        output,
                                        uint32_t*    scales,
                                        size_t       rows,
                                        size_t       hidden_size,
                                        size_t       scale_stride,
                                        cudaStream_t stream);
template void invokeAddBiasGeluQuantFp8(const __nv_bfloat16* input,
                                        const __nv_bfloat16* bias,
                                        void*                output,
                                        uint32_t*            scales,
                                        size_t               rows,
                                        size_t               hidden_size,
                                        size_t               scale_stride,
                                        cudaStream_t         stream);
#endif

template void invokeAddBiasSoftMax(half*        logits,
                                   const half*  bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    m,
                                   const int    n_padded,
                                   const int    n,
                                   cudaStream_t stream);

template void invokeAddBiasSoftMax(__nv_bfloat16*       logits,
                                   const __nv_bfloat16* bias,
                                   const int*           end_ids,
                                   const bool*          finished,
                                   const int            m,
                                   const int            n_padded,
                                   const int            n,
                                   cudaStream_t         stream);

}  // namespace rtp_llm
