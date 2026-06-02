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
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

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
