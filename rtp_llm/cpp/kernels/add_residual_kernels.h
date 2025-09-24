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

#pragma once

#include <assert.h>
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#include <stdint.h>

namespace rtp_llm {

template<typename T>
void invokeAddBiasResidual(T*           output,
                           const T*     input,
                           const T*     residual1,
                           const T*     residual2,
                           const T*     bias,
                           const float* scale_inter,
                           const float* scale_out,
                           const int    m,
                           const int    n,
                           cudaStream_t stream);

template<typename T>
void invokeAlphaAddBiasResidual(T*           output,
                                const T*     input,
                                const T*     residual,
                                const T*     bias,
                                const T      alpha,
                                const int    m,
                                const int    n,
                                cudaStream_t stream);

template<typename T>
void invokeT5AddResidual(T* output, const T* input, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeT5AddResidual(T* output, const T* input_1, const T* input_2, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeT5AddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeT5AddBiasResidual(
    T* output, const T* input_1, const T* input_2, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasAttentionFfnResidual(T*           block_output,
                                       const T*     ffn_output,
                                       const T*     attn_output,
                                       const T*     block_input,
                                       const T*     bias,
                                       const int    m,
                                       const int    n,
                                       const int    block_input_tp_split,
                                       cudaStream_t stream);

template<typename T>
void invokeAddBiasAttentionFfnResidual(T*           block_output,
                                       const T*     ffn_output,
                                       const T*     attn_output,
                                       const T*     block_input,
                                       const T*     bias,
                                       const int    m,
                                       const int    n,
                                       cudaStream_t stream) {
    invokeAddBiasAttentionFfnResidual(block_output, ffn_output, attn_output, block_input, bias, m, n, 1, stream);
}

template<typename T>
void invokeAddBiasResidualCol32(T*            output,
                                const int8_t* input1,
                                const T*      input2,
                                const T*      bias,
                                int           m,
                                int           n,
                                cudaStream_t  stream,
                                const float*  input1_deQFactor_ptr);

template<typename T>
void invokeAddBiasResidualCol32(T*             output,
                                const int32_t* input1,
                                const T*       input2,
                                const T*       bias,
                                int            m,
                                int            n,
                                cudaStream_t   stream,
                                const float*   weight_amax,
                                const float*   input1_amax_ptr,
                                const int      scale_is_vector = 0);

}  // namespace rtp_llm
