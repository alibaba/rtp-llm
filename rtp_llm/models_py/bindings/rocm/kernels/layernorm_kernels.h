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

#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include <assert.h>
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#endif

namespace rtp_llm {

template<typename T>
struct LayerNormWeight {
    const T* gamma = nullptr;
    const T* beta  = nullptr;
};

template<typename T, typename QUANT_OUT_T = int8_t>
void invokeGeneralAddBiasResidualLayerNorm(T*           out,
                                           T*           norm_output,
                                           const T*     input,
                                           const T*     bias,
                                           const T*     residual,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float  eps,
                                           const int    tokens,
                                           const int    hidden_dim,
                                           cudaStream_t stream               = 0,
                                           bool         use_diff_of_squares  = true,
                                           const float* scale                = nullptr,
                                           float*       dynamic_scale        = nullptr,
                                           QUANT_OUT_T* out_quant            = nullptr,
                                           bool         return_normed_output = false);

template<typename T, typename QUANT_OUT_T = int8_t>
void invokeGeneralLayerNorm(T*           out,
                            T*           normed_output,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  eps,
                            const int    tokens,
                            const int    hidden_dim,
                            cudaStream_t stream               = 0,
                            bool         use_diff_of_squares  = true,
                            const float* scale                = nullptr,
                            float*       dynamic_scale        = nullptr,
                            QUANT_OUT_T* out_quant            = nullptr,
                            bool         return_normed_output = false);

template<typename T, bool IS_BIAS>
void invokeQkLayerNorm(T* __restrict qkv,
                       const T* __restrict gamma,
                       const float  layernorm_eps,
                       const int    tokens,
                       const int    head_num,
                       const int    head_num_kv,
                       const int    size_per_head,
                       cudaStream_t stream = 0);

template<typename T>
void invokeLayerNormWithStride(T* __restrict output,
                               const int out_stride,
                               const T* __restrict input,
                               const int in_stride,
                               const T* __restrict gamma,
                               const T* __restrict beta,
                               const float  layernorm_eps,
                               const int    tokens,
                               const int    hidden_size,
                               const int    norm_size,
                               cudaStream_t stream);

}  // namespace rtp_llm
