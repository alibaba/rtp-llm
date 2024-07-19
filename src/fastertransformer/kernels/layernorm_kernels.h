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


#include "src/fastertransformer/utils/layernorm_types.h"
#include <assert.h>
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#endif

namespace fastertransformer {

template<typename T>
struct LayerNormWeight {
    const T* gamma = nullptr;
    const T* beta  = nullptr;
};

template <typename T>
void invokeGeneralAddBiasResidualLayerNorm(T* out, T* norm_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, const int tokens, const int hidden_dim, cudaStream_t stream = 0,
    bool use_diff_of_squares = true, const float* scale = nullptr, float* dynamic_scale = nullptr,
    int8_t* out_quant = nullptr, bool return_normed_output = false);

template <typename T>
void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, bool use_diff_of_squares = true, const float* scale = nullptr,
    float* dynamic_scale = nullptr, int8_t* out_quant = nullptr, bool return_normed_output = false);


template<typename T>
void invokeQkLayerNorm(T* __restrict qkv,
                       const T* __restrict gamma,
                       const float layernorm_eps,
                       const int tokens,
                       const int head_num,
                       const int head_num_kv,
                       const int size_per_head,
                       cudaStream_t stream = 0);

template<typename T>
void invokeLayerNormWithStride(T* __restrict data,
                               const T* __restrict gamma,
                               const T* __restrict beta,
                               const float  layernorm_eps,
                               const int    tokens,
                               const int    hidden_size,
                               const int    stride,
                               const int    offset,
                               cudaStream_t stream);

}  // namespace fastertransformer
