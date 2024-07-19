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

#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include <assert.h>
#if USING_ROCM
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace fastertransformer
{
template <typename T>
void invokeGeneralRmsNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, const float* scale = nullptr, float* dynamic_scale = nullptr,
    int8_t* out_quant = nullptr);

template <typename T>
void invokeAddBiasResidualRmsNorm(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, const int tokens, const int hidden_dim, cudaStream_t stream = 0,
    const float* scale = nullptr, float* dynamic_scale = nullptr, int8_t* out_quant = nullptr);
} // namespace fastertransformer
