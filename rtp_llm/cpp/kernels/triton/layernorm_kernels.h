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

namespace rtp_llm {

template<typename T, typename QUANT_OUT_T = int8_t, bool HAS_BIAS>
void invokeTritonLayerNorm(T*           out,
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

}  // namespace rtp_llm
