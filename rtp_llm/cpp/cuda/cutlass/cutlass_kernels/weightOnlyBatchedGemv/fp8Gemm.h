/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "rtp_llm/cpp/model_utils/quantization.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
namespace fp8_gemm
{
using SizeType32 = std::int32_t;

struct Params
{
    void const* act;
    void const* weight;
    float alpha;
    void* output;
    SizeType32 m, n, k;
    tensorrt_llm::common::QuantMode quantMode;

    Params(void const* _act, void const* _weight, float _alpha, void* _output, SizeType32 _m, SizeType32 _n,
        SizeType32 _k, tensorrt_llm::common::QuantMode _quant_mode)
        : act(_act)
        , weight(_weight)
        , alpha(_alpha)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , quantMode(_quant_mode)
    {
    }
};

template <typename InputType, typename OutputType>
void fp8GemmLauncher(Params& params, cudaStream_t stream);
} // namespace fp8_gemm
} // namespace kernels
} // namespace tensorrt_llm
