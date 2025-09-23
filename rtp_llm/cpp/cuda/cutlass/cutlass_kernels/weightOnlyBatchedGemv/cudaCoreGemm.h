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
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/trt_utils.h"
#include "rtp_llm/cpp/model_utils/quantization.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/cutlass_type_conversion.h"

// #include <NvInferRuntime.h>

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
namespace cuda_core_gemm
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;

struct Params
{
    void const* act;
    void const* weight;
    float alpha;
    void* output;
    SizeType32 m, n, k;
    tensorrt_llm::common::QuantMode quantMode;
    nvinfer1::DataType inputType;
    nvinfer1::DataType outputType;

    Params(void const* _act, void const* _weight, float _alpha, void* _output, SizeType32 _m, SizeType32 _n,
        SizeType32 _k, tensorrt_llm::common::QuantMode _quant_mode, nvinfer1::DataType _inputType,
        nvinfer1::DataType _outputType)
        : act(_act)
        , weight(_weight)
        , alpha(_alpha)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , quantMode(_quant_mode)
        , inputType(_inputType)
        , outputType(_outputType)
    {
    }
};

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream);
} // namespace cuda_core_gemm
} // namespace kernels
} // namespace tensorrt_llm
