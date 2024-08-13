/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>


namespace tensorrt_llm::runtime
{

#define FMT_DIM "%ld"

// typedefs
// Note that we use signed size types as recommended by TensorRT:
// https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md#signed-vs-unsigned-integers
using SizeType32 = std::int32_t;

// Token ID type
using TokenIdType = std::int32_t;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

} // namespace tensorrt_llm::runtime


namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tllm to Cutlass

template <typename T>
struct TllmToCutlassTypeAdapter
{
    using type = T;
};

template <>
struct TllmToCutlassTypeAdapter<half>
{
    using type = cutlass::half_t;
};

#if defined(ENABLE_BF16)
template <>
struct TllmToCutlassTypeAdapter<__nv_bfloat16>
{
    using type = cutlass::bfloat16_t;
};
#endif

#if defined(ENABLE_FP8)
template <>
struct TllmToCutlassTypeAdapter<__nv_fp8_e4m3>
{
    using type = cutlass::float_e4m3_t;
};

template <>
struct TllmToCutlassTypeAdapter<__nv_fp8_e5m2>
{
    using type = cutlass::float_e5m2_t;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cutlass to Tllm

template <typename T>
struct CutlassToTllmTypeAdapter
{
    using type = T;
};

template <>
struct CutlassToTllmTypeAdapter<cutlass::half_t>
{
    using type = half;
};

#if defined(ENABLE_BF16)
template <>
struct CutlassToTllmTypeAdapter<cutlass::bfloat16_t>
{
    using type = __nv_bfloat16;
};
#endif

#if defined(ENABLE_FP8)
template <>
struct CutlassToTllmTypeAdapter<cutlass::float_e4m3_t>
{
    using type = __nv_fp8_e4m3;
};

template <>
struct CutlassToTllmTypeAdapter<cutlass::float_e5m2_t>
{
    using type = __nv_fp8_e5m2;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
