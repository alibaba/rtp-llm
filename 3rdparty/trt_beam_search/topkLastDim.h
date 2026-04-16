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

#include "common.h"
#include <optional>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
size_t invokeComputeTopkLastDimWorkspaceSize(
    runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k, bool is_largest);

/* 
 * @brief Compute topk in the last dimension.
 *
 * if `mask_val` is specified, when the value at top-k boundary is equal to `mask_val`, the indice sorting for 
 * result stablization is skipped for the `mask_val`, which avoids the performance penalty of sorting useless masked results.
 */
template <typename T>
void invokeTopkLastDim(runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k, bool is_largest, 
    std::optional<T> mask_val, void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_ind,
    void* workspace, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
