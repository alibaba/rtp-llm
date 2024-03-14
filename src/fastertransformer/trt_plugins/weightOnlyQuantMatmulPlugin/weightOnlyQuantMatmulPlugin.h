/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "src/fastertransformer/utils/quantization.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/trt_utils.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/numeric_types.h"

namespace tensorrt_llm::plugins
{
enum class WeightTypeId
{
    INT8 = 1,
    INT4 = 2,
};

constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId)
{
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantMatmulPlugin
{
public:
    // using PluginProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = delete;

    WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId);

    ~WeightOnlyQuantMatmulPlugin() = default;

    size_t getWorkspaceSize(const int m, const int n, const int k) noexcept;
    int    enqueue(const void*  inputs,
                   const void*  weights,
                   const void*  scales,
                   void*        outputs,
                   void*        workspace,
                   const int    m,
                   const int    n,
                   const int    k,
                   cudaStream_t stream) noexcept;

    int  initialize() noexcept;

private:
    void init(nvinfer1::DataType type, WeightTypeId weightTypeId);

    void configGemm();

private:
    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    WeightTypeId mWeightTypeId;
    bool mCudaKernelEnabled;

    static constexpr int SMALL_M_FAST_PATH = 4;
};

} // namespace tensorrt_llm::plugins
