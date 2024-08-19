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
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{
using SqGemmRunnerPtr = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>;

class SmoothQuantGemmPlugin
{
public:
    SmoothQuantGemmPlugin() = default;

    SmoothQuantGemmPlugin(tensorrt_llm::common::QuantMode quantMode, nvinfer1::DataType type);

    ~SmoothQuantGemmPlugin() = default;

    size_t getWorkspaceSize(const int m, const int n, const int k) noexcept;
    int enqueue(const void* A, const void* B, const float* alphaCol, const float* alphaRow, void* C, char* workspace,
        void* bias, tkc::CutlassActivationType activation, const int m, const int n, const int k, cudaStream_t stream) noexcept;

    void init(tensorrt_llm::common::QuantMode quantMode,
              nvinfer1::DataType type);

    bool addBiasActivationEpilogueSupported(tkc::CutlassActivationType activation) const;
private:

    void configGemm();

private:
    const std::string               mLayerName;
    SqGemmRunnerPtr                 m_sqGemmRunner;
    tensorrt_llm::common::QuantMode mQuantMode;
    size_t                          m_workspaceMaxSize;
    nvinfer1::DataType              mType;
};

} // namespace tensorrt_llm::plugins
