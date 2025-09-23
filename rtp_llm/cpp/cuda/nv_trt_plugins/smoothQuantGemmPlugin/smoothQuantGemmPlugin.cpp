/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "trt_plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::SmoothQuantGemmPlugin;

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(QuantMode quantMode, nvinfer1::DataType type): mQuantMode(quantMode) {
    init(quantMode, type);
}

void SmoothQuantGemmPlugin::init(tensorrt_llm::common::QuantMode quantMode, nvinfer1::DataType type) {
    mQuantMode = quantMode;
    mType      = type;
    if (mType == nvinfer1::DataType::kHALF) {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    } else if (mType == nvinfer1::DataType::kFLOAT) {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    } else if (mType == nvinfer1::DataType::kINT32) {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    } else {
        // TODO: add bf16 support
        TLLM_THROW("Support for bf16 is missing");
    }
}

bool SmoothQuantGemmPlugin::addBiasActivationEpilogueSupported(rtp_llm::ActivationType activation) const {
    CutlassActivationType cutlass_activation_type = ActivationToCutlassType(activation);
    return CutlassInt8GemmRunnerInterface::addBiasActivationEpilogueSupported(cutlass_activation_type);
}

CutlassActivationType SmoothQuantGemmPlugin::ActivationToCutlassType(rtp_llm::ActivationType act_type) const {
    switch (act_type) {
        case rtp_llm::ActivationType::Identity:
            return CutlassActivationType::IDENTITY;
        case rtp_llm::ActivationType::Geglu:
        case rtp_llm::ActivationType::Gelu:
            return CutlassActivationType::GELU_FAST;
        case rtp_llm::ActivationType::GeluNoneApproximate:
        case rtp_llm::ActivationType::GeGluNoneApproximate:
            return CutlassActivationType::GELU;
        case rtp_llm::ActivationType::Relu:
            return CutlassActivationType::RELU;
        case rtp_llm::ActivationType::Sigmoid:
            return CutlassActivationType::SIGMOID;
        case rtp_llm::ActivationType::Silu:
        case rtp_llm::ActivationType::Swiglu:
            return CutlassActivationType::SILU;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "ERROR ACTIVATION TYPE");
    }
    return CutlassActivationType::INVALID;
}

size_t SmoothQuantGemmPlugin::getWorkspaceSize(const int m, const int n, const int k) {
    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);
    return m_workspaceMaxSize;
}

int SmoothQuantGemmPlugin::enqueue(const void*             A,
                                   const void*             B,
                                   const float*            alphaCol,
                                   const float*            alphaRow,
                                   void*                   C,
                                   char*                   workspace,
                                   void*                   bias,
                                   rtp_llm::ActivationType activation,
                                   const int               m,
                                   const int               n,
                                   const int               k,
                                   cudaStream_t            stream) {
    // inputs
    //     mat1           [M(*), K]
    //     mat2           [N, K]
    //     scale_channels [1, N] if has_per_channel_scaling else [1, 1]
    //     scale_tokens   [M, 1] if has_per_token_scaling else [1, 1]
    // outputs
    //     mat [M(*), N]
    const int wsSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);

    const auto bestTactic =
        m_sqGemmRunner->getChosenConfig(A, B, mQuantMode, alphaCol, alphaRow, C, m, n, k, workspace, wsSize, stream);

    TLLM_CHECK_WITH_INFO(
        &bestTactic,
        "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    CutlassActivationType cutlass_activation_type = ActivationToCutlassType(activation);
    m_sqGemmRunner->gemm(A,
                         B,
                         mQuantMode,
                         alphaCol,
                         alphaRow,
                         C,
                         bias,
                         cutlass_activation_type,
                         m,
                         n,
                         k,
                         bestTactic,
                         workspace,
                         wsSize,
                         stream);

    return 0;
}