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
#include "trt_plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyQuantMatmulPlugin;

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId) {
    init(type, weightTypeId);
}

void WeightOnlyQuantMatmulPlugin::init(nvinfer1::DataType type, WeightTypeId weightTypeId)
{
    if (m_weightOnlyGemmRunner != nullptr && mType == type && mWeightTypeId == weightTypeId) {
        return;
    }
    mArch = rtp_llm::get_sm();
    mType = type;
    mWeightTypeId = weightTypeId;
    
    if (mWeightTypeId == WeightTypeId::INT8) {
        if (mType == nvinfer1::DataType::kHALF) {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int8PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int8PerChannel;
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16) {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int8PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int8PerChannel;
        }
#endif
        else {
            TLLM_CHECK(false);
        }
    } else {
        TLLM_CHECK(false);
    }
}

size_t WeightOnlyQuantMatmulPlugin::getWorkspaceSize(const int m, const int n, const int k) {
    m_workspaceMaxSize = m_weightOnlyGemmRunner->getWorkspaceSize(m, n, k);
    return m_workspaceMaxSize;
}

int WeightOnlyQuantMatmulPlugin::enqueue(const void*  inputs,
                                         const void*  weights,
                                         const void*  scales,
                                         void*        outputs,
                                         void*        workspace,
                                         const int    m,
                                         const int    n,
                                         const int    k,
                                         cudaStream_t stream) {
    const bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;

    if (use_cuda_kernel) {
        tensorrt_llm::kernels::weight_only::Params params(reinterpret_cast<const void*>(inputs),
                                                          nullptr,
                                                          reinterpret_cast<const uint8_t*>(weights),
                                                          reinterpret_cast<const void*>(scales),
                                                          nullptr,
                                                          nullptr,
                                                          reinterpret_cast<void*>(outputs),
                                                          1.f,
                                                          m,
                                                          n,
                                                          k,
                                                          0,
                                                          mCudaKernelType);
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    } else {
        const int  ws_size    = m_weightOnlyGemmRunner->getWorkspaceSize(m, n, k);
        const auto bestTactic = m_weightOnlyGemmRunner->getChosenConfig(inputs,
                                                                        weights,
                                                                        scales,
                                                                        nullptr,
                                                                        nullptr,
                                                                        outputs,
                                                                        m,
                                                                        n,
                                                                        k,
                                                                        k,
                                                                        reinterpret_cast<char*>(workspace),
                                                                        ws_size,
                                                                        stream);

        TLLM_CHECK_WITH_INFO(
            &bestTactic,
            "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
            "engine.)");

        m_weightOnlyGemmRunner->gemm(
            inputs, weights, scales, outputs, m, n, k, bestTactic, reinterpret_cast<char*>(workspace), ws_size, stream);
    }

    return 0;
}
