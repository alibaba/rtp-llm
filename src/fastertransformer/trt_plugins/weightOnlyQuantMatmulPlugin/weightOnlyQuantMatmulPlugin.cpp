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
#include "src/fastertransformer/trt_plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyQuantMatmulPlugin;

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId)
{
    init(type, weightTypeId);
}

void WeightOnlyQuantMatmulPlugin::init(nvinfer1::DataType type, WeightTypeId weightTypeId)
{
    mType = type;
    mWeightTypeId = weightTypeId;
    if (mWeightTypeId == WeightTypeId::INT8)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#endif
        else
        {
            TLLM_CHECK(false);
        }

        mCudaKernelEnabled 
            = fastertransformer::kernels::isWeightOnlyBatchedGemvEnabled(fastertransformer::kernels::WeightOnlyQuantType::Int8b);
    }
    else if (mWeightTypeId == WeightTypeId::INT4)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#endif
        else
        {
            TLLM_CHECK(false);
        }
        mCudaKernelEnabled
            = fastertransformer::kernels::isWeightOnlyBatchedGemvEnabled(fastertransformer::kernels::WeightOnlyQuantType::Int4b);
    }
    else
    {
        TLLM_CHECK(false);
    }

}

size_t WeightOnlyQuantMatmulPlugin::getWorkspaceSize(const int m, const int n, const int k) noexcept
{
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
                                         cudaStream_t stream) noexcept
{
    const bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyQuantMatmul configuration");
#endif

    int real_n;

    fastertransformer::kernels::WeightOnlyActivationType weight_only_act_type;

    if (mType == nvinfer1::DataType::kHALF)
    {
        weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::FP16;
    }
#if defined(ENABLE_BF16)
    else if (mType == nvinfer1::DataType::kBF16)
    {
        weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::BF16;
    }
# endif
    else {
        FT_LOG_ERROR("weight only batched gemv only support half and bf16");
    }

    fastertransformer::kernels::WeightOnlyQuantType weight_only_quant_type;
    if (mWeightTypeId == WeightTypeId::INT8)
    {
        weight_only_quant_type = fastertransformer::kernels::WeightOnlyQuantType::Int8b;
        real_n = n;
    }
    else if (mWeightTypeId == WeightTypeId::INT4)
    {
        weight_only_quant_type = fastertransformer::kernels::WeightOnlyQuantType::Int4b;
        real_n = n * INT8_INT4_RATIO;
    }

    if (use_cuda_kernel) {
        // Use CUDA kernels for small batch size
        // The CUDA kernel is designed for ColumnMajorTileInterleave weight layout used in fpAIntB cutlass
        // kernel when sm >= 75 and the preprocessing of cutlass on sm70 does not interleave the weights.
        fastertransformer::kernels::WeightOnlyParams weight_only_batched_gemv_params{
            reinterpret_cast<const uint8_t*>(weights),
            reinterpret_cast<const void*>(scales),
            nullptr,
            reinterpret_cast<const void*>(inputs),
            nullptr,
            reinterpret_cast<void*>(outputs),
            m,
            n,
            k,
            0,
            fastertransformer::kernels::WeightOnlyQuantType::Int8b,
            fastertransformer::kernels::WeightOnlyType::PerChannel,
            fastertransformer::kernels::WeightOnlyActivationFunctionType::Identity,
            weight_only_act_type};
        fastertransformer::kernels::weight_only_batched_gemv_launcher(weight_only_batched_gemv_params, stream);
    }
    else {
        const int  ws_size    = m_weightOnlyGemmRunner->getWorkspaceSize(m, real_n, k);
        const auto bestTactic = m_weightOnlyGemmRunner->getChosenConfig(inputs,
                                                                        weights,
                                                                        scales,
                                                                        nullptr,
                                                                        nullptr,
                                                                        outputs,
                                                                        m,
                                                                        real_n,
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

        m_weightOnlyGemmRunner->gemm(inputs,
                                     weights,
                                     scales,
                                     outputs,
                                     m,
                                     real_n,
                                     k,
                                     bestTactic,
                                     reinterpret_cast<char*>(workspace),
                                     ws_size,
                                     stream);
    }

    return 0;
}
