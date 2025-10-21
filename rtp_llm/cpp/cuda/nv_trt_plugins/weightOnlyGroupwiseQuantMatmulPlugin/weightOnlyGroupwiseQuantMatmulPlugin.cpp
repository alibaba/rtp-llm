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
#include "trt_plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "rtp_llm/cpp/utils/utils.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPlugin;

static constexpr int PRE_QUANT_SCALE = int(1) << 2;

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(nvinfer1::DataType type,
                                                                           bool               has_zeros,
                                                                           int                group_size,
                                                                           int                weight_bits) {
    init(type, has_zeros, group_size, weight_bits);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::init(nvinfer1::DataType type,
                                                bool               has_zeros,
                                                int                group_size,
                                                int                weight_bits) {
    if (m_weightOnlyGroupwiseGemmRunner != nullptr && mType == type && mGroupSize == group_size
        && mHasZeros == has_zeros && mWeightBits == weight_bits) {
        // 参数相同且已经初始化过，直接返回
        return;
    }
    mArch      = rtp_llm::get_sm();
    mType      = type;
    mGroupSize = group_size;
    mHasZeros  = has_zeros;
    FT_SWITCH_T(mType == nvinfer1::DataType::kHALF, T, half, __nv_bfloat16, [&] {
        FT_SWITCH_V(
            has_zeros,
            Q,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
            [&] {
                FT_SWITCH_T(weight_bits == 4, WT, cutlass::uint4b_t, uint8_t, [&] {
                    m_weightOnlyGroupwiseGemmRunner =
                        std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WT, Q>>();
                });
            });
    });

    if (weight_bits == 4) {
        if (mType == nvinfer1::DataType::kHALF) {

            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise;
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16) {
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise;
        }
#endif
    } else {
        mCudaKernelEnabled = false;
    }
}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getWorkspaceSize(const int m, const int n, const int k) {
    m_workspaceMaxSize = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, n, k);
    return m_workspaceMaxSize;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::enqueue(const void*  inputs,
                                                  const void*  weights,
                                                  const void*  scales,
                                                  const void*  zeros,
                                                  const void*  biases,
                                                  void*        outputs,
                                                  void*        workspace,
                                                  const int    m,
                                                  const int    n,
                                                  const int    k,
                                                  cudaStream_t stream) {
    // inputs
    //   0 activations      [M, K]
    //   1 weights          [K, N/2]
    //   2 scales           [K // group_size, N]
    //   3 zeros            [K // group_size, N]
    //   4 biases           [M]
    // outputs
    //   mat                [M, N]

    bool        use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
    const void* act_ptr         = reinterpret_cast<const void*>(inputs);

#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
                         "No valid weightOnlyGropwiseQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyGropwiseQuantMatmul configuration");
#endif

    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    if (use_cuda_kernel) {
        tensorrt_llm::kernels::weight_only::Params params{inputs,
                                                          nullptr,
                                                          weights,
                                                          scales,
                                                          zeros,
                                                          biases,
                                                          outputs,
                                                          1.0f,
                                                          m,
                                                          n,
                                                          k,
                                                          mGroupSize,
                                                          mCudaKernelType,
                                                          false};
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    } else {
        // Use cutlass kernels for large batch size
        const int ws_size = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, n, k);

        const auto bestTactic = m_weightOnlyGroupwiseGemmRunner->getChosenConfig(inputs,
                                                                                 weights,
                                                                                 scales,
                                                                                 zeros,
                                                                                 biases,
                                                                                 outputs,
                                                                                 m,
                                                                                 n,
                                                                                 k,
                                                                                 mGroupSize,
                                                                                 reinterpret_cast<char*>(workspace),
                                                                                 ws_size,
                                                                                 stream);
        TLLM_CHECK_WITH_INFO(
            &bestTactic,
            "No valid weight only groupwise GEMM tactic(It is usually caused by the failure to execute all candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
            "engine.)");
        m_weightOnlyGroupwiseGemmRunner->gemm(act_ptr,
                                              weights,
                                              scales,
                                              zeros,
                                              biases,
                                              outputs,
                                              m,
                                              n,
                                              k,
                                              mGroupSize,
                                              bestTactic,
                                              reinterpret_cast<char*>(workspace),
                                              ws_size,
                                              stream);
    }

    return 0;
}
