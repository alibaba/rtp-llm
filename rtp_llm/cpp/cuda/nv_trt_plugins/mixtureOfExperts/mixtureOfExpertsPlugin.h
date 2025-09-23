/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
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
#ifndef TRT_MIXTURE_OF_EXPERTS_PLUGIN_H
#define TRT_MIXTURE_OF_EXPERTS_PLUGIN_H

#include "rtp_llm/cpp/model_utils/quantization.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_kernels.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_fp8_kernels.h"
#include "rtp_llm/cpp/cuda/trt_utils.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "cutlass/numeric_types.h"

#include <cassert>
// #include <mpi.h>
#include <set>
#include <string>
#include <vector>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::plugins {

class MixtureOfExpertsPlugin {
public:
    // using MOEParallelismMode = tensorrt_llm::kernels::MOEParallelismMode;
    using MOEExpertScaleNormalizationMode = tensorrt_llm::kernels::MOEExpertScaleNormalizationMode;

    MixtureOfExpertsPlugin() = default;
    MixtureOfExpertsPlugin(int                             number_of_experts,
                           int                             top_k,
                           bool                            normalize_expert_scale,
                           int                             expert_hidden_size,
                           int                             expert_inter_size,
                           rtp_llm::ActivationType         activation_type,
                           nvinfer1::DataType              type,
                           nvinfer1::DataType              weight_type,
                           bool                            has_zeros,
                           int                             group_size,
                           MOEExpertScaleNormalizationMode normalization_mode);

    void init(int                             number_of_experts,
              int                             top_k,
              bool                            normalize_expert_scale,
              int                             expert_hidden_size,
              int                             expert_inter_size,
              rtp_llm::ActivationType         activation_type,
              nvinfer1::DataType              type,
              nvinfer1::DataType              weight_type,
              bool                            has_zeros,
              int                             group_size,
              MOEExpertScaleNormalizationMode normalization_mode,
              int                             tp_size = 1,
              int                             tp_rank = 0);

    ~MixtureOfExpertsPlugin() = default;

    size_t getWorkspaceSize(int num_tokens);
    int    enqueue(void const*  input,
                   float const* moe_gates,
                   float const* moe_gates_with_bias,
                   void const*  fc1_expert_weight,
                   void const*  fc1_quant_scale,
                   void const*  fc1_quant_zero,
                   void const*  fc1_expert_bias,
                   void const*  fc2_expert_weight,
                   void const*  fc2_quant_scale,
                   void const*  fc2_quant_zero,
                   void const*  fc2_expert_bias,
                   int const    num_rows,
                   void*        workspace,
                   void*        final_output,
                   void*        fc2_result,
                   bool const*  finished,
                   void*        expert_scale,
                   int*         src_row_to_dst_row,
                   int*         export_for_src_row,
                   cudaStream_t stream);
    template<typename TOPK_T>
    static void selectExpertsForTokens(float const*                    input,
                                       float const*                    input_with_bias,
                                       float*                          output,
                                       float*                          mixer_temp_output,
                                       float*                          softmax_temp_output,
                                       TOPK_T*                         indices,
                                       int*                            source_row,
                                       int64_t const                   num_rows,
                                       int const                       num_experts,
                                       int const                       k,
                                       int const                       start_expert,
                                       int const                       end_expert,
                                       float                           mixer_epsilon,
                                       MOEExpertScaleNormalizationMode norm_mode,
                                       cudaStream_t                    stream);

private:
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mMOERunner{};
    int                                                   mNumExperts{};
    int                                                   mK{};
    int                                                   mExpertHiddenSize{};
    int                                                   mExpertInterSize{};
    bool                                                  mNormalizeExpertScale = false;
    rtp_llm::ActivationType                               mActivationType;
    nvinfer1::DataType                                    mType{};
    nvinfer1::DataType                                    mWeightType{};
    bool                                                  mHasZeros;
    int                                                   mGroupSize;
    // tensorrt_llm::common::QuantMode mQuantMode;
    int mEPSize = 1;
    int mEPRank = 0;
    // MOEParallelismMode mParallelismMode{};
    MOEExpertScaleNormalizationMode mNormalizationMode{};

    kernels::MOEParallelismConfig getParallelismConfig() const;
};

}  // namespace tensorrt_llm::plugins

#endif  // TRT_MIXTURE_OF_EXPERTS_PLUGIN_H
