/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/cpp/cuda/trt_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/weight_only_quant_op.h"
#include "rtp_llm/cpp/utils/utils.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::plugins;
using namespace tensorrt_llm::kernels;
// using tensorrt_llm::common::QuantMode;
using tensorrt_llm::plugins::MixtureOfExpertsPlugin;

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(int                             number_of_experts,
                                               int                             top_k,
                                               bool                            normalize_expert_scale,
                                               int                             expert_hidden_size,
                                               int                             expert_inter_size,
                                               rtp_llm::ActivationType         activation_type,
                                               nvinfer1::DataType              type,
                                               nvinfer1::DataType              weight_type,
                                               bool                            has_zeros,
                                               int                             group_size,
                                               MOEExpertScaleNormalizationMode normalization_mode) {
    init(number_of_experts,
         top_k,
         normalize_expert_scale,
         expert_hidden_size,
         expert_inter_size,
         activation_type,
         type,
         weight_type,
         has_zeros,
         group_size,
         normalization_mode);
}

void MixtureOfExpertsPlugin::init(int                             number_of_experts,
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
                                  int                             ep_size,
                                  int                             ep_rank) {
    mNumExperts           = number_of_experts;
    mK                    = top_k;
    mExpertHiddenSize     = expert_hidden_size;
    mExpertInterSize      = expert_inter_size;
    mNormalizeExpertScale = normalize_expert_scale;
    mActivationType       = activation_type;
    mType                 = type;
    mWeightType           = weight_type;
    mHasZeros             = has_zeros;
    mGroupSize            = group_size;
    mNormalizationMode    = normalization_mode;
    mEPSize               = ep_size;
    mEPRank               = ep_rank;
    if (mWeightType == DataType::kINT8 || mWeightType == DataType::kINT4) {
        FT_SWITCH_T(mType == nvinfer1::DataType::kHALF, T, half, __nv_bfloat16, [&] {
            FT_SWITCH_V(mHasZeros,
                        Q,
                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS,
                        cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY,
                        [&] {
                            FT_SWITCH_T(mWeightType == DataType::kINT4, WT, cutlass::uint4b_t, uint8_t, [&] {
                                mMOERunner = std::make_shared<CutlassMoeFCRunner<T, WT, Q>>();
                            });
                        });
        });
    } else if (mWeightType == DataType::kHALF) {
        mMOERunner = std::make_shared<CutlassMoeFCRunner<half, half, cutlass::WeightOnlyQuantOp::UNDEFINED>>();
    }
#ifdef ENABLE_BF16
    else if (mWeightType == DataType::kBF16) {
        mMOERunner =
            std::make_shared<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, cutlass::WeightOnlyQuantOp::UNDEFINED>>();
    }
#endif
    else {
        TLLM_THROW("Could not construct the mixture of experts plugin with the requested input combination");
    }

    // mMOERunner->setTactic(mMOERunner->getTactics()[0], mMOERunner->getTactics()[0]);
}

size_t MixtureOfExpertsPlugin::getWorkspaceSize(int num_tokens) {

    size_t moe_workspace_size = mMOERunner->getWorkspaceSize(num_tokens,
                                                             mExpertHiddenSize,
                                                             mExpertInterSize,
                                                             mNumExperts,
                                                             mK,
                                                             mActivationType,
                                                             mNormalizationMode,
                                                             getParallelismConfig(),
                                                             false);

    return moe_workspace_size;
}

MOEParallelismConfig MixtureOfExpertsPlugin::getParallelismConfig() const {
    // 默认MOE都走EP
    return MOEParallelismConfig(1, 0, mEPSize, mEPRank);
    // return {};
    // switch (mParallelismMode)
    // {
    // case kernels::MOEParallelismMode::NONE: return {};
    // case kernels::MOEParallelismMode::EXPERT_PARALLELISM:
    // return MOEParallelismConfig(1, 0, mTPSize, mTPRank);
    // case kernels::MOEParallelismMode::TENSOR_PARALLELISM:
    //     return MOEParallelismConfig::TensorParallelism(mTPSize, mTPRank);
    // }
    // assert(false);
    // return {};
}

int MixtureOfExpertsPlugin::enqueue(void const*  input,
                                    float const* moe_gates,
                                    float const* moe_gates_with_bias,
                                    void const*  fc1_expert_weight,
                                    void const*  fc1_quant_scale,
                                    void const*  fc1_quant_zeros,
                                    void const*  fc1_expert_bias,
                                    void const*  fc2_expert_weight,
                                    void const*  fc2_quant_scale,
                                    void const*  fc2_quant_zeros,
                                    void const*  fc2_expert_bias,
                                    int const    num_rows,
                                    void*        workspace,
                                    void*        final_output,
                                    void*        fc2_result,
                                    bool const*  finished,
                                    void*        expert_scale,
                                    int*         src_row_to_dst_row,
                                    int*         export_for_src_row,
                                    cudaStream_t stream) {
    int const            num_not_finished   = num_rows;  // TODO Take this as an input
    MOEParallelismConfig parallelism_config = getParallelismConfig();
    LoraParams           lora_params;

    mMOERunner->setTactic(num_rows, mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, mActivationType, stream);
    mMOERunner->runMoe(input,  // const void*
                       moe_gates,
                       moe_gates_with_bias,
                       fc1_expert_weight,
                       fc1_expert_bias,
                       mActivationType,
                       fc2_expert_weight,
                       fc2_expert_bias,
                       QuantParams::Int(fc1_quant_scale, fc1_quant_zeros, fc2_quant_scale, fc2_quant_zeros, mGroupSize),
                       num_rows,
                       mExpertHiddenSize,
                       mExpertInterSize,
                       mNumExperts,
                       mK,
                       // mNormalizeExpertScale,
                       reinterpret_cast<char*>(workspace),
                       // Outputs
                       final_output,
                       finished,            // bool
                       num_not_finished,    // const int
                       expert_scale,        // void*
                       src_row_to_dst_row,  // int*
                       export_for_src_row,  // int*
                       0,
                       parallelism_config,
                       mNormalizationMode,
                       false,
                       lora_params,
                       stream);

    return 0;
}

template<typename TOPK_T>
void MixtureOfExpertsPlugin::selectExpertsForTokens(float const*                    input,
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
                                                    cudaStream_t                    stream) {
    invokeSelectExpertsForTokens<TOPK_T>(input,
                                         input_with_bias,
                                         output,
                                         mixer_temp_output,
                                         softmax_temp_output,
                                         indices,
                                         source_row,
                                         num_rows,
                                         num_experts,
                                         k,
                                         start_expert,
                                         end_expert,
                                         mixer_epsilon,
                                         norm_mode,
                                         stream);
}

template void MixtureOfExpertsPlugin::selectExpertsForTokens(float const*                    input,
                                                             float const*                    input_with_bias,
                                                             float*                          output,
                                                             float*                          mixer_temp_output,
                                                             float*                          softmax_temp_output,
                                                             int*                            indices,
                                                             int*                            source_row,
                                                             int64_t const                   num_rows,
                                                             int const                       num_experts,
                                                             int const                       k,
                                                             int const                       start_expert,
                                                             int const                       end_expert,
                                                             float                           mixer_epsilon,
                                                             MOEExpertScaleNormalizationMode norm_mode,
                                                             cudaStream_t                    stream);

template void MixtureOfExpertsPlugin::selectExpertsForTokens(float const*                    input,
                                                             float const*                    input_with_bias,
                                                             float*                          output,
                                                             float*                          mixer_temp_output,
                                                             float*                          softmax_temp_output,
                                                             int64_t*                        indices,
                                                             int*                            source_row,
                                                             int64_t const                   num_rows,
                                                             int const                       num_experts,
                                                             int const                       k,
                                                             int const                       start_expert,
                                                             int const                       end_expert,
                                                             float                           mixer_epsilon,
                                                             MOEExpertScaleNormalizationMode norm_mode,
                                                             cudaStream_t                    stream);
