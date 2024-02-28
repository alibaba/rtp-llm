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
#include "src/fastertransformer/trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/utils/trt_utils.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::plugins;
using namespace tensorrt_llm::kernels;
// using tensorrt_llm::common::QuantMode;
using tensorrt_llm::plugins::MixtureOfExpertsPlugin;


MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(int number_of_experts, int top_k, int expert_hidden_size,
    int expert_inter_size, fastertransformer::ActivationType activation_type, nvinfer1::DataType type,
    nvinfer1::DataType weight_type, MOEExpertScaleNormalizationMode normalization_mode)
    : mNumExperts(number_of_experts)
    , mK(top_k)
    , mExpertHiddenSize(expert_hidden_size)
    , mExpertInterSize(expert_inter_size)
    , mActivationType(activation_type)
    , mType(type)
    , mWeightType(weight_type)
    // , mQuantMode(quant_mode)
    // , mTPSize(tp_size)
    // , mTPRank(tp_rank)
    // , mParallelismMode(parallelism_mode)
    , mNormalizationMode(normalization_mode)
{
    init();
}

void MixtureOfExpertsPlugin::init()
{
    if (mType == DataType::kHALF && mWeightType == DataType::kHALF)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<half, half>>();
    }
    else if (mType == DataType::kFLOAT && mWeightType == DataType::kFLOAT)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<float, float>>();
    }
    else if (mType == DataType::kHALF && mWeightType == DataType::kINT8)
    {
        // if (mQuantMode.hasInt4Weights())
        // {
        //     mMOERunner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
        // }
        // else
        // {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<half, uint8_t>>();
        // }
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16 && mWeightType == DataType::kBF16)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
    else if (mType == DataType::kBF16 && mWeightType == DataType::kINT8)
    {
        // if (mQuantMode.hasInt4Weights())
        // {
        //     mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
        // }
        // else
        // {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t>>();
        // }
    }
#endif
    else
    {
        TLLM_THROW("Could not construct the mixture of experts plugin with the requested input combination");
    }
}

size_t MixtureOfExpertsPlugin::getWorkspaceSize(int num_tokens)
{

    size_t dtype_size = tensorrt_llm::common::getDTypeSize(mType);

    size_t moe_workspace_size = mMOERunner->getWorkspaceSize(
        num_tokens, mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, mActivationType, getParallelismConfig());

    return moe_workspace_size;
}

MOEParallelismConfig MixtureOfExpertsPlugin::getParallelismConfig() const
{
    return {};
    // switch (mParallelismMode)
    // {
    // case kernels::MOEParallelismMode::NONE: return {};
    // case kernels::MOEParallelismMode::EXPERT_PARALLELISM:
    //     return MOEParallelismConfig::ExpertParallelism(mTPSize, mTPRank);
    // case kernels::MOEParallelismMode::TENSOR_PARALLELISM:
    //     return MOEParallelismConfig::TensorParallelism(mTPSize, mTPRank);
    // }
    // assert(false);
    // return {};
}

int MixtureOfExpertsPlugin::enqueue(
    const void* input,
    const float* moe_gates,
    const void* fc1_expert_weight,
    const void* fc1_quant_scale,
    const void* fc1_expert_bias,
    const void* fc2_expert_weight,
    const void* fc2_quant_scale,
    const void* fc2_expert_bias,
    const void* fc3_expert_weight,
    const void* fc3_quant_scale,
    const void* fc3_expert_bias,
    const int num_rows,
    void* workspace,
    void* final_output,
    void* fc2_result,
    const bool* finished,
    void* expert_scale,
    int* src_row_to_dst_row,
    int* export_for_src_row,
    cudaStream_t stream) noexcept
{
    const int num_not_finished = num_rows; // TODO Take this as an input
    MOEParallelismConfig parallelism_config = getParallelismConfig(); 

    mMOERunner->runMoe(
        input,   // const void*
        moe_gates,
        fc1_expert_weight,
        fc1_quant_scale,
        fc1_expert_bias,
        mActivationType, 
        fc2_expert_weight,
        fc2_quant_scale,
        fc2_expert_bias,
        fc3_expert_weight,
        fc3_quant_scale,
        fc3_expert_bias,
        num_rows, 
        mExpertHiddenSize, 
        mExpertInterSize,
        mNumExperts, 
        mK, 
        reinterpret_cast<char*>(workspace),
        // Outputs
        final_output,
        fc2_result,
        finished, //bool
        num_not_finished,  //const int
        expert_scale, // void*
        src_row_to_dst_row, // int*
        export_for_src_row,  // int*
        parallelism_config, 
        mNormalizationMode, 
        stream);

    return 0;
}
