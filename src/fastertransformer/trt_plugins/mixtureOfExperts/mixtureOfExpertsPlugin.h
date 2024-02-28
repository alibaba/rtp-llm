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

#include "src/fastertransformer/utils/quantization.h"
#include "src/fastertransformer/cutlass/cutlass_kernels/moe_gemm/moe_kernels.h"
#include "src/fastertransformer/utils/trt_utils.h"
#include "src/fastertransformer/utils/activation_types.h"

#include <cassert>
// #include <mpi.h>
#include <set>
#include <string>
#include <vector>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::plugins
{

class MixtureOfExpertsPlugin 
{
public:
    // using MOEParallelismMode = tensorrt_llm::kernels::MOEParallelismMode;
    using MOEExpertScaleNormalizationMode = tensorrt_llm::kernels::MOEExpertScaleNormalizationMode;

    MixtureOfExpertsPlugin() = delete;
    MixtureOfExpertsPlugin(int number_of_experts, int top_k, int expert_hidden_size, int expert_inter_size,
        fastertransformer::ActivationType activation_type, nvinfer1::DataType type, nvinfer1::DataType weight_type,
        MOEExpertScaleNormalizationMode normalization_mode);

    void init();

    ~MixtureOfExpertsPlugin() = default;

    size_t getWorkspaceSize(int num_tokens);
    int enqueue(
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
    cudaStream_t stream) noexcept;

private:
    std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> mMOERunner{};
    int mNumExperts{};
    int mK{};
    int mExpertHiddenSize{};
    int mExpertInterSize{};
    fastertransformer::ActivationType mActivationType;
    nvinfer1::DataType mType{};
    nvinfer1::DataType mWeightType{};
    // tensorrt_llm::common::QuantMode mQuantMode;
    // int mTPSize{};
    // int mTPRank{};
    // MOEParallelismMode mParallelismMode{};
    MOEExpertScaleNormalizationMode mNormalizationMode{};

    kernels::MOEParallelismConfig getParallelismConfig() const;

};

} // namespace tensorrt_llm::plugins

#endif // TRT_MIXTURE_OF_EXPERTS_PLUGIN_H
