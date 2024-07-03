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

#pragma once
#include "src/fastertransformer/cutlass/cutlass_kernels/gemm_configs.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/cutlass/cutlass_kernels/weight_only_quant_op.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace tensorrt_llm
{

class MoeGemmRunnerInterface
{
public:
    MoeGemmRunnerInterface () {}
    
    virtual ~MoeGemmRunnerInterface () {}

    virtual void setBestConfig(std::optional<cutlass_extensions::CutlassGemmConfig> best_config) = 0;

    virtual void moeGemmBiasAct(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, const void* biases, void* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size,
        fastertransformer::ActivationType activation_type, cudaStream_t stream) = 0;

    virtual void moeGemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, void* C, int64_t* total_rows_before_expert,
        int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size, cudaStream_t stream) = 0;

    virtual std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs() = 0;


}; 

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class MoeGemmRunner : public virtual MoeGemmRunnerInterface
{
public:
    MoeGemmRunner();

    void setBestConfig(std::optional<cutlass_extensions::CutlassGemmConfig> best_config)
    {
        best_config_ = std::move(best_config);
    }

    void moeGemmBiasAct(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, const void* biases, void* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size,
        fastertransformer::ActivationType activation_type, cudaStream_t stream);

    void moeGemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, void* C, int64_t* total_rows_before_expert,
        int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size, cudaStream_t stream);

    std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs();

private:
    template <typename EpilogueTag>
    void dispatchToArch(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size,
        cutlass_extensions::CutlassGemmConfig gemm_config, cudaStream_t stream, int* occupancy = nullptr);

    template <typename EpilogueTag>
    void runGemm(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size,
        cudaStream_t stream);

private:
    int sm_;
    int multi_processor_count_;
    std::optional<cutlass_extensions::CutlassGemmConfig> best_config_{};
};

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class MoeGemmRunner<float, WeightType, QuantOp> : public virtual MoeGemmRunnerInterface
{
public:
    MoeGemmRunner();

    void setBestConfig(std::optional<cutlass_extensions::CutlassGemmConfig> best_config) {}

    void moeGemmBiasAct(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, const void* biases, void* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size,
        fastertransformer::ActivationType activation_type, cudaStream_t stream);

    void moeGemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points,void* C, int64_t* total_rows_before_expert,
        int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int group_size, cudaStream_t stream);

    std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs();
};

} // namespace tensorrt_llm
