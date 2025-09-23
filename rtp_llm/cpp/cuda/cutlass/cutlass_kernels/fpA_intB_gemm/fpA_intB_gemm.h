/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/weight_only_quant_op.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_configs.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut.h"
#include <cuda_runtime_api.h>

namespace tc = tensorrt_llm::cutlass_extensions;
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

/*
  This runner only supports:
  T in {half, __nv_bfloat} WeightType in {int8_t, cutlass::uint4b_t}

  Activations, biases, scales and outputs are all assumed to be row-major.

  However, it is assumed that B is in a special format governed by cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.
  In this case, B must be preprocessed using the cutlass weight only quant preprocessors. The weight preprocessor
  will instantiate the layout and preprocess based on the instantiation, so layout changes should only require
  modifications to mix_gemm_B_layout.h.
*/

class CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunnerInterface() {}

    virtual ~CutlassFpAIntBGemmRunnerInterface() {}

    virtual void gemm(const void* A, const void* B, const void* weight_scales, void* C, int m, int n, int k,
        tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual void gemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points,
        const void* biases, void* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(const int m, const int n, const int k) = 0;

    virtual std::vector<tc::CutlassGemmConfig> getConfigs() const = 0;

    virtual tc::CutlassGemmConfig getChosenConfig(const void* A, const void* B, const void* weight_scales,
        const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual std::vector<tc::CutlassGemmConfig> getValidConfigs(const void* A, const void* B, const void* weight_scales,
        const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 128;
};

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class CutlassFpAIntBGemmRunner : public virtual CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunner();
    ~CutlassFpAIntBGemmRunner();

    void gemm(const void* A, const void* B, const void* weight_scales, void* C, int m, int n, int k,
        tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points,
        const void* biases, void* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

    // Disabled since the fused GEMM, activation kernels will not be used in v1.

    // void gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C, int m, int n,
    //     int k, ActivationType activation_type, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t
    //     stream);

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(const int m, const int n, const int k) override;

    std::vector<tc::CutlassGemmConfig> getConfigs() const override;

    tc::CutlassGemmConfig getChosenConfig(const void* A, const void* B, const void* weight_scales,
        const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

    std::vector<tc::CutlassGemmConfig> getValidConfigs(const void* A, const void* B, const void* weight_scales,
        const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

private:
    template <typename EpilogueTag>
    void dispatch_to_arch(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points,
        const T* biases, T* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemm_config,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr);

private:
    int sm_;
    int multi_processor_count_;
    const GemmLut* gemm_lut_;
};


template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class CutlassFpAIntBGemmRunner<float, WeightType, QuantOp> : public virtual CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunner() = default;
    ~CutlassFpAIntBGemmRunner() = default;

    void gemm(const void* A, const void* B, const void* weight_scales, void* C, int m, int n, int k,
        tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points,
        const void* biases, void* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream);

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(const int m, const int n, const int k);

    std::vector<tc::CutlassGemmConfig> getConfigs() const;

    tc::CutlassGemmConfig getChosenConfig(const void* A, const void* B, const void* weight_scales,
        const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream);
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
