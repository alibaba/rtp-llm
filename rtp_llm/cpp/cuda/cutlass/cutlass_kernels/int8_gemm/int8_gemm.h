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

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_configs.h"
#include "rtp_llm/cpp/model_utils/quantization.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut.h"
#include <cuda_runtime_api.h>

namespace tk = tensorrt_llm::common;
namespace tc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

enum class CutlassActivationType {
    INVALID,
    IDENTITY,
    GELU_FAST,
    GELU,
    SIGMOID,
    RELU,
    SILU
};
/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

class CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunnerInterface() {}

    virtual ~CutlassInt8GemmRunnerInterface() {}

    virtual void gemm(const void* A, const void* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, void* C, void* bias, CutlassActivationType activation_type, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream)
        = 0;

    static bool addBiasActivationEpilogueSupported(CutlassActivationType activation) {
        return activation == CutlassActivationType::IDENTITY || activation == CutlassActivationType::GELU_FAST;
    }

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(const int m, const int n, const int k) = 0;

    virtual std::vector<tc::CutlassGemmConfig> getConfigs() const = 0;

    virtual tc::CutlassGemmConfig getChosenConfig(const void* A, const void* B, tk::QuantMode quantOption,
        const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream)
        = 0;

    virtual std::vector<tc::CutlassGemmConfig> getValidConfigs(const void* A, const void* B, tk::QuantMode quantOption,
        const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) = 0;


protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunner();
    ~CutlassInt8GemmRunner();

    void gemm(const void* A, const void* B, tk::QuantMode quantOption, const float* alphaCol, const float* alphaRow,
        void* C, void* bias, CutlassActivationType activation_type, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) override;

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(const int m, const int n, const int k) override;

    std::vector<tc::CutlassGemmConfig> getConfigs() const override;

    tc::CutlassGemmConfig getChosenConfig(const void* A, const void* B, tk::QuantMode quantOption,
        const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream);

    std::vector<tc::CutlassGemmConfig> getValidConfigs(const void* A, const void* B, tk::QuantMode quantOption,
        const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream);

private:
    void dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, T* Bias, CutlassActivationType activation, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    int mSm;
    int mMultiProcessorCount;
    const GemmLut* gemm_lut_;
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm