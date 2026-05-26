/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cublas_v2.h>
#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "rtp_llm/cpp/cuda/cutlass/interface.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/quantize_utils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {

namespace tkc = tensorrt_llm::kernels::cutlass_kernels;
namespace tc  = tensorrt_llm::cutlass_extensions;

template<typename T, typename WeightType>
Tensor fused_gemm_dq_helper(Tensor        input_activations,
                            Tensor        weight,
                            Tensor        scales,
                            Tensor        zeros,
                            const float   group_size,
                            const int64_t timing_iterations,
                            float&        avg_time,
                            bool          select_config) {
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(1);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);
    const T*          zeros_ptr     = get_ptr<const T>(zeros);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    using WeightOnlyGemmRunner    = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
    using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

    WeightOnlyGemmRunnerPtr runner;

    if (std::is_same<WeightType, cutlass::uint4b_t>::value) {
        runner =
            std::make_shared<tkc::CutlassFpAIntBGemmRunner<T,
                                                           cutlass::uint4b_t,
                                                           cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    }
    const int ws_bytes = runner->getWorkspaceSize(m, n, k);
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    T*          output_tensor_ptr = get_ptr<T>(output_tensor);
    char*       ws_ptr            = get_ptr<char>(ws_tensor);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (!select_config) {
        tc::CutlassGemmConfig config = runner->getChosenConfig(input_act_ptr,
                                                               weight_ptr,
                                                               scales_ptr,
                                                               zeros_ptr,
                                                               nullptr,
                                                               output_tensor_ptr,
                                                               m,
                                                               n,
                                                               k,
                                                               group_size,
                                                               ws_ptr,
                                                               ws_bytes,
                                                               stream);

        auto runner_operation = [&](cudaStream_t stream) {
            runner->gemm(input_act_ptr,
                         weight_ptr,
                         scales_ptr,
                         zeros_ptr,
                         nullptr,
                         output_tensor_ptr,
                         m,
                         n,
                         k,
                         group_size,
                         config,
                         ws_ptr,
                         ws_bytes,
                         stream);
        };
        avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);
    } else {
        tc::CutlassGemmConfig              best_config;
        std::vector<tc::CutlassGemmConfig> configs  = runner->getConfigs();
        float                              min_time = 100000.0f;

        for (int i = 0; i < configs.size(); i++) {
            float total_time_fpaintb = 0;
            if (tkc::is_valid_split_k_factor(m, n, k, configs[i], ws_bytes, true) == false) {
                continue;
            }

            auto runner_operation = [&](cudaStream_t stream) {
                runner->gemm(input_act_ptr,
                             weight_ptr,
                             scales_ptr,
                             zeros_ptr,
                             nullptr,
                             output_tensor_ptr,
                             m,
                             n,
                             k,
                             group_size,
                             configs[i],
                             ws_ptr,
                             ws_bytes,
                             stream);
            };
            float cur_avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);

            if (cur_avg_time < min_time) {
                min_time    = cur_avg_time;
                best_config = configs[i];
            }
        }
        avg_time = min_time;
        tkc::print_config_file(best_config, m, n, k, min_time, "./config.ini", std::ios::app);
    }
    return output_tensor;
}

template<typename T, typename WeightType>
Tensor fused_gemm_dq_helper(Tensor        input_activations,
                            Tensor        weight,
                            Tensor        scales,
                            const int64_t timing_iterations,
                            float&        avg_time,
                            bool          select_config) {
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(0);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    using WeightOnlyGemmRunner    = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
    using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;
    WeightOnlyGemmRunnerPtr runner;

    if (std::is_same<WeightType, uint8_t>::value) {
        runner = std::make_shared<
            tkc::CutlassFpAIntBGemmRunner<T, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
    }
    const int ws_bytes = runner->getWorkspaceSize(m, n, k);
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    T*    output_tensor_ptr = get_ptr<T>(output_tensor);
    char* ws_ptr            = get_ptr<char>(ws_tensor);

    if (!select_config) {
        tc::CutlassGemmConfig config = runner->getChosenConfig(input_act_ptr,
                                                               weight_ptr,
                                                               scales_ptr,
                                                               nullptr,
                                                               nullptr,
                                                               output_tensor_ptr,
                                                               m,
                                                               n,
                                                               k,
                                                               k,
                                                               ws_ptr,
                                                               ws_bytes,
                                                               stream);

        auto runner_operation = [&](cudaStream_t stream) {
            runner->gemm(
                input_act_ptr, weight_ptr, scales_ptr, output_tensor_ptr, m, n, k, config, ws_ptr, ws_bytes, stream);
        };
        avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);
    } else {
        tc::CutlassGemmConfig              best_config;
        std::vector<tc::CutlassGemmConfig> configs  = runner->getConfigs();
        float                              min_time = 10000.0f;

        for (int i = 0; i < configs.size(); i++) {
            float total_time_fpaintb = 0;
            // tensorrt_llm::kernels::cutlass_kernels::print_config(configs[i]);
            if (tkc::is_valid_split_k_factor(m, n, k, configs[i], ws_bytes, true) == false) {
                continue;
            }

            auto runner_operation = [&](cudaStream_t stream) {
                runner->gemm(input_act_ptr,
                             weight_ptr,
                             scales_ptr,
                             output_tensor_ptr,
                             m,
                             n,
                             k,
                             configs[i],
                             ws_ptr,
                             ws_bytes,
                             stream);
            };
            float cur_avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);

            if (cur_avg_time < min_time) {
                min_time    = cur_avg_time;
                best_config = configs[i];
            }
        }
        avg_time = min_time;
        tkc::print_config_file(best_config, m, n, k, min_time, "./config.ini", std::ios::app);
    }
    return output_tensor;
}

template<typename T, typename WeightType>
Tensor fused_gemv_dq_helper(
    Tensor input_activations, Tensor weight, Tensor scales, const int64_t timing_iterations, float& avg_time) {
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(0);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    T* output_tensor_ptr = get_ptr<T>(output_tensor);

    int arch = rtp_llm::get_sm();

    tensorrt_llm::kernels::weight_only::KernelType type =
        tensorrt_llm::kernels::weight_only::KernelType::FP16Int8PerChannel;

    tensorrt_llm::kernels::weight_only::Params params(
        input_act_ptr, nullptr, weight_ptr, scales_ptr, nullptr, nullptr, output_tensor_ptr, 1.f, m, n, k, 0, type);
    tensorrt_llm::kernels::weight_only::kernel_launcher(arch, params, stream);

    avg_time = 0;

    return output_tensor;
}

Tensor _fused_gemm_dq(Tensor  input_activations,
                      Tensor  weight,
                      Tensor  scales,
                      Tensor  zeros,
                      int64_t group_size,
                      int64_t timing_iterations,
                      float&  avg_time,
                      bool    use_tensor_core,
                      bool    select_config) {
    const at::ScalarType _st = input_activations.scalar_type();
    CHECK_INPUT(scales, _st);

    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");

    TORCH_CHECK(input_activations.size(1) == weight.size(0), "dim 1 of act and dim 0 of weight must be equal");

    // We signal int4 by having the last weight dim be half the size of the scales.
    // This is because int4 elements are packed into a single byte.
    torch::ScalarType quant_type = weight.scalar_type();
    if (weight.size(-1) == scales.size(-1) / 2) {
        quant_type = at::ScalarType::QUInt4x2;
    } else {
        TORCH_CHECK(weight.size(-1) == scales.size(-1),
                    "Last dim of weight and scales must be equal for int8 "
                    "or last dim of scale must be 2x last dim of weight for int4.");
    }

    Tensor output_tensor;
    switch (_st) {
        case at::ScalarType::Half: {
            if (quant_type == torch::kInt8) {
                if (use_tensor_core) {
                    output_tensor = fused_gemm_dq_helper<half, uint8_t>(
                        input_activations, weight, scales, timing_iterations, avg_time, select_config);
                } else {
                    output_tensor = fused_gemv_dq_helper<half, uint8_t>(
                        input_activations, weight, scales, timing_iterations, avg_time);
                }
            } else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<half, cutlass::uint4b_t>(
                    input_activations, weight, scales, zeros, group_size, timing_iterations, avg_time, select_config);
            } else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == torch::kInt8) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, uint8_t>(
                    input_activations, weight, scales, timing_iterations, avg_time, select_config);
            } else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, cutlass::uint4b_t>(
                    input_activations, weight, scales, zeros, group_size, timing_iterations, avg_time, select_config);
            } else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported tensor type. Got " + std::string(at::toString(_st)));
    }
    return output_tensor;
}

Tensor fused_gemm_dq(
    Tensor input_activations, Tensor weight, Tensor scales, Tensor zeros, int64_t group_size, bool use_tensor_core) {
    float dummy = 0.f;
    return _fused_gemm_dq(input_activations, weight, scales, zeros, group_size, 1, dummy, use_tensor_core, false);
}

Tensor gemm_config_select(Tensor        input_activations,
                          Tensor        weight,
                          Tensor        scales,
                          Tensor        zeros,
                          int64_t       group_size,
                          const int64_t timing_iterations) {
    float avg_time = 0;
    return _fused_gemm_dq(
        input_activations, weight, scales, zeros, group_size, timing_iterations, avg_time, true, true);
}

Tensor
bench_cublas(Tensor input_activations, Tensor weight_dequantized, const int64_t timing_iterations, float& avg_time) {

    const int m = input_activations.size(0);
    const int n = weight_dequantized.size(1);
    const int k = input_activations.size(1);

    const void* input_act_ptr = get_ptr<const void>(input_activations);
    const void* weight_ptr    = get_ptr<const void>(weight_dequantized);

    cublasHandle_t       handle = at::cuda::getCurrentCUDABlasHandle();
    const at::ScalarType _st    = input_activations.scalar_type();

    TORCH_CHECK(input_activations.size(1) == weight_dequantized.size(0),
                "CUBLAS_BENCH: dim 1 of act and dim 0 of weight must be equal");
    CHECK_INPUT(input_activations, _st);
    CHECK_INPUT(weight_dequantized, _st);

    auto  output_tensor     = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    void* output_tensor_ptr = get_ptr<void>(output_tensor);

    TORCH_CHECK(_st == at::ScalarType::Half || _st == at::ScalarType::BFloat16, "Input type must be float or bfloat");
    cudaDataType_t cublasType = _st == at::ScalarType::Half ? CUDA_R_16F : CUDA_R_16BF;

    float alpha = 1.0f;
    float beta  = 0.0f;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaEventRecord(start, stream);
    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        status = cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              weight_ptr,
                              cublasType,
                              n,
                              input_act_ptr,
                              cublasType,
                              k,
                              &beta,
                              output_tensor_ptr,
                              cublasType,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    avg_time = total_time_ms / float(timing_iterations);
    check_cuda_value(status);
    return output_tensor;
}

std::vector<std::vector<Tensor>> benchmark_against_cublas_fp(Tensor        input_activations,
                                                             Tensor        weight_quantized,
                                                             Tensor        scales,
                                                             Tensor        zeros,
                                                             Tensor        weight_dequantized,
                                                             int64_t       group_size,
                                                             const int64_t timing_iterations,
                                                             bool          use_tensor_core) {
    float  cublas_time   = 0.f;
    float  ft_time       = 0.f;
    Tensor cublas_result = bench_cublas(input_activations, weight_dequantized, timing_iterations, cublas_time);
    Tensor ft_result     = _fused_gemm_dq(input_activations,
                                      weight_quantized,
                                      scales,
                                      zeros,
                                      group_size,
                                      timing_iterations,
                                      ft_time,
                                      use_tensor_core,
                                      false);

    auto timing_tensor =
        torch::empty({2}, torch::dtype(at::ScalarType::Float).device(torch::kCPU).requires_grad(false));
    timing_tensor[0] = cublas_time;
    timing_tensor[1] = ft_time;

    return {{timing_tensor}, {cublas_result, ft_result}};
}

Tensor unpack_int4_packed_tensor_to_int8(Tensor weight) {
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a packed int8 tensor");

    std::vector<long int> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
        int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    Tensor unpacked_weight =
        torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr   = get_ptr<int8_t>(weight);
    int8_t* unpacked_ptr = get_ptr<int8_t>(unpacked_weight);

    for (int packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
        int8_t packed_data = packed_ptr[packed_idx];

        int8_t elt_0 = (int8_t(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
        int8_t elt_1 = packed_data >> 4;

        unpacked_ptr[2 * packed_idx + 0] = elt_0;
        unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
}

TORCH_LIBRARY(gemm_dq_unit_ops, m) {
    m.def("fused_gemm_dq", fused_gemm_dq);
    m.def("gemm_config_select", gemm_config_select);
    m.def("benchmark_against_cublas_fp", benchmark_against_cublas_fp);
    m.def("unpack_int4_packed_tensor_to_int8", unpack_int4_packed_tensor_to_int8);
}

}  // namespace torch_ext
