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
#include <cstdlib>
#include <chrono>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "rtp_llm/cpp/cuda/cutlass/interface.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/cuda/cublas/cublas.h"

#include "cutlass/numeric_types.h"

// using torch::Tensor;
using torch_ext::get_ptr;

namespace tk  = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

template<typename T>
void int8_gemm_test(const int m, const int n, const int k, const int iters) {
    tk::QuantMode quant_mode      = tk::QuantMode::fromDescription(true, true, true, true, false, false, false, false);
    const bool    per_token_quant = quant_mode.hasPerTokenScaling();
    const bool    per_channel_quant = quant_mode.hasPerChannelScaling();
    const int     row_scale_size    = per_token_quant ? m : 1;
    const int     col_scale_size    = per_channel_quant ? n : 1;

    const at::ScalarType at_int32 = at::ScalarType::Int;
    const at::ScalarType at_int8  = at::ScalarType::Char;
    const at::ScalarType at_fp16  = at::ScalarType::Half;
    const at::ScalarType at_bf16  = at::ScalarType::BFloat16;
    const at::ScalarType at_fp32  = at::ScalarType::Float;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;

    torch::manual_seed(0);

    auto x = torch::randint(-128, 128, {m, k}, torch::dtype(at_int32).requires_grad(false));
    auto w = torch::randint(-128, 128, {k, n}, torch::dtype(at_int32).requires_grad(false));

    RTP_LLM_CHECK(torch::allclose(x, x.to(at_int8).to(at_int32)));
    RTP_LLM_CHECK(torch::allclose(w, w.to(at_int8).to(at_int32)));

    auto y = torch::matmul(x, w);

    // Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)m, (size_t)k}, get_ptr<int32_t>(x)}.saveNpy("x.npy");
    // Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)k, (size_t)n}, get_ptr<int32_t>(w)}.saveNpy("w.npy");
    // Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)m, (size_t)n}, get_ptr<int32_t>(y)}.saveNpy("y.npy");

    auto x_gpu       = x.to(at_int8).to(torch::kCUDA);
    auto w_T_gpu     = w.to(at_int8).to(torch::kCUDA).t().contiguous();
    auto w_gpu       = w.to(at_int8).to(torch::kCUDA);
    auto y_gpu       = torch::zeros({m, n}, torch::dtype(at_fp16).device(torch::kCUDA).requires_grad(false));
    auto y_gpu_int32 = torch::zeros({m, n}, torch::dtype(at_int32).device(torch::kCUDA).requires_grad(false));

    auto alpha_row_cultass = torch::ones({row_scale_size, 1}, torch::dtype(at_fp32).requires_grad(false)) * (1.0 / 100)
                             * torch::randint(1, 10, {row_scale_size, 1}, torch::dtype(at_fp32));
    auto alpha_col_cutlass = torch::ones({1, col_scale_size}, torch::dtype(at_fp32).requires_grad(false)) * (1.0 / 100)
                             * torch::randint(1, 10, {1, col_scale_size}, torch::dtype(at_fp32));

    auto alpha_row_torch = alpha_row_cultass.expand({m, 1});
    auto alpha_col_torch = alpha_col_cutlass.expand({1, n});

    // std::cout << alpha_row << std::endl;
    auto alpha_row_gpu = alpha_row_cultass.to(torch::kCUDA);
    auto alpha_col_gpu = alpha_col_cutlass.to(torch::kCUDA);

    auto alpha_row_col_scale_gpu = torch::matmul(alpha_row_torch, alpha_col_torch).to(torch::kCUDA);

    tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunner<T> runner;
    char*                                                            ws_ptr = nullptr;
    const int                                                        wsSize = runner.getWorkspaceSize(m, n, k);
    deviceMalloc(&ws_ptr, wsSize);

    // warm up
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const auto gemmConfig = runner.getChosenConfig(get_ptr<int8_t>(x_gpu),
                                                   get_ptr<int8_t>(w_T_gpu),
                                                   quant_mode,
                                                   get_ptr<float>(alpha_col_gpu),
                                                   get_ptr<float>(alpha_row_gpu),
                                                   get_ptr<T>(y_gpu),
                                                   m,
                                                   n,
                                                   k,
                                                   ws_ptr,
                                                   wsSize,
                                                   stream);
    printf("gemm run \n ");
    runner.gemm(get_ptr<int8_t>(x_gpu),
                get_ptr<int8_t>(w_T_gpu),
                quant_mode,
                get_ptr<float>(alpha_col_gpu),
                get_ptr<float>(alpha_row_gpu),
                get_ptr<T>(y_gpu),
                m,
                n,
                k,
                gemmConfig,
                ws_ptr,
                wsSize,
                stream);

    // Tensor{MEMORY_GPU, TYPE_INT8, {(size_t)m, (size_t)k}, get_ptr<int8_t>(x_gpu)}.saveNpy("x_gpu.npy");
    // Tensor{MEMORY_GPU, TYPE_INT8, {(size_t)n, (size_t)k}, get_ptr<int8_t>(w_T_gpu)}.saveNpy("w_T_gpu.npy");
    // Tensor{MEMORY_GPU, TYPE_INT8, {(size_t)k, (size_t)n}, get_ptr<int8_t>(w_gpu)}.saveNpy("w_gpu.npy");
    // Tensor{MEMORY_GPU, TYPE_FP16, {(size_t)m, (size_t)n}, get_ptr<T>(y_gpu)}.saveNpy("y_gpu.npy");
    // Tensor{MEMORY_GPU, TYPE_INT32, {(size_t)m, (size_t)n}, get_ptr<int32_t>(y_gpu_int32)}.saveNpy("y_gpu_int32.npy");

    check_cuda_value(cudaStreamSynchronize(stream));
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iters; ++i) {
        runner.gemm(get_ptr<int8_t>(x_gpu),
                    get_ptr<int8_t>(w_T_gpu),
                    quant_mode,
                    get_ptr<float>(alpha_col_gpu),
                    get_ptr<float>(alpha_row_gpu),
                    get_ptr<T>(y_gpu),
                    m,
                    n,
                    k,
                    gemmConfig,
                    ws_ptr,
                    wsSize,
                    stream);
    }

    check_cuda_value(cudaStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    if (torch::allclose((y.to(torch::kCUDA).to(at_fp32) * alpha_row_col_scale_gpu.to(torch::kCUDA)).to(at_fp16),
                        y_gpu)) {
        RTP_LLM_LOG_INFO("SUCCESS " + std::to_string((double(duration.count()) / iters) / 1000) + " ms");
    } else {
        RTP_LLM_LOG_ERROR("FAILED " + std::to_string((double(duration.count()) / iters) / 1000) + " ms");
        // std::cout << "diff " << (y.to(torch::kCUDA).to(at_fp32) *
        // alpha_row_col_scale_gpu.to(torch::kCUDA)).to(at_fp16) - y_gpu << std::endl;
    }
    deviceFree(ws_ptr);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        RTP_LLM_LOG_ERROR("arguments missing, needs m, n, k, iters.");
        return 0;
    }

    const int m     = atoi(argv[1]);
    const int n     = atoi(argv[2]);
    const int k     = atoi(argv[3]);
    const int iters = atoi(argv[4]);
    int8_gemm_test<half>(m, n, k, iters);

    return 0;
}
