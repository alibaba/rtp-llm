// based on flashinfer 0.4.1 https://github.com/flashinfer-ai/flashinfer/tree/a88349f9f43df74d31d1d52ad5aa20c28824a790
/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/hip/HIPGeneratorImpl.h>

#include "sampling.h"
#include "utils.h"
#include "kernel.cuh"

#include "json.hpp"
#include <fstream>

namespace rtp_llm {

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator) {
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      generator, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(increment_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;
  return std::make_tuple(philox_seed, philox_offset);
}

void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, torch::Tensor philox_seed, torch::Tensor philox_offset, uintptr_t stream) {

if (std::getenv("XBJ_CHECK_PARAMS")) {
    const char* json_file_path = std::getenv("XBJ_CHECK_PARAMS");
    
    nlohmann::json j;
    j["function"] = "top_p_sampling_from_probs";
    
    // 记录 probs 信息
    nlohmann::json probs_info;
    probs_info["shape"] = nlohmann::json::array();
    for (int i = 0; i < probs.dim(); ++i) {
        probs_info["shape"].push_back(probs.size(i));
    }
    probs_info["dtype"] = probs.dtype().name();
    probs_info["device"] = probs.device().str();
    
    // 获取 probs 所有数据
    torch::Tensor cpu_probs = probs;
    if (probs.is_cuda()) {
        cpu_probs = probs.cpu();
    }
    probs_info["data"] = nlohmann::json::array();
    std::cout << "XBJ 1\n";
    auto probs_acc = cpu_probs.accessor<float, 2>();
    for (int i = 0; i < cpu_probs.size(0); ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int k = 0; k < cpu_probs.size(1); ++k) {
            row.push_back(probs_acc[i][k]);
        }
        probs_info["data"].push_back(row);
    }
    j["probs"] = probs_info;
    
    // 记录 output 信息
    nlohmann::json output_info;
    output_info["shape"] = nlohmann::json::array();
    for (int i = 0; i < output.dim(); ++i) {
        output_info["shape"].push_back(output.size(i));
    }
    output_info["dtype"] = output.dtype().name();
    output_info["device"] = output.device().str();
    j["output"] = output_info;
    
    // 记录 maybe_indices 信息
    if (maybe_indices.has_value()) {
        auto indices = maybe_indices.value();
        nlohmann::json indices_info;
        indices_info["shape"] = nlohmann::json::array();
        for (int i = 0; i < indices.dim(); ++i) {
            indices_info["shape"].push_back(indices.size(i));
        }
        indices_info["dtype"] = indices.dtype().name();
        indices_info["device"] = indices.device().str();
        
        // 获取 indices 所有数据
        torch::Tensor cpu_indices = indices;
        if (indices.is_cuda()) {
            cpu_indices = indices.cpu();
        }
        indices_info["data"] = nlohmann::json::array();
    std::cout << "XBJ 2\n";
        auto indices_acc = cpu_indices.accessor<int, 1>();
        for (int i = 0; i < cpu_indices.size(0); ++i) {
            indices_info["data"].push_back(indices_acc[i]);
        }
        j["indices"] = indices_info;
    } else {
        j["indices"] = nullptr;
    }
    
    // 记录 maybe_top_p_arr 信息
    if (maybe_top_p_arr.has_value()) {
        auto top_p_arr = maybe_top_p_arr.value();
        nlohmann::json top_p_arr_info;
        top_p_arr_info["shape"] = nlohmann::json::array();
        for (int i = 0; i < top_p_arr.dim(); ++i) {
            top_p_arr_info["shape"].push_back(top_p_arr.size(i));
        }
        top_p_arr_info["dtype"] = top_p_arr.dtype().name();
        top_p_arr_info["device"] = top_p_arr.device().str();
        
        // 获取 top_p_arr 所有数据
        torch::Tensor cpu_top_p_arr = top_p_arr;
        if (top_p_arr.is_cuda()) {
            cpu_top_p_arr = top_p_arr.cpu();
        }
        top_p_arr_info["data"] = nlohmann::json::array();
    std::cout << "XBJ 3\n";
        auto top_p_acc = cpu_top_p_arr.accessor<float, 1>();
        for (int i = 0; i < cpu_top_p_arr.size(0); ++i) {
            top_p_arr_info["data"].push_back(top_p_acc[i]);
        }
        j["top_p_arr"] = top_p_arr_info;
    } else {
        j["top_p_arr"] = nullptr;
    }
    
    // 记录其他参数
    j["top_p_val"] = top_p_val;
    j["deterministic"] = deterministic;
    
    // 记录 philox_seed 信息
    nlohmann::json seed_info;
    seed_info["shape"] = nlohmann::json::array();
    for (int i = 0; i < philox_seed.dim(); ++i) {
        seed_info["shape"].push_back(philox_seed.size(i));
    }
    seed_info["dtype"] = philox_seed.dtype().name();
    seed_info["device"] = philox_seed.device().str();
    
    // 获取 philox_seed 所有数据
    torch::Tensor cpu_seed = philox_seed;
    if (philox_seed.is_cuda()) {
        cpu_seed = philox_seed.cpu();
    }
    seed_info["data"] = nlohmann::json::array();
    std::cout << "XBJ 4\n";
    auto seed_acc = cpu_seed.accessor<int64_t, 1>();
    for (int i = 0; i < cpu_seed.size(0); ++i) {
        seed_info["data"].push_back(seed_acc[i]);
    }
    j["philox_seed"] = seed_info;
    
    // 记录 philox_offset 信息
    nlohmann::json offset_info;
    offset_info["shape"] = nlohmann::json::array();
    for (int i = 0; i < philox_offset.dim(); ++i) {
        offset_info["shape"].push_back(philox_offset.size(i));
    }
    offset_info["dtype"] = philox_offset.dtype().name();
    offset_info["device"] = philox_offset.device().str();
    
    // 获取 philox_offset 所有数据
    torch::Tensor cpu_offset = philox_offset;
    if (philox_offset.is_cuda()) {
        cpu_offset = philox_offset.cpu();
    }
    offset_info["data"] = nlohmann::json::array();
    std::cout << "XBJ 5\n";
    auto offset_acc = cpu_offset.accessor<int64_t, 1>();
    for (int i = 0; i < cpu_offset.size(0); ++i) {
        offset_info["data"].push_back(offset_acc[i]);
    }
    j["philox_offset"] = offset_info;
    
    // 写入 JSON 文件
    std::ofstream file(json_file_path);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
    }
}
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  hipSetDevice(probs.get_device());
  hipError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, deterministic, static_cast<uint64_t*>(philox_seed.data_ptr()), static_cast<uint64_t*>(philox_offset.data_ptr()), reinterpret_cast<hipStream_t>(stream));
  TORCH_CHECK(status == hipSuccess, "TopPSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
// 在函数末尾添加以下代码，放在 hipError_t status 检查之后
if (std::getenv("XBJ_CHECK_PARAMS")) {
    const char* json_file_path = std::getenv("XBJ_CHECK_PARAMS");
    
    // 等待内核执行完成
    hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream));
    
    // 读取现有JSON文件
    std::ifstream input_file(json_file_path);
    nlohmann::json j_final;
    
    if (input_file.is_open()) {
        input_file >> j_final;
        input_file.close();
    } else {
        j_final = nlohmann::json::object();
    }
    
    // 获取 output 所有数据
    nlohmann::json output_result;
    output_result["shape"] = nlohmann::json::array();
    for (int i = 0; i < output.dim(); ++i) {
        output_result["shape"].push_back(output.size(i));
    }
    output_result["dtype"] = output.dtype().name();
    output_result["device"] = output.device().str();
    
    torch::Tensor cpu_output = output;
    if (output.is_cuda()) {
        cpu_output = output.cpu();
    }
    
    output_result["data"] = nlohmann::json::array();
    std::cout << "XBJ 6\n";
    auto output_acc = cpu_output.accessor<int, 1>();
    for (int i = 0; i < cpu_output.size(0); ++i) {
        output_result["data"].push_back(output_acc[i]);
    }
    
    j_final["output_result"] = output_result;
    
    // 写回文件
    std::ofstream output_file(json_file_path);
    if (output_file.is_open()) {
        output_file << j_final.dump(2);
        output_file.close();
    }
}
}

void top_k_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, torch::Tensor philox_seed, torch::Tensor philox_offset, uintptr_t stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  hipSetDevice(probs.get_device());
  hipError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_k_arr ? static_cast<float*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, deterministic, static_cast<uint64_t*>(philox_seed.data_ptr()), static_cast<uint64_t*>(philox_offset.data_ptr()), reinterpret_cast<hipStream_t>(stream));
  TORCH_CHECK(status == hipSuccess, "TopKSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, torch::Tensor philox_seed,
                                     torch::Tensor philox_offset, uintptr_t stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  hipSetDevice(probs.get_device());
  hipError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, top_k_val, top_p_val, vocab_size, deterministic, static_cast<uint64_t*>(philox_seed.data_ptr()), static_cast<uint64_t*>(philox_offset.data_ptr()),
      reinterpret_cast<hipStream_t>(stream));
  TORCH_CHECK(status == hipSuccess, "TopKTopPSamplingFromProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_p_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val, uintptr_t stream) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  hipSetDevice(probs.get_device());
  hipError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, reinterpret_cast<hipStream_t>(stream));
  
  TORCH_CHECK(status == hipSuccess, "TopPRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val, uintptr_t stream) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  hipSetDevice(probs.get_device());
  hipError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, reinterpret_cast<hipStream_t>(stream));

  TORCH_CHECK(status == hipSuccess, "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

}