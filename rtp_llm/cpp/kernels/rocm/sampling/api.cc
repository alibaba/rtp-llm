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
#include "sampling.h"
#include "utils.h"
#include "kernel.cuh"

namespace rtp_llm {

void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset, uintptr_t stream) {
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
      top_p_val, vocab_size, deterministic, philox_seed, philox_offset, reinterpret_cast<hipStream_t>(stream));
  TORCH_CHECK(status == hipSuccess, "TopPSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset, uintptr_t stream) {
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
      top_k_val, vocab_size, deterministic, philox_seed, philox_offset, reinterpret_cast<hipStream_t>(stream));
  TORCH_CHECK(status == hipSuccess, "TopKSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset, uintptr_t stream) {
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
      batch_size, top_k_val, top_p_val, vocab_size, deterministic, philox_seed, philox_offset,
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