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
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "sampling.h"
#include "utils.h"
#include "kernel.cuh"

namespace rtp_llm {

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator) {
    uint64_t philox_seed, philox_offset;
    auto     gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(generator, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    at::PhiloxCudaState         rng_engine_inputs = gen->philox_cuda_state(increment_size);
    philox_seed                                   = rng_engine_inputs.seed_.val;
    philox_offset                                 = rng_engine_inputs.offset_.val;
    return std::make_tuple(philox_seed, philox_offset);
}

void top_p_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               bool                         deterministic,
                               torch::Tensor                philox_seed,
                               torch::Tensor                philox_offset,
                               uintptr_t                    stream,
                               std::optional<torch::Tensor> success) {
    if (success.has_value()) {
        CHECK_INPUT(success.value());
        CHECK_DEVICE(success.value(), probs);
        TORCH_CHECK(success->scalar_type() == torch::kBool && success->dim() == 1 && success->numel() == output.numel(),
                    "success must be a bool tensor with one entry per sample");
    }
    CHECK_INPUT(probs);
    CHECK_INPUT(output);
    CHECK_INPUT(philox_seed);
    CHECK_INPUT(philox_offset);
    CHECK_DEVICE(output, probs);
    CHECK_DEVICE(philox_seed, probs);
    CHECK_DEVICE(philox_offset, probs);
    if (maybe_indices.has_value()) {
        CHECK_INPUT(maybe_indices.value());
        CHECK_DEVICE(maybe_indices.value(), probs);
    }
    if (maybe_top_p_arr.has_value()) {
        CHECK_INPUT(maybe_top_p_arr.value());
        CHECK_DEVICE(maybe_top_p_arr.value(), probs);
    }
    CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
    unsigned int batch_size    = output.sizes()[0];
    unsigned int vocab_size    = probs.sizes()[1];
    bool         has_top_p_arr = maybe_top_p_arr.has_value();

    hipSetDevice(probs.get_device());
    hipError_t status = sampling::TopPSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
        batch_size,
        top_p_val,
        vocab_size,
        deterministic,
        static_cast<uint64_t*>(philox_seed.data_ptr()),
        static_cast<uint64_t*>(philox_offset.data_ptr()),
        reinterpret_cast<hipStream_t>(stream),
        success.has_value() ? success->data_ptr<bool>() : nullptr);
    TORCH_CHECK(status == hipSuccess,
                "TopPSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr,
                               int64_t                      top_k_val,
                               bool                         deterministic,
                               torch::Tensor                philox_seed,
                               torch::Tensor                philox_offset,
                               uintptr_t                    stream,
                               std::optional<torch::Tensor> success) {
    if (success.has_value()) {
        CHECK_INPUT(success.value());
        CHECK_DEVICE(success.value(), probs);
        TORCH_CHECK(success->scalar_type() == torch::kBool && success->dim() == 1 && success->numel() == output.numel(),
                    "success must be a bool tensor with one entry per sample");
    }
    CHECK_INPUT(probs);
    CHECK_INPUT(output);
    CHECK_INPUT(philox_seed);
    CHECK_INPUT(philox_offset);
    CHECK_DEVICE(output, probs);
    CHECK_DEVICE(philox_seed, probs);
    CHECK_DEVICE(philox_offset, probs);
    if (maybe_indices.has_value()) {
        CHECK_INPUT(maybe_indices.value());
        CHECK_DEVICE(maybe_indices.value(), probs);
    }
    if (maybe_top_k_arr.has_value()) {
        CHECK_INPUT(maybe_top_k_arr.value());
        CHECK_DEVICE(maybe_top_k_arr.value(), probs);
    }
    CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
    CHECK_DIM(1, output);  // output: (batch_size)
    unsigned int batch_size    = output.sizes()[0];
    unsigned int vocab_size    = probs.sizes()[1];
    bool         has_top_k_arr = maybe_top_k_arr.has_value();

    hipSetDevice(probs.get_device());
    hipError_t status = sampling::TopKSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
        batch_size,
        top_k_val,
        vocab_size,
        deterministic,
        static_cast<uint64_t*>(philox_seed.data_ptr()),
        static_cast<uint64_t*>(philox_offset.data_ptr()),
        reinterpret_cast<hipStream_t>(stream),
        success.has_value() ? success->data_ptr<bool>() : nullptr);
    TORCH_CHECK(status == hipSuccess,
                "TopKSamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(torch::Tensor                probs,
                                     torch::Tensor                output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                     double                       top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                     double                       top_p_val,
                                     bool                         deterministic,
                                     torch::Tensor                philox_seed,
                                     torch::Tensor                philox_offset,
                                     uintptr_t                    stream,
                                     std::optional<torch::Tensor> success) {
    if (success.has_value()) {
        CHECK_INPUT(success.value());
        CHECK_DEVICE(success.value(), probs);
        TORCH_CHECK(success->scalar_type() == torch::kBool && success->dim() == 1 && success->numel() == output.numel(),
                    "success must be a bool tensor with one entry per sample");
    }
    CHECK_INPUT(probs);
    CHECK_INPUT(output);
    CHECK_INPUT(philox_seed);
    CHECK_INPUT(philox_offset);
    CHECK_DEVICE(output, probs);
    CHECK_DEVICE(philox_seed, probs);
    CHECK_DEVICE(philox_offset, probs);
    if (maybe_indices.has_value()) {
        CHECK_INPUT(maybe_indices.value());
        CHECK_DEVICE(maybe_indices.value(), probs);
    }
    if (maybe_top_k_arr.has_value()) {
        CHECK_INPUT(maybe_top_k_arr.value());
        CHECK_DEVICE(maybe_top_k_arr.value(), probs);
    }
    if (maybe_top_p_arr.has_value()) {
        CHECK_INPUT(maybe_top_p_arr.value());
        CHECK_DEVICE(maybe_top_p_arr.value(), probs);
    }
    CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
    CHECK_DIM(1, output);  // output: (batch_size)
    unsigned int batch_size    = output.sizes()[0];
    unsigned int vocab_size    = probs.sizes()[1];
    bool         has_top_k_arr = maybe_top_k_arr.has_value();
    bool         has_top_p_arr = maybe_top_p_arr.has_value();

    hipSetDevice(probs.get_device());
    hipError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
        static_cast<int*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
        batch_size,
        top_k_val,
        top_p_val,
        vocab_size,
        deterministic,
        static_cast<uint64_t*>(philox_seed.data_ptr()),
        static_cast<uint64_t*>(philox_offset.data_ptr()),
        reinterpret_cast<hipStream_t>(stream),
        success.has_value() ? success->data_ptr<bool>() : nullptr);
    TORCH_CHECK(status == hipSuccess,
                "TopKTopPSamplingFromProb failed with error code " + std::string(hipGetErrorString(status)));
}

void finalize_sampling_probs(torch::Tensor                probs,
                             torch::Tensor                samples,
                             torch::Tensor                success,
                             std::optional<torch::Tensor> log_probs,
                             uintptr_t                    stream) {
    CHECK_INPUT(probs);
    CHECK_INPUT(samples);
    CHECK_INPUT(success);
    CHECK_DEVICE(samples, probs);
    CHECK_DEVICE(success, probs);
    CHECK_DIM(2, probs);
    CHECK_DIM(1, samples);
    CHECK_DIM(1, success);
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32 && samples.scalar_type() == torch::kInt32
                    && success.scalar_type() == torch::kBool,
                "invalid sampling finalization tensor dtype");
    TORCH_CHECK(probs.size(1) > 0 && samples.numel() == probs.size(0) && success.numel() == probs.size(0),
                "sampling finalization shapes must match");
    if (log_probs.has_value()) {
        CHECK_INPUT(log_probs.value());
        CHECK_DEVICE(log_probs.value(), probs);
        TORCH_CHECK(log_probs->scalar_type() == torch::kFloat32 && log_probs->dim() == 1
                        && log_probs->numel() == samples.numel(),
                    "log_probs must have one float per sample");
    }
    if (samples.numel() == 0) {
        return;
    }
    hipLaunchKernelGGL(sampling::FinalizeSamplingProbKernel,
                       dim3(samples.numel()),
                       dim3(256),
                       0,
                       reinterpret_cast<hipStream_t>(stream),
                       probs.data_ptr<float>(),
                       samples.data_ptr<int>(),
                       success.data_ptr<bool>(),
                       log_probs.has_value() ? log_probs->data_ptr<float>() : nullptr,
                       static_cast<uint32_t>(probs.size(1)));
    const auto status = hipGetLastError();
    TORCH_CHECK(status == hipSuccess, "FinalizeSamplingProb failed: ", hipGetErrorString(status));
}

void top_p_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr,
                        double                       top_p_val,
                        uintptr_t                    stream) {
    CHECK_INPUT(probs);
    CHECK_INPUT(renorm_probs);
    CHECK_DEVICE(renorm_probs, probs);
    if (maybe_top_p_arr.has_value()) {
        CHECK_INPUT(maybe_top_p_arr.value());
        CHECK_DEVICE(maybe_top_p_arr.value(), probs);
    }
    CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
    unsigned int batch_size    = probs.sizes()[0];
    unsigned int vocab_size    = probs.sizes()[1];
    bool         has_top_p_arr = maybe_top_p_arr.has_value();

    hipSetDevice(probs.get_device());
    hipError_t status =
        sampling::TopPRenormProb<float>(static_cast<float*>(probs.data_ptr()),
                                        static_cast<float*>(renorm_probs.data_ptr()),
                                        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
                                        batch_size,
                                        top_p_val,
                                        vocab_size,
                                        reinterpret_cast<hipStream_t>(stream));

    TORCH_CHECK(status == hipSuccess,
                "TopPRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr,
                        int64_t                      top_k_val,
                        uintptr_t                    stream) {
    CHECK_INPUT(probs);
    CHECK_INPUT(renorm_probs);
    CHECK_DEVICE(renorm_probs, probs);
    if (maybe_top_k_arr.has_value()) {
        CHECK_INPUT(maybe_top_k_arr.value());
        CHECK_DEVICE(maybe_top_k_arr.value(), probs);
    }
    CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
    unsigned int batch_size    = probs.sizes()[0];
    unsigned int vocab_size    = probs.sizes()[1];
    bool         has_top_k_arr = maybe_top_k_arr.has_value();

    hipSetDevice(probs.get_device());
    hipError_t status =
        sampling::TopKRenormProb<float>(static_cast<float*>(probs.data_ptr()),
                                        static_cast<float*>(renorm_probs.data_ptr()),
                                        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
                                        batch_size,
                                        top_k_val,
                                        vocab_size,
                                        reinterpret_cast<hipStream_t>(stream));

    TORCH_CHECK(status == hipSuccess,
                "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

}  // namespace rtp_llm
