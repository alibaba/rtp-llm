#include "sampling.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>

#include "flashinfer/air_top_p.cuh"
#include "flashinfer/sampling.cuh"

namespace rtp_llm {
namespace {

void check_renorm_inputs(const torch::Tensor& probs, const torch::Tensor& renorm_probs) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(renorm_probs.is_cuda(), "renorm_probs must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor");
    TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor");
    TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "renorm_probs shape must match probs shape");
    TORCH_CHECK(probs.scalar_type() == renorm_probs.scalar_type(), "renorm_probs dtype must match probs dtype");
    TORCH_CHECK(probs.device() == renorm_probs.device(), "renorm_probs device must match probs device");
    TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
    TORCH_CHECK(renorm_probs.is_contiguous(), "renorm_probs must be contiguous");
}

void check_sampling_inputs(const torch::Tensor& probs, const torch::Tensor& output, const torch::Tensor& valid) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(valid.is_cuda(), "valid must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor");
    TORCH_CHECK(output.dim() == 1, "output must be a 1D tensor");
    TORCH_CHECK(valid.dim() == 1, "valid must be a 1D tensor");
    TORCH_CHECK(output.size(0) == valid.size(0), "output and valid must have the same batch size");
    TORCH_CHECK(output.size(0) == probs.size(0), "output batch size must match probs batch size");
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "sampling from probs currently expects float32 probs");
    TORCH_CHECK(output.scalar_type() == at::kInt, "sampling output currently expects int32 dtype");
    TORCH_CHECK(valid.scalar_type() == at::kBool, "valid must be bool dtype");
    TORCH_CHECK(probs.device() == output.device(), "output device must match probs device");
    TORCH_CHECK(probs.device() == valid.device(), "valid device must match probs device");
    TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(valid.is_contiguous(), "valid must be contiguous");
}

torch::Tensor prepare_optional_param(const std::optional<torch::Tensor>& maybe_param,
                                     int64_t                             batch_size,
                                     c10::Device                         device,
                                     c10::ScalarType                     dtype,
                                     const char*                         name) {
    if (!maybe_param.has_value()) {
        return torch::Tensor();
    }

    const auto& param = maybe_param.value();
    TORCH_CHECK(param.dim() == 1, name, " must be a 1D tensor with shape [batch_size]");
    TORCH_CHECK(
        param.numel() == batch_size, name, " length must match batch_size, got ", param.numel(), " vs ", batch_size);

    // Return the input tensor directly when it already matches; callers must clone before any in-place mutation.
    if (param.device() == device && param.scalar_type() == dtype && param.is_contiguous()) {
        return param;
    }
    return param.to(torch::TensorOptions().device(device).dtype(dtype), /*non_blocking=*/true).contiguous();
}

torch::Tensor
prepare_optional_index(const std::optional<torch::Tensor>& maybe_indices, int64_t batch_size, c10::Device device) {
    if (!maybe_indices.has_value()) {
        return torch::Tensor();
    }

    const auto& indices = maybe_indices.value();
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor with shape [batch_size]");
    TORCH_CHECK(indices.numel() == batch_size,
                "indices length must match batch_size, got ",
                indices.numel(),
                " vs ",
                batch_size);
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices currently expects int32 dtype");
    if (indices.device() == device && indices.is_contiguous()) {
        return indices;
    }
    return indices.to(torch::TensorOptions().device(device).dtype(at::kInt), /*non_blocking=*/true).contiguous();
}

cudaStream_t resolve_stream(int64_t cuda_stream) {
    if (cuda_stream == 0) {
        return at::cuda::getCurrentCUDAStream().stream();
    }
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

template<bool IsDeterministic, typename DType>
size_t air_top_p_workspace_size(uint32_t batch_size, uint32_t vocab_size) {
    namespace air = flashinfer::sampling::air_top_p;
    auto align256 = [](size_t x) { return ((x + 255) / 256) * 256; };

    using HistT        = air::HisT<IsDeterministic, DType>;
    const auto buf_len = static_cast<size_t>(air::calcBufLen<DType>(static_cast<air::IdxT>(vocab_size)));

    const size_t counters_size = align256(sizeof(air::Counter<DType>) * batch_size);
    const size_t hist_size     = align256(sizeof(HistT) * air::NUM_BUCKETS * batch_size);
    const size_t count_size    = align256(sizeof(air::IdxT) * air::NUM_BUCKETS * batch_size);
    const size_t buf_size      = align256(sizeof(DType) * buf_len * batch_size);
    return counters_size + hist_size + count_size + 2 * buf_size;
}

void check_cuda_status(cudaError_t status, const char* kernel_name) {
    TORCH_CHECK(status == cudaSuccess, kernel_name, " failed with error code ", cudaGetErrorString(status));
    status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, kernel_name, " launch failed with error code ", cudaGetErrorString(status));
}

}  // namespace

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator) {
    auto gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(generator, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    at::PhiloxCudaState         rng_engine_inputs = gen->philox_cuda_state(increment_size);
    return {rng_engine_inputs.seed_.val, rng_engine_inputs.offset_.val};
}

void top_p_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_p  = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed   = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream = resolve_stream(cuda_stream);

    cudaError_t status = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        batch_size,
        static_cast<float>(top_p_val),
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopPSamplingFromProb");
}

void top_k_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr,
                               int64_t                      top_k_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                top_k = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto                seed  = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto       offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto       stream = resolve_stream(cuda_stream);
    const auto top_k_limit =
        static_cast<uint32_t>(top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, vocab_size));

    cudaError_t status = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        top_k.defined() ? static_cast<int32_t*>(top_k.data_ptr()) : nullptr,
        batch_size,
        top_k_limit,
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopKSamplingFromProb");
}

void top_k_top_p_sampling_from_probs(torch::Tensor                probs,
                                     torch::Tensor                output,
                                     torch::Tensor                valid,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                     int64_t                      top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                     double                       top_p_val,
                                     bool                         deterministic,
                                     std::optional<torch::Tensor> maybe_seed_arr,
                                     uint64_t                     seed_val,
                                     std::optional<torch::Tensor> maybe_offset_arr,
                                     uint64_t                     offset_val,
                                     int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                top_k = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto       top_p  = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto       seed   = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto       offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto       stream = resolve_stream(cuda_stream);
    const auto top_k_limit =
        static_cast<int32_t>(top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, vocab_size));

    cudaError_t status = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        top_k.defined() ? static_cast<int32_t*>(top_k.data_ptr()) : nullptr,
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        batch_size,
        top_k_limit,
        static_cast<float>(top_p_val),
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopKTopPSamplingFromProb");
}

void top_p_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr,
                        double                       top_p_val,
                        int64_t                      cuda_stream) {
    check_renorm_inputs(probs, renorm_probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "top_p_renorm_probs currently expects float32 probs");

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(probs.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto   top_p     = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto   stream    = resolve_stream(cuda_stream);
    float* top_p_ptr = top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr;

    cudaError_t status = cudaSuccess;
    if (vocab_size < flashinfer::sampling::air_top_p::NUM_BUCKETS) {
        status = flashinfer::sampling::TopPRenormProb<float>(static_cast<float*>(probs.data_ptr()),
                                                             static_cast<float*>(renorm_probs.data_ptr()),
                                                             top_p_ptr,
                                                             batch_size,
                                                             static_cast<float>(top_p_val),
                                                             vocab_size,
                                                             stream);
        check_cuda_status(status, "TopPRenormProb");
        return;
    }

    constexpr bool deterministic  = false;
    const auto     workspace_size = air_top_p_workspace_size<deterministic, float>(batch_size, vocab_size);
    auto           workspace = torch::empty({static_cast<int64_t>(workspace_size)}, probs.options().dtype(at::kByte));

    status = flashinfer::sampling::air_top_p::AirTopPRenormProb<deterministic, float>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(renorm_probs.data_ptr()),
        top_p_ptr,
        batch_size,
        static_cast<float>(top_p_val),
        vocab_size,
        workspace.data_ptr(),
        stream);
    check_cuda_status(status, "AirTopPRenormProb");
}

void top_k_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr,
                        int64_t                      top_k_val,
                        int64_t                      cuda_stream) {
    check_renorm_inputs(probs, renorm_probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat || probs.scalar_type() == at::kHalf
                    || probs.scalar_type() == at::kBFloat16,
                "top_k_renorm_probs expects float32, float16, or bfloat16 probs");

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(probs.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                top_k  = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto                stream = resolve_stream(cuda_stream);
    int*                top_k_ptr   = top_k.defined() ? static_cast<int*>(top_k.data_ptr()) : nullptr;
    const auto          top_k_limit = static_cast<uint32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, std::numeric_limits<uint32_t>::max()));

    auto  row_states     = torch::zeros({1024 * 1024}, probs.options().dtype(at::kByte));
    auto* row_states_ptr = reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states.data_ptr());

    cudaError_t status = cudaSuccess;
    switch (probs.scalar_type()) {
        case at::kFloat:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<float, int>(
                static_cast<float*>(probs.data_ptr()),
                static_cast<float*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        case at::kHalf:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<half, int>(
                reinterpret_cast<half*>(probs.data_ptr()),
                reinterpret_cast<half*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        case at::kBFloat16:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<nv_bfloat16, int>(
                reinterpret_cast<nv_bfloat16*>(probs.data_ptr()),
                reinterpret_cast<nv_bfloat16*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for top_k_renorm_probs");
    }
    check_cuda_status(status, "RadixTopKRenormProbMultiCTA");
}

}  // namespace rtp_llm
