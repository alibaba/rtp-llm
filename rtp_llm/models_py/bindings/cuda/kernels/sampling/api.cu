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
#include <vector>

#include "fast_renorm.cuh"
#include "flashinfer/sampling.cuh"

namespace rtp_llm {
namespace {

struct FastRenormWorkspaceKey {
    int      device_id;
    uint64_t stream_id;

    bool operator==(const FastRenormWorkspaceKey& other) const {
        return device_id == other.device_id && stream_id == other.stream_id;
    }
};

struct FastRenormWorkspaceEntry {
    FastRenormWorkspaceKey key;
    torch::Tensor          buffer;
    uint64_t               last_used_seq = 0;
};

class FastRenormWorkspaceCache {
public:
    static constexpr size_t kMaxEntriesPerDevice = 4;

    torch::Tensor
    acquire(int device_id, uint64_t stream_id, int64_t required_bytes, const torch::TensorOptions& options) {
        std::lock_guard<std::mutex>  lock(mutex_);
        const FastRenormWorkspaceKey key{device_id, stream_id};
        const uint64_t               seq = ++seq_counter_;

        for (auto& entry : entries_) {
            if (entry.key == key) {
                entry.last_used_seq = seq;
                if (entry.buffer.numel() < required_bytes) {
                    entry.buffer = torch::empty({required_bytes}, options);
                }
                return entry.buffer;
            }
        }

        size_t   device_count = 0;
        size_t   lru_idx      = entries_.size();
        uint64_t lru_seq      = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].key.device_id == device_id) {
                ++device_count;
                if (entries_[i].last_used_seq < lru_seq) {
                    lru_seq = entries_[i].last_used_seq;
                    lru_idx = i;
                }
            }
        }
        if (device_count >= kMaxEntriesPerDevice && lru_idx < entries_.size()) {
            entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(lru_idx));
        }

        auto buffer = torch::empty({required_bytes}, options);
        entries_.push_back(FastRenormWorkspaceEntry{key, buffer, seq});
        return buffer;
    }

private:
    std::mutex                            mutex_;
    std::vector<FastRenormWorkspaceEntry> entries_;
    uint64_t                              seq_counter_ = 0;
};

torch::Tensor get_fast_renorm_workspace(const c10::Device& device, cudaStream_t stream, size_t required_bytes) {
    if (required_bytes == 0) {
        return torch::Tensor();
    }
    const auto              options        = torch::TensorOptions().dtype(at::kByte).device(device);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long      stream_id      = 0;
    if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess && capture_status == cudaStreamCaptureStatusNone
        && cudaStreamGetId(stream, &stream_id) == cudaSuccess) {
        static auto* cache = new FastRenormWorkspaceCache();
        return cache->acquire(
            device.index(), static_cast<uint64_t>(stream_id), static_cast<int64_t>(required_bytes), options);
    }
    return torch::empty({static_cast<int64_t>(required_bytes)}, options);
}

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

void check_cuda_status(cudaError_t status, const char* kernel_name) {
    TORCH_CHECK(status == cudaSuccess, kernel_name, " failed with error code ", cudaGetErrorString(status));
    status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, kernel_name, " launch failed with error code ", cudaGetErrorString(status));
}

bool is_unfiltered_cpu_top_p(const std::optional<torch::Tensor>& maybe_top_p_arr,
                             uint32_t                            batch_size,
                             double                              top_p_val) {
    if (!maybe_top_p_arr.has_value()) {
        return top_p_val >= 1.0;
    }
    const auto& t = maybe_top_p_arr.value();
    if (!t.is_cpu() || t.dim() != 1 || t.numel() != batch_size || t.scalar_type() != at::kFloat || !t.is_contiguous()) {
        return false;
    }
    const float* ptr = t.data_ptr<float>();
    return std::all_of(ptr, ptr + batch_size, [](float v) { return v >= 1.0f; });
}

bool is_unfiltered_cpu_top_k(const std::optional<torch::Tensor>& maybe_top_k_arr,
                             uint32_t                            batch_size,
                             uint32_t                            top_k_limit,
                             uint32_t                            vocab_size) {
    if (!maybe_top_k_arr.has_value()) {
        return top_k_limit >= vocab_size;
    }
    const auto& t = maybe_top_k_arr.value();
    if (!t.is_cpu() || t.dim() != 1 || t.numel() != batch_size || t.scalar_type() != at::kInt || !t.is_contiguous()) {
        return false;
    }
    const int32_t* ptr = t.data_ptr<int32_t>();
    return std::all_of(
        ptr, ptr + batch_size, [vocab_size](int32_t v) { return v > 0 && static_cast<uint32_t>(v) >= vocab_size; });
}

}  // namespace

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator) {
    auto gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(generator, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    at::PhiloxCudaState         rng_engine_inputs = gen->philox_cuda_state(increment_size);
    return {rng_engine_inputs.seed_.val, rng_engine_inputs.offset_.val};
}

void sampling_from_probs(torch::Tensor                probs,
                         torch::Tensor                output,
                         torch::Tensor                valid,
                         std::optional<torch::Tensor> maybe_indices,
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
    auto                seed = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto        offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto        stream = resolve_stream(cuda_stream);
    cudaError_t status = flashinfer::sampling::SamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        batch_size,
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "SamplingFromProb");
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
    if (is_unfiltered_cpu_top_p(maybe_top_p_arr, batch_size, top_p_val)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_p   = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<false, false, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                nullptr,
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                nullptr,
                0,
                top_p.defined() ? static_cast<const float*>(top_p.data_ptr()) : nullptr,
                static_cast<float>(top_p_val),
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopPSamplingFromProb");
            return;
        }
    }

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
    const auto          batch_size  = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size  = static_cast<uint32_t>(probs.size(1));
    const auto          top_k_limit = static_cast<uint32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
    if (is_unfiltered_cpu_top_k(maybe_top_k_arr, batch_size, top_k_limit, vocab_size)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_k   = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<true, false, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                nullptr,
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                top_k.defined() ? static_cast<const int32_t*>(top_k.data_ptr()) : nullptr,
                static_cast<int32_t>(top_k_limit),
                nullptr,
                1.0f,
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopKSamplingFromProb");
            return;
        }
    }

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
    const auto          batch_size  = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size  = static_cast<uint32_t>(probs.size(1));
    const auto          top_k_limit = static_cast<int32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
    if (is_unfiltered_cpu_top_k(maybe_top_k_arr, batch_size, static_cast<uint32_t>(top_k_limit), vocab_size)
        && is_unfiltered_cpu_top_p(maybe_top_p_arr, batch_size, top_p_val)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_k   = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto top_p   = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<true, false, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                nullptr,
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                top_k.defined() ? static_cast<const int32_t*>(top_k.data_ptr()) : nullptr,
                top_k_limit,
                top_p.defined() ? static_cast<const float*>(top_p.data_ptr()) : nullptr,
                static_cast<float>(top_p_val),
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopKTopPSamplingFromProb");
            return;
        }
    }

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
    auto         top_p     = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto         stream    = resolve_stream(cuda_stream);
    float*       top_p_ptr = top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr;
    const size_t ws_bytes  = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
    auto         ws_t      = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);

    cudaError_t status = rtp_llm::fast_renorm::FastRenormProb<false>(static_cast<const float*>(probs.data_ptr()),
                                                                     static_cast<float*>(renorm_probs.data_ptr()),
                                                                     nullptr,
                                                                     0,
                                                                     top_p_ptr,
                                                                     static_cast<float>(top_p_val),
                                                                     batch_size,
                                                                     vocab_size,
                                                                     ws_t.defined() ? ws_t.data_ptr() : nullptr,
                                                                     ws_bytes,
                                                                     stream);
    check_cuda_status(status, "FastTopPRenormProb");
}

void top_k_top_p_renorm_probs(torch::Tensor                probs,
                              torch::Tensor                renorm_probs,
                              std::optional<torch::Tensor> maybe_top_k_arr,
                              int64_t                      top_k_val,
                              std::optional<torch::Tensor> maybe_top_p_arr,
                              double                       top_p_val,
                              int64_t                      cuda_stream) {
    check_renorm_inputs(probs, renorm_probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "top_k_top_p_renorm_probs currently expects float32 probs");

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(probs.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                top_k = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto          top_p  = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto          stream = resolve_stream(cuda_stream);
    const int32_t top_k_limit = static_cast<int32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
    const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
    auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);

    cudaError_t status = rtp_llm::fast_renorm::FastRenormProb<true>(
        static_cast<const float*>(probs.data_ptr()),
        static_cast<float*>(renorm_probs.data_ptr()),
        top_k.defined() ? static_cast<const int32_t*>(top_k.data_ptr()) : nullptr,
        top_k_limit,
        top_p.defined() ? static_cast<const float*>(top_p.data_ptr()) : nullptr,
        static_cast<float>(top_p_val),
        batch_size,
        vocab_size,
        ws_t.defined() ? ws_t.data_ptr() : nullptr,
        ws_bytes,
        stream);
    check_cuda_status(status, "FusedTopKTopPRenormProb");
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
    int*                top_k_ptr = top_k.defined() ? static_cast<int*>(top_k.data_ptr()) : nullptr;

    if (probs.scalar_type() == at::kFloat) {
        const int32_t top_k_limit_i32 = static_cast<int32_t>(
            top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        cudaError_t  status   = rtp_llm::fast_renorm::FastRenormProb<true>(static_cast<const float*>(probs.data_ptr()),
                                                                        static_cast<float*>(renorm_probs.data_ptr()),
                                                                        top_k_ptr,
                                                                        top_k_limit_i32,
                                                                        nullptr,
                                                                        1.0f,
                                                                        batch_size,
                                                                        vocab_size,
                                                                        ws_t.defined() ? ws_t.data_ptr() : nullptr,
                                                                        ws_bytes,
                                                                        stream);
        check_cuda_status(status, "FastTopKRenormProb");
        return;
    }

    const auto top_k_limit = static_cast<uint32_t>(
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

void top_p_renorm_and_sample_probs(torch::Tensor                probs,
                                   torch::Tensor                renorm_probs,
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
    check_renorm_inputs(probs, renorm_probs);
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    if (is_unfiltered_cpu_top_p(maybe_top_p_arr, batch_size, top_p_val)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        top_p_renorm_probs(probs, renorm_probs, maybe_top_p_arr, top_p_val, cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_p   = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && probs.data_ptr() != renorm_probs.data_ptr()
        && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<false, true, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                static_cast<float*>(renorm_probs.data_ptr()),
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                nullptr,
                0,
                top_p.defined() ? static_cast<const float*>(top_p.data_ptr()) : nullptr,
                static_cast<float>(top_p_val),
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopPRenormAndSampleProb");
            return;
        }
    }

    top_p_sampling_from_probs(probs,
                              output,
                              valid,
                              maybe_indices,
                              maybe_top_p_arr,
                              top_p_val,
                              deterministic,
                              maybe_seed_arr,
                              seed_val,
                              maybe_offset_arr,
                              offset_val,
                              cuda_stream);
    top_p_renorm_probs(probs, renorm_probs, maybe_top_p_arr, top_p_val, cuda_stream);
}

void top_k_renorm_and_sample_probs(torch::Tensor                probs,
                                   torch::Tensor                renorm_probs,
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
    check_renorm_inputs(probs, renorm_probs);
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size  = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size  = static_cast<uint32_t>(probs.size(1));
    const int32_t       top_k_limit = static_cast<int32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
    if (is_unfiltered_cpu_top_k(maybe_top_k_arr, batch_size, static_cast<uint32_t>(top_k_limit), vocab_size)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        top_k_renorm_probs(probs, renorm_probs, maybe_top_k_arr, top_k_val, cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_k   = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && probs.data_ptr() != renorm_probs.data_ptr()
        && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<true, true, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                static_cast<float*>(renorm_probs.data_ptr()),
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                top_k.defined() ? static_cast<const int32_t*>(top_k.data_ptr()) : nullptr,
                top_k_limit,
                nullptr,
                1.0f,
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopKRenormAndSampleProb");
            return;
        }
    }

    top_k_sampling_from_probs(probs,
                              output,
                              valid,
                              maybe_indices,
                              maybe_top_k_arr,
                              top_k_val,
                              deterministic,
                              maybe_seed_arr,
                              seed_val,
                              maybe_offset_arr,
                              offset_val,
                              cuda_stream);
    top_k_renorm_probs(probs, renorm_probs, maybe_top_k_arr, top_k_val, cuda_stream);
}

void top_k_top_p_renorm_and_sample_probs(torch::Tensor                probs,
                                         torch::Tensor                renorm_probs,
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
    check_renorm_inputs(probs, renorm_probs);
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size  = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size  = static_cast<uint32_t>(probs.size(1));
    const auto          top_k_limit = static_cast<int32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, static_cast<int64_t>(vocab_size)));
    if (is_unfiltered_cpu_top_k(maybe_top_k_arr, batch_size, static_cast<uint32_t>(top_k_limit), vocab_size)
        && is_unfiltered_cpu_top_p(maybe_top_p_arr, batch_size, top_p_val)) {
        sampling_from_probs(probs,
                            output,
                            valid,
                            maybe_indices,
                            deterministic,
                            maybe_seed_arr,
                            seed_val,
                            maybe_offset_arr,
                            offset_val,
                            cuda_stream);
        top_k_top_p_renorm_probs(
            probs, renorm_probs, maybe_top_k_arr, top_k_val, maybe_top_p_arr, top_p_val, cuda_stream);
        return;
    }
    auto indices = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_k   = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto top_p   = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed    = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset  = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream  = resolve_stream(cuda_stream);

    if (!indices.defined() && probs.data_ptr() != renorm_probs.data_ptr()
        && rtp_llm::fast_renorm::FastRenormCTAsPerRow(batch_size, vocab_size) > 1u
        && rtp_llm::fast_renorm::FastSampleSupportsVocab(vocab_size)) {
        const size_t ws_bytes = rtp_llm::fast_renorm::FastRenormWorkspaceBytes(batch_size, vocab_size);
        auto         ws_t     = get_fast_renorm_workspace(probs.device(), stream, ws_bytes);
        if (ws_t.defined() && static_cast<size_t>(ws_t.numel()) >= ws_bytes) {
            cudaError_t status = rtp_llm::fast_renorm::FastRenormOrSampleProb<true, true, int32_t>(
                static_cast<const float*>(probs.data_ptr()),
                static_cast<float*>(renorm_probs.data_ptr()),
                static_cast<int32_t*>(output.data_ptr()),
                static_cast<bool*>(valid.data_ptr()),
                top_k.defined() ? static_cast<const int32_t*>(top_k.data_ptr()) : nullptr,
                top_k_limit,
                top_p.defined() ? static_cast<const float*>(top_p.data_ptr()) : nullptr,
                static_cast<float>(top_p_val),
                batch_size,
                vocab_size,
                seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
                seed_val,
                offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
                offset_val,
                ws_t.data_ptr(),
                ws_bytes,
                stream);
            check_cuda_status(status, "FastTopKTopPRenormAndSampleProb");
            return;
        }
    }

    top_k_top_p_sampling_from_probs(probs,
                                    output,
                                    valid,
                                    maybe_indices,
                                    maybe_top_k_arr,
                                    top_k_val,
                                    maybe_top_p_arr,
                                    top_p_val,
                                    deterministic,
                                    maybe_seed_arr,
                                    seed_val,
                                    maybe_offset_arr,
                                    offset_val,
                                    cuda_stream);
    top_k_top_p_renorm_probs(probs, renorm_probs, maybe_top_k_arr, top_k_val, maybe_top_p_arr, top_p_val, cuda_stream);
}

}  // namespace rtp_llm
