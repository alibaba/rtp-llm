#pragma once

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace rtp_llm {

// Layout and key validation happen in the object codec before this call.
using EmbTensorSlice = std::tuple<int64_t, int64_t, at::ScalarType, std::vector<int64_t>, at::Device>;

using EmbTensorInput = std::tuple<int64_t, int64_t, at::Tensor>;

inline std::vector<at::Tensor> snapshotEmbTensors(const std::vector<at::Tensor>& inputs) {
    std::vector<at::Tensor> snapshots;
    snapshots.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (input.is_cuda()) {
            const c10::impl::VirtualGuardImpl guard(input.device().type());
            input.record_stream(guard.getStream(input.device()));
        }
        snapshots.push_back(input.detach().clone(at::MemoryFormat::Contiguous));
    }
    return snapshots;
}

inline void packEmbTensorSlices(const at::Tensor& output,
                                std::string prefix,
                                int64_t payloadOffset,
                                const std::vector<EmbTensorInput>& inputs) {
    TORCH_CHECK(output.dim() == 1 && output.is_contiguous() && output.scalar_type() == at::kByte,
                "expected a contiguous byte buffer");
    const auto size = output.numel();
    TORCH_CHECK(payloadOffset >= 0 && payloadOffset <= size
                    && prefix.size() <= static_cast<size_t>(payloadOffset),
                "invalid EMB prefix extent");
    int64_t end = payloadOffset;
    for (const auto& [offset, bytes, input] : inputs) {
        TORCH_CHECK(offset >= end && offset <= size && bytes >= 0 && bytes <= size - offset,
                    "invalid EMB tensor extent");
        TORCH_CHECK(bytes % input.element_size() == 0 && input.numel() == bytes / input.element_size(),
                    "invalid EMB tensor size");
        end = offset + bytes;
    }
    TORCH_CHECK(end == size, "invalid EMB packed object size");

    if (output.is_cuda() && payloadOffset > 0) {
        auto header = at::empty({payloadOffset}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
        auto* bytes = header.data_ptr<uint8_t>();
        std::memcpy(bytes, prefix.data(), prefix.size());
        std::memset(bytes + prefix.size(), 0, payloadOffset - prefix.size());
        // The CUDA copy kernel records the source's pinned allocation on the
        // copy stream. The caller waits for the stream before handing off to PACE.
        output.narrow(0, 0, payloadOffset).copy_(header, /*non_blocking=*/true);
    } else if (!prefix.empty()) {
        const auto length = static_cast<int64_t>(prefix.size());
        auto header = at::from_blob(prefix.data(), {length}, at::TensorOptions().dtype(at::kByte));
        output.narrow(0, 0, length).copy_(header);
        output.narrow(0, length, payloadOffset - length).zero_();
    } else {
        output.narrow(0, 0, payloadOffset).zero_();
    }
    end = payloadOffset;
    for (const auto& [offset, bytes, input] : inputs) {
        if (offset > end) {
            output.narrow(0, end, offset - end).zero_();
        }
        end = offset + bytes;
        if (bytes == 0) {
            continue;
        }
        if (input.is_cuda() && output.is_cuda()) {
            const c10::impl::VirtualGuardImpl guard(input.device().type());
            input.record_stream(guard.getStream(input.device()));
        }
        auto dst = output.narrow(0, offset, bytes).view(input.scalar_type()).reshape(input.sizes());
        dst.copy_(input.detach(), /*non_blocking=*/output.is_cuda());
    }
}

#if USING_CUDA
inline at::Tensor readEmbCudaHeader(const at::Tensor& buffer, int64_t offset, int64_t size) {
    TORCH_CHECK(buffer.is_cuda() && buffer.dim() == 1 && buffer.is_contiguous() && buffer.scalar_type() == at::kByte,
                "expected a contiguous CUDA byte buffer");
    TORCH_CHECK(offset >= 0 && offset <= buffer.numel() && size > 0 && size <= buffer.numel() - offset,
                "invalid EMB header extent");
    const c10::cuda::CUDAGuard deviceGuard(buffer.device());
    auto header = at::empty({size}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    header.copy_(buffer.narrow(0, offset, size), /*non_blocking=*/true);
    at::cuda::CUDAEvent ready(cudaEventBlockingSync | cudaEventDisableTiming);
    ready.record(c10::cuda::getCurrentCUDAStream(buffer.get_device()));
    ready.synchronize();
    return header;
}
#endif

// Own routing hashes without keeping Python tensor wrappers or lists alive.
// The batch is immutable and readers receive independent CPU tensor copies.
class EmbHashTensorBatch {
public:
    explicit EmbHashTensorBatch(const std::vector<at::Tensor>& inputs) {
        hashes_.reserve(inputs.size());
        for (const auto& input : inputs) {
            hashes_.push_back(input.detach().to(input.options().dtype(at::kInt).device(at::kCPU),
                                                 /*non_blocking=*/false,
                                                 /*copy=*/true));
        }
    }

    std::vector<at::Tensor> cloneHashes() const {
        std::vector<at::Tensor> copies;
        copies.reserve(hashes_.size());
        for (const auto& hash : hashes_) {
            copies.push_back(hash.clone());
        }
        return copies;
    }

private:
    std::vector<at::Tensor> hashes_;
};

inline constexpr char kEmbHashBatchName[] = "rtp_llm.EmbHashTensorBatch";

inline pybind11::capsule storeEmbHashes(const std::vector<at::Tensor>& inputs) {
    std::unique_ptr<EmbHashTensorBatch> batch;
    {
        pybind11::gil_scoped_release release;
        batch = std::make_unique<EmbHashTensorBatch>(inputs);
    }
    pybind11::capsule handle(batch.get(), kEmbHashBatchName, [](void* pointer) {
        delete static_cast<EmbHashTensorBatch*>(pointer);
    });
    batch.release();
    return handle;
}

inline std::vector<at::Tensor> cloneEmbHashes(const pybind11::capsule& handle) {
    auto* batch = static_cast<EmbHashTensorBatch*>(PyCapsule_GetPointer(handle.ptr(), kEmbHashBatchName));
    if (batch == nullptr) {
        throw pybind11::error_already_set();
    }
    pybind11::gil_scoped_release release;
    return batch->cloneHashes();
}

inline void registerEmbTensorCodec(pybind11::module& module) {
    module.def("snapshot_emb_tensors", &snapshotEmbTensors, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("store_emb_hashes", &storeEmbHashes);
    module.def("clone_emb_hashes", &cloneEmbHashes);
#if USING_CUDA
    module.def("read_emb_cuda_header", &readEmbCudaHeader, pybind11::call_guard<pybind11::gil_scoped_release>());
#endif
    module.def("pack_emb_tensor_slices", &packEmbTensorSlices, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "restore_emb_tensor_slices",
        [](const at::Tensor& buffer, const std::vector<EmbTensorSlice>& slices) {
            TORCH_CHECK(buffer.dim() == 1 && buffer.is_contiguous() && buffer.scalar_type() == at::kByte,
                        "expected a contiguous byte buffer");
            for (const auto& slice : slices) {
                const auto offset = std::get<0>(slice);
                const auto bytes = std::get<1>(slice);
                TORCH_CHECK(offset >= 0 && bytes >= 0 && offset <= buffer.numel() && bytes <= buffer.numel() - offset,
                            "invalid EMB tensor slice");
            }

            std::vector<at::Tensor> result;
            result.reserve(slices.size());
            for (const auto& [offset, bytes, dtype, shape, device] : slices) {
                const auto options = buffer.options().dtype(dtype).device(device);
                if (bytes == 0) {
                    result.push_back(at::empty(shape, options));
                } else {
                    const auto view = buffer.narrow(0, offset, bytes).view(dtype).reshape(shape);
                    result.push_back(view.to(options, /*non_blocking=*/true, /*copy=*/true));
                }
            }
            return result;
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
}

}  // namespace rtp_llm
