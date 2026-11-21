#pragma once

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include "rtp_llm/cpp/pybind/EmbTransportCudaStreams.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
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
class EmbCudaSnapshotReady {
public:
    EmbCudaSnapshotReady(c10::cuda::CUDAStream producer, c10::cuda::CUDAStream caller, bool blocking):
        event_(cudaEventDisableTiming | (blocking ? cudaEventBlockingSync : 0)) {
        event_.record(producer);
        if (producer != caller) {
            event_.block(caller);
        }
    }

    void synchronize() const {
        event_.synchronize();
    }

private:
    at::cuda::CUDAEvent event_;
};

inline auto snapshotEmbCudaTensors(const std::vector<at::Tensor>& inputs,
                                   std::optional<c10::StreamId> producerId,
                                   bool blocking) {
    TORCH_CHECK(!inputs.empty() && inputs.front().is_cuda(), "expected CUDA EMB tensors");
    const auto device = inputs.front().device();
    for (const auto& input : inputs) {
        TORCH_CHECK(input.device() == device, "expected CUDA EMB tensors on one device");
    }
    const auto caller = c10::cuda::getCurrentCUDAStream(device.index());
    const auto producer = producerId
        ? c10::cuda::CUDAStream(c10::Stream::unpack3(*producerId, device.index(), c10::kCUDA))
        : caller;
    // Include caller-side preparation, then order reuse on either stream
    // after cloning without blocking the submitting thread.
    if (producer != caller) {
        at::cuda::CUDAEvent dependency(cudaEventDisableTiming);
        dependency.record(caller);
        dependency.block(producer);
    }
    const c10::cuda::CUDAStreamGuard streamGuard(producer);
    auto snapshots = snapshotEmbTensors(inputs);
    auto ready = std::make_unique<EmbCudaSnapshotReady>(producer, caller, blocking);
    return std::make_pair(std::move(snapshots), std::move(ready));
}

inline void waitEmbCudaStream(const at::Tensor& buffer) {
    at::cuda::CUDAEvent ready(cudaEventBlockingSync | cudaEventDisableTiming);
    ready.record(c10::cuda::getCurrentCUDAStream(buffer.get_device()));
    ready.synchronize();
}

inline at::Tensor readEmbCudaHeader(const at::Tensor& buffer, int64_t offset, int64_t size) {
    TORCH_CHECK(buffer.is_cuda() && buffer.dim() == 1 && buffer.is_contiguous() && buffer.scalar_type() == at::kByte,
                "expected a contiguous CUDA byte buffer");
    TORCH_CHECK(offset >= 0 && offset <= buffer.numel() && size > 0 && size <= buffer.numel() - offset,
                "invalid EMB header extent");
    const c10::cuda::CUDAGuard deviceGuard(buffer.device());
    auto header = at::empty({size}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    header.copy_(buffer.narrow(0, offset, size), /*non_blocking=*/true);
    waitEmbCudaStream(buffer);
    return header;
}

inline std::string readEmbCudaObjectHeader(const at::Tensor& buffer) {
    constexpr int64_t prefixSize = 16;
    constexpr int64_t firstReadSize = 1024;
    constexpr int64_t alignment = 64;
    constexpr uint64_t maxHeaderSize = 64 * 1024;
    const auto size = buffer.numel();
    if (size < prefixSize) {
        throw std::invalid_argument("invalid multimodal object buffer");
    }
    const auto first = readEmbCudaHeader(buffer, 0, std::min(size, firstReadSize));
    const auto* bytes = first.data_ptr<uint8_t>();
    uint64_t headerSize = 0;
    for (unsigned i = 0; i < 8; ++i) {
        headerSize |= static_cast<uint64_t>(bytes[8 + i]) << (8 * i);
    }
    if (std::memcmp(bytes, "RTPMM002", 8) != 0 || headerSize > maxHeaderSize) {
        throw std::invalid_argument("invalid multimodal object header");
    }
    const auto end = prefixSize + static_cast<int64_t>(headerSize);
    if ((end + alignment - 1) / alignment * alignment > size) {
        throw std::invalid_argument("invalid multimodal object header");
    }
    const auto initial = std::min(end, first.numel());
    std::string header(reinterpret_cast<const char*>(bytes + prefixSize), initial - prefixSize);
    if (end > initial) {
        const auto suffix = readEmbCudaHeader(buffer, initial, end - initial);
        header.append(reinterpret_cast<const char*>(suffix.data_ptr<uint8_t>()), end - initial);
    }
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

inline std::vector<at::Tensor> restoreEmbTensorSlices(const at::Tensor& buffer,
                                                     const std::vector<EmbTensorSlice>& slices) {
    TORCH_CHECK(buffer.dim() == 1 && buffer.is_contiguous() && buffer.scalar_type() == at::kByte,
                "expected a contiguous byte buffer");
    for (const auto& [offset, bytes, dtype, shape, device] : slices) {
        TORCH_CHECK(offset >= 0 && bytes >= 0 && offset <= buffer.numel() && bytes <= buffer.numel() - offset,
                    "invalid EMB tensor slice");
    }

    std::vector<at::Tensor> result;
    result.reserve(slices.size());
#if USING_CUDA
    const auto copyStream = buffer.is_cuda()
        ? std::optional<c10::cuda::CUDAStream>(c10::cuda::getCurrentCUDAStream(buffer.get_device()))
        : std::nullopt;
    const auto allocationStream = copyStream ? EmbTransportCudaStreams::instance().allocationFor(*copyStream)
                                             : std::nullopt;
    if (allocationStream) {
        // One allocator stream avoids building a separate caching-allocator
        // free list for each concurrent transport stream. It submits allocations
        // only, so the copy stream never waits behind encoder compute here.
        at::cuda::CUDAEvent ready(cudaEventDisableTiming);
        {
            const c10::cuda::CUDAStreamGuard guard(*allocationStream);
            for (const auto& [offset, bytes, dtype, shape, device] : slices) {
                if (bytes > 0 && device == buffer.device()) {
                    result.push_back(at::empty(shape, buffer.options().dtype(dtype).device(device)));
                } else {
                    result.emplace_back();
                }
            }
            ready.record(*allocationStream);
        }
        ready.block(*copyStream);
    } else
#endif
    {
        result.resize(slices.size());
    }
    size_t index = 0;
    for (const auto& [offset, bytes, dtype, shape, device] : slices) {
        auto& output = result[index++];
        const auto options = buffer.options().dtype(dtype).device(device);
        if (bytes == 0) {
            output = at::empty(shape, options);
        } else {
            const auto view = buffer.narrow(0, offset, bytes).view(dtype).reshape(shape);
#if USING_CUDA
            if (output.defined()) {
                output.record_stream(*copyStream);
                output.copy_(view, /*non_blocking=*/true);
                continue;
            }
#endif
            output = view.to(options, /*non_blocking=*/true, /*copy=*/true);
        }
    }
    return result;
}

inline void registerEmbTensorCodec(pybind11::module& module) {
    module.def("snapshot_emb_tensors", &snapshotEmbTensors, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("store_emb_hashes", &storeEmbHashes);
    module.def("clone_emb_hashes", &cloneEmbHashes);
#if USING_CUDA
    pybind11::class_<EmbTransportCudaStream>(module, "EmbTransportCudaStream")
        .def(pybind11::init<c10::DeviceIndex>(), pybind11::call_guard<pybind11::gil_scoped_release>())
        .def_property_readonly("address", &EmbTransportCudaStream::address);
    pybind11::class_<EmbCudaSnapshotReady>(module, "EmbCudaSnapshotReady")
        .def("synchronize", &EmbCudaSnapshotReady::synchronize, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("snapshot_emb_cuda_tensors",
               &snapshotEmbCudaTensors,
               pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("read_emb_cuda_header", &readEmbCudaHeader, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("read_emb_cuda_object_header", [](const at::Tensor& buffer) {
        std::string header;
        {
            pybind11::gil_scoped_release release;
            header = readEmbCudaObjectHeader(buffer);
        }
        return pybind11::bytes(header);
    });
    module.def(
        "pack_emb_cuda_tensor_slices_and_wait",
        [](const at::Tensor& output,
           std::string prefix,
           int64_t payloadOffset,
           const std::vector<EmbTensorInput>& inputs) {
            TORCH_CHECK(output.is_cuda(), "expected a CUDA EMB destination");
            const c10::cuda::CUDAGuard deviceGuard(output.device());
            packEmbTensorSlices(output, std::move(prefix), payloadOffset, inputs);
            waitEmbCudaStream(output);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "restore_emb_cuda_tensor_slices_and_wait",
        [](const at::Tensor& buffer, const std::vector<EmbTensorSlice>& slices) {
            TORCH_CHECK(buffer.is_cuda(), "expected a CUDA EMB source");
            for (const auto& slice : slices) {
                const auto device = std::get<4>(slice);
                TORCH_CHECK(device.is_cpu() || (device.is_cuda() && (!device.has_index() || device == buffer.device())),
                            "invalid fenced EMB tensor destination");
            }
            const c10::cuda::CUDAGuard deviceGuard(buffer.device());
            auto result = restoreEmbTensorSlices(buffer, slices);
            waitEmbCudaStream(buffer);
            return result;
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
#endif
    module.def("pack_emb_tensor_slices", &packEmbTensorSlices, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("restore_emb_tensor_slices", &restoreEmbTensorSlices, pybind11::call_guard<pybind11::gil_scoped_release>());
}

}  // namespace rtp_llm
