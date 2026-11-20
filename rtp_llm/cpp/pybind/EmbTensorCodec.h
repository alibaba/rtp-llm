#pragma once

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace rtp_llm {

// Layout and key validation happen in the object codec before this call.
using EmbTensorSlice = std::tuple<int64_t, int64_t, at::ScalarType, std::vector<int64_t>, at::Device>;

using EmbTensorInput = std::tuple<int64_t, int64_t, at::Tensor>;

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

inline void registerEmbTensorCodec(pybind11::module& module) {
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
