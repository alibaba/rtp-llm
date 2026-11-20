#pragma once

#include <ATen/ops/empty.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace rtp_llm {

// Layout and key validation happen in the object codec before this call.
using EmbTensorSlice = std::tuple<int64_t, int64_t, at::ScalarType, std::vector<int64_t>, at::Device>;

inline void registerEmbTensorRestore(pybind11::module& module) {
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
