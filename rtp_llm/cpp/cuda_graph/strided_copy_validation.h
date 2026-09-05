#pragma once

#include <string>
#include <utility>

#include <torch/torch.h>

namespace rtp_llm {

inline bool isStridedCopyCompatible(const torch::Tensor& src, const torch::Tensor& dst, std::string* reason = nullptr) {
    if (reason != nullptr) {
        reason->clear();
    }
    const auto reject = [reason](std::string message) {
        if (reason != nullptr) {
            *reason = std::move(message);
        }
        return false;
    };

    if (!src.defined() || src.numel() == 0) {
        return true;
    }
    if (!dst.defined()) {
        return reject("destination is undefined");
    }
    if (src.scalar_type() != dst.scalar_type()) {
        return reject("dtype mismatch");
    }
    if ((src.dim() != 1 && src.dim() != 2) || dst.dim() != src.dim()) {
        return reject("only matching 1D or 2D tensors are supported");
    }
    if (src.dim() == 1) {
        if (src.stride(0) != 1 || dst.stride(0) != 1) {
            return reject("1D tensors must have unit stride");
        }
        return src.size(0) <= dst.size(0) ? true : reject("source length exceeds destination");
    }
    if (src.size(0) > dst.size(0) || src.size(1) > dst.size(1)) {
        return reject("source rows or columns exceed destination");
    }
    if (src.stride(1) != 1 || dst.stride(1) != 1) {
        return reject("2D tensor rows must be contiguous");
    }

    const size_t row_bytes  = static_cast<size_t>(src.size(1)) * src.element_size();
    const size_t src_stride = static_cast<size_t>(src.stride(0)) * src.element_size();
    const size_t dst_stride = static_cast<size_t>(dst.stride(0)) * dst.element_size();
    if (row_bytes > src_stride || row_bytes > dst_stride) {
        return reject("row bytes exceed source or destination stride");
    }
    return true;
}

}  // namespace rtp_llm
