#include "rtp_llm/cpp/pybind/PreprocessSharedStorage.h"

#include <ATen/StorageUtils.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rtp_llm {

std::vector<at::Tensor> packPreprocessCpuTensors(const std::vector<at::Tensor>& tensors) {
    constexpr size_t alignment = 64;
    constexpr auto   max_bytes = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    TORCH_CHECK(!tensors.empty(), "preprocess tensor group is empty");

    std::vector<size_t> offsets;
    offsets.reserve(tensors.size());
    size_t total_bytes = 0;
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.defined() && tensor.device().is_cpu() && tensor.layout() == at::kStrided
                        && tensor.is_contiguous() && !tensor.requires_grad() && !tensor.is_conj() && !tensor.is_neg()
                        && !tensor.is_quantized() && tensor.numel() > 0,
                    "preprocess tensor group requires nonempty contiguous CPU tensors");
        TORCH_CHECK(total_bytes <= max_bytes - (alignment - 1), "preprocess tensor group is too large");
        total_bytes = (total_bytes + alignment - 1) & ~(alignment - 1);
        TORCH_CHECK(tensor.nbytes() <= max_bytes - total_bytes, "preprocess tensor group is too large");
        offsets.push_back(total_bytes);
        total_bytes += tensor.nbytes();
    }

    // The multiprocessing pickler sends one fd for the shared storage and
    // preserves each tensor's dtype, size, and offset in its ordinary reducer.
    at::Storage             storage(at::new_shm_fd_storage(total_bytes));
    auto*                   destination = static_cast<char*>(storage.mutable_data());
    std::vector<at::Tensor> views;
    views.reserve(tensors.size());
    for (size_t index = 0; index < tensors.size(); ++index) {
        const auto& tensor = tensors[index];
        const auto  offset = offsets[index];
        std::memcpy(destination + offset, tensor.const_data_ptr(), tensor.nbytes());
        views.push_back(at::empty({0}, tensor.options())
                            .set_(storage, offset / tensor.element_size(), tensor.sizes(), tensor.strides()));
    }
    return views;
}

}  // namespace rtp_llm
