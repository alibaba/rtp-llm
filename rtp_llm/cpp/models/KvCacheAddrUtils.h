#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <torch/extension.h>
#include <cstdint>

namespace rtp_llm {

// GPU addresses are stored as int64_t because PyTorch lacks an unsigned 64-bit
// tensor dtype.  The bit pattern is preserved through matching reinterpret_casts
// on both store and load sides; no pointer arithmetic is performed on the int64_t
// representation, so the signedness does not affect correctness.
static_assert(sizeof(void*) <= sizeof(int64_t), "GPU pointer must fit in int64_t");

inline torch::Tensor buildLayerAddrBuffer(const std::vector<torch::Tensor>& layer_kv_buffer_ptrs) {
    auto  num_layers = layer_kv_buffer_ptrs.size();
    auto  layer_addrs = torch::empty({static_cast<int64_t>(num_layers)}, torch::kInt64);
    auto* addr_data   = layer_addrs.data_ptr<int64_t>();
    for (size_t i = 0; i < num_layers; ++i) {
        if (!layer_kv_buffer_ptrs[i].defined()) {
            RTP_LLM_LOG_WARNING("layer_kv_buffer_ptrs[%zu] is undefined", i);
            return torch::Tensor();
        }
        addr_data[i] = reinterpret_cast<int64_t>(layer_kv_buffer_ptrs[i].data_ptr());
    }
    return layer_addrs.cuda();
}

}  // namespace rtp_llm
