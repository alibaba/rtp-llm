#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <exception>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

AlignedHostMemory::AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name) {
    try {
        backing_ = torch::empty({static_cast<int64_t>(usable_bytes + alignment)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("allocate pinned host memory failed, allocation=%s usable_bytes=%zu error=%s",
                     allocation_name.c_str(),
                     usable_bytes,
                     e.what());
    }
    RTP_LLM_CHECK_WITH_INFO(
        backing_.is_pinned(), "host allocation [%s] must use pinned CPU memory", allocation_name.c_str());

    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    data_               = reinterpret_cast<uint8_t*>((raw_base + alignment - 1) / alignment * alignment);
}

uint8_t* AlignedHostMemory::data() const {
    return data_;
}

}  // namespace rtp_llm
