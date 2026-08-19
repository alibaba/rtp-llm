#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <exception>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

AlignedHostMemory::AlignedHostMemory(size_t             usable_bytes,
                                     size_t             alignment,
                                     bool               try_pin_memory,
                                     const std::string& allocation_name) {
    auto cpu = torch::empty({static_cast<int64_t>(usable_bytes + alignment)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    if (try_pin_memory) {
        try {
            backing_ = cpu.pin_memory();
            if (!backing_.is_pinned()) {
                RTP_LLM_LOG_WARNING("pin host memory unavailable, fallback to pageable CPU memory, allocation=%s",
                                    allocation_name.c_str());
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("pin host memory failed, fallback to pageable CPU memory, allocation=%s error=%s",
                                allocation_name.c_str(),
                                e.what());
        }
    }
    if (!backing_.defined() || !backing_.is_pinned()) {
        backing_ = cpu;
    }

    pinned_             = backing_.is_pinned();
    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    data_               = reinterpret_cast<uint8_t*>((raw_base + alignment - 1) / alignment * alignment);
}

uint8_t* AlignedHostMemory::data() const {
    return data_;
}

bool AlignedHostMemory::isPinned() const {
    return pinned_;
}

}  // namespace rtp_llm
