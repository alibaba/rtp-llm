#pragma once

#include <cstddef>
#include <memory>
#include <shared_mutex>

namespace rtp_llm {

class MemoryUtil {
public:
    virtual ~MemoryUtil() = default;

    virtual bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size = 0)       = 0;
    virtual bool deregUserMr(void* buf, bool gpu)                                               = 0;
    virtual bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted)                   = 0;
    virtual bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) = 0;
    virtual bool
    regUserMrWithOwner(void* buf, uint64_t size, bool gpu, uint64_t aligned_size, const std::shared_ptr<void>&) {
        return regUserMr(buf, size, gpu, aligned_size);
    }

    virtual bool supportsDeviceMemory() {
        return isRdmaMode();
    }

    // Direct DMA keeps source and destination pages pinned until physical completion.
    virtual bool requiresBlockLease() const {
        return false;
    }

    virtual bool isRdmaMode() = 0;
};

}  // namespace rtp_llm
