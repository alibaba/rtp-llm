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
    virtual bool isRdmaMode()                                                                   = 0;
    virtual bool supportsLoadWriteStop() {
        return !isRdmaMode();
    }
    // The owning KV pool calls deregUserMr before releasing this allocation.
    // Transports may reuse completed load registrations within that lifetime.
    virtual bool regLoadDestinationMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size = 0) {
        return regUserMr(buf, size, gpu, aligned_size);
    }
};

}  // namespace rtp_llm
