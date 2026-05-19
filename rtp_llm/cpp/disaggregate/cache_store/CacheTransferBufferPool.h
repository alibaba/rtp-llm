#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

constexpr size_t kGatherAlignBytes = 16;

inline size_t alignGatherSize(size_t size) {
    return (size + kGatherAlignBytes - 1) & ~(kGatherAlignBytes - 1);
}

class CacheTransferBufferPool {
public:
    struct BufferHandle {
        void*  ptr    = nullptr;
        size_t size   = 0;
        size_t offset = 0;
    };

    CacheTransferBufferPool(size_t pool_size, const std::shared_ptr<MemoryUtil>& memory_util);
    ~CacheTransferBufferPool();

    CacheTransferBufferPool(const CacheTransferBufferPool&)            = delete;
    CacheTransferBufferPool& operator=(const CacheTransferBufferPool&) = delete;

    BufferHandle* tryAllocate(size_t size);
    void          free(BufferHandle* handle);

    size_t totalBytes() const;
    size_t freeBytes() const;
    size_t largestFreeBlock() const;
    void*  baseAddr() const;

private:
    struct FreeBlock {
        size_t offset;
        size_t size;
    };

    BufferHandle* allocateInternal(size_t aligned_size);

    void*  base_      = nullptr;
    size_t pool_size_ = 0;

    std::list<FreeBlock> free_list_;

    mutable std::mutex mutex_;
    size_t             free_bytes_ = 0;

    std::shared_ptr<MemoryUtil> memory_util_;
    bool                        mr_registered_ = false;
};

}  // namespace rtp_llm
