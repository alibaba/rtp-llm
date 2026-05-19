#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cuda_runtime.h>
#include <memory>

namespace rtp_llm {

struct TempStagingGuard {
    void*                                  ptr    = nullptr;
    size_t                                 size   = 0;
    CacheTransferBufferPool*               pool   = nullptr;
    CacheTransferBufferPool::BufferHandle* handle = nullptr;
    std::shared_ptr<MemoryUtil>            memory_util;
    bool                                   is_fallback = false;

    ~TempStagingGuard() {
        if (handle && pool) {
            pool->free(handle);
        } else if (is_fallback && ptr) {
            if (memory_util) {
                memory_util->deregUserMr(ptr, false);
            }
            cudaFreeHost(ptr);
        }
    }

    TempStagingGuard()                                   = default;
    TempStagingGuard(const TempStagingGuard&)            = delete;
    TempStagingGuard& operator=(const TempStagingGuard&) = delete;
};

inline std::shared_ptr<TempStagingGuard>
allocateStaging(CacheTransferBufferPool* pool, const std::shared_ptr<MemoryUtil>& memory_util, size_t size) {
    auto guard         = std::make_shared<TempStagingGuard>();
    guard->size        = size;
    guard->memory_util = memory_util;

    if (pool) {
        auto* handle = pool->tryAllocate(size);
        if (handle) {
            guard->pool   = pool;
            guard->handle = handle;
            guard->ptr    = handle->ptr;
            return guard;
        }
    }

    // Fallback: temporary pinned allocation + MR registration
    void* tmp = nullptr;
    auto  err = cudaHostAlloc(&tmp, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("allocateStaging fallback cudaHostAlloc failed: %s", cudaGetErrorString(err));
        return nullptr;
    }

    if (memory_util && memory_util->isRdmaMode()) {
        if (!memory_util->regUserMr(tmp, size, false)) {
            RTP_LLM_LOG_WARNING("allocateStaging fallback regUserMr failed");
            cudaFreeHost(tmp);
            return nullptr;
        }
    }

    guard->ptr         = tmp;
    guard->is_fallback = true;
    return guard;
}

}  // namespace rtp_llm
