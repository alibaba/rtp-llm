#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/cache/TpCacheTransferPlan.h"
#include <torch/all.h>
#include <memory>

namespace rtp_llm {

// Shared by request callbacks. Deregistration happens before pinned storage is released.
class RegisteredCacheStaging {
public:
    RegisteredCacheStaging(size_t source_bytes, std::shared_ptr<MemoryUtil> memory);
    ~RegisteredCacheStaging();
    RegisteredCacheStaging(const RegisteredCacheStaging&)            = delete;
    RegisteredCacheStaging& operator=(const RegisteredCacheStaging&) = delete;

    const torch::Tensor& tensor() const {
        return tensor_;
    }
    const size_t source_bytes;

private:
    friend class RegisteredCacheStagingPool;
    RegisteredCacheStaging(size_t bytes, std::shared_ptr<RegisteredCacheStaging> owner, size_t offset);
    std::shared_ptr<RegisteredCacheStaging> owner_;
    torch::Tensor                           tensor_;
    std::shared_ptr<MemoryUtil>             memory_;
    bool                                    registered_ = false;
};

// Request-local slabs amortize registration over many cache fragments. A slice
// owns its slab through completion/cancellation, independently of this allocator.
class RegisteredCacheStagingPool {
public:
    explicit RegisteredCacheStagingPool(std::shared_ptr<MemoryUtil> memory): memory_(std::move(memory)) {}
    std::shared_ptr<RegisteredCacheStaging> allocate(size_t bytes);

private:
    std::shared_ptr<MemoryUtil>             memory_;
    std::shared_ptr<RegisteredCacheStaging> current_;
    size_t                                  used_       = 0;
    size_t                                  slab_bytes_ = 16 * 1024 * 1024;
};

struct PendingCacheCopy {
    std::shared_ptr<RegisteredCacheStaging> source;
    BlockInfo                               destination;
    TpCacheTransferPlan                     plan;
};

// Validate every destination's shards, assemble on host, then publish each complete GPU payload.
void reassembleCacheTransfers(const std::vector<PendingCacheCopy>& copies, int device_index);

}  // namespace rtp_llm
