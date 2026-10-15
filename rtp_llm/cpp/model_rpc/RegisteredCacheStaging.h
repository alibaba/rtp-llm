#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/model_rpc/TpCacheTransferPlan.h"
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
    torch::Tensor               tensor_;
    std::shared_ptr<MemoryUtil> memory_;
    bool                        registered_ = false;
};

struct PendingCacheCopy {
    std::shared_ptr<RegisteredCacheStaging> source;
    BlockInfo                               destination;
    TpCacheTransferPlan                     plan;
};

// Validate every destination's shards, assemble on host, then publish each complete GPU payload.
void reassembleCacheTransfers(const std::vector<PendingCacheCopy>& copies, int device_index);

}  // namespace rtp_llm
