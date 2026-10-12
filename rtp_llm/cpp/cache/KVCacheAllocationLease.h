#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator;
class BlockPool;
using BlockPoolPtr = std::shared_ptr<BlockPool>;

// A physical allocation snapshot, independent of prefix-cache keys. Acquire while
// the caller owns the request and serializes allocation/free. Retain the lease
// until GPU work and external readers complete; destruction does not synchronize.
class KVCacheAllocationLease final {
public:
    using BlockTables = std::map<std::string, BlockIndicesType>;

    ~KVCacheAllocationLease();
    KVCacheAllocationLease(const KVCacheAllocationLease&)            = delete;
    KVCacheAllocationLease& operator=(const KVCacheAllocationLease&) = delete;

    const BlockTables& blockTables() const {
        return block_tables_;
    }
    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        return topology_;
    }

private:
    friend class HybridPoolKVCacheAllocator;
    KVCacheAllocationLease(const KVCacheResource&               resource,
                           std::shared_ptr<const CacheTopology> topology,
                           const std::vector<BlockPoolPtr>&     pools);
    void release();

    struct Hold {
        BlockPoolPtr     pool;
        BlockIndicesType blocks;
    };
    std::shared_ptr<const CacheTopology> topology_;
    BlockTables                          block_tables_;
    std::vector<Hold>                    holds_;
    size_t                               acquired_ = 0;
};

using KVCacheAllocationLeasePtr = std::shared_ptr<const KVCacheAllocationLease>;

}  // namespace rtp_llm
