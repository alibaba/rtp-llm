#include "rtp_llm/cpp/cache/KVCacheAllocationLease.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

KVCacheAllocationLease::KVCacheAllocationLease(const KVCacheResource&               resource,
                                               std::shared_ptr<const CacheTopology> topology,
                                               const std::vector<BlockPoolPtr>&     pools):
    topology_(std::move(topology)) {
    if (!topology_ || resource.topologyPtr() != topology_ || pools.size() != topology_->groups().size()
        || static_cast<size_t>(resource.groupNums()) != pools.size()) {
        throw std::invalid_argument("cache allocation lease requires its allocator's exact topology");
    }
    holds_.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool   = pools[gid];
        const auto& blocks = resource.blocks(static_cast<int>(gid));
        if (!pool) {
            throw std::invalid_argument("cache allocation lease requires initialized pools");
        }
        block_tables_.emplace(topology_->groupById(gid).tag, blocks);
        BlockIndicesType retained;
        for (const auto block : blocks) {
            if (isNullBlockIdx(block)) {
                continue;
            }
            if (block <= 0 || static_cast<size_t>(block) > pool->totalBlocksNum()) {
                throw std::invalid_argument("cache allocation lease has an invalid physical block");
            }
            retained.push_back(block);
        }
        std::sort(retained.begin(), retained.end());
        retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
        holds_.push_back({pool, std::move(retained)});
    }
    // Finish all allocating/metadata work before taking any pool references.
    try {
        for (const auto& hold : holds_) {
            if (!hold.pool->tryRetainRequestBlocks(hold.blocks)) {
                throw std::runtime_error("cache allocation lease requires live request-owned blocks");
            }
            ++acquired_;
        }
    } catch (...) {
        release();
        throw;
    }
}

KVCacheAllocationLease::~KVCacheAllocationLease() {
    release();
}

void KVCacheAllocationLease::release() {
    while (acquired_) {
        const auto& hold = holds_[--acquired_];
        hold.pool->requestFree(hold.blocks);
    }
}

}  // namespace rtp_llm
