#include "rtp_llm/cpp/cache/SharedBlockCache.h"

#include <set>

namespace rtp_llm {

bool SharedBlockCache::putDsv41Checkpoint(const DSV41GpuCheckpointData& checkpoint, bool is_resident) {
    checkpoint.validate();
    if (group_pools_.size() != checkpoint.blocks.size())
        throw std::logic_error("V4.1 GPU checkpoint requires all six physical pools");
    for (size_t group = 0; group < checkpoint.blocks.size(); ++group) {
        const auto& pool = group_pools_[group];
        if (!pool || std::any_of(checkpoint.blocks[group].begin(), checkpoint.blocks[group].end(), [&](int32_t id) {
                return static_cast<size_t>(id) > pool->totalBlocksNum();
            }))
            throw std::invalid_argument("V4.1 GPU checkpoint references an unavailable physical page");
    }
    return dsv41_cache_.publish(
        checkpoint,
        [pools = group_pools_](const DSV41GpuCheckpointData& data) {
            struct Backing {
                std::vector<BlockPoolPtr>       pools;
                std::array<BlockIndicesType, 6> blocks;
                size_t                          retained{0};
                ~Backing() {
                    for (size_t group = 0; group < retained; ++group)
                        pools[group]->blockCacheFree(blocks[group]);
                }
            };
            auto backing    = std::make_shared<Backing>();
            backing->pools  = pools;
            backing->blocks = data.blocks;
            for (size_t group = 0; group < pools.size(); ++group) {
                pools[group]->blockCacheReference(backing->blocks[group]);
                ++backing->retained;
            }
            return std::static_pointer_cast<void>(backing);
        },
        is_resident);
}

DSV41GpuCheckpointCache::Lease SharedBlockCache::matchDsv41Checkpoint(const DSV41CacheIdentity& identity,
                                                                      const CacheKeysType&      keys,
                                                                      size_t                    reuse_unit,
                                                                      size_t                    limit) {
    return dsv41_cache_.match(identity, keys, reuse_unit, limit);
}

bool SharedBlockCache::hasDsv41Checkpoints() const {
    return dsv41_cache_.size() != 0;
}

DSV41GpuCheckpointCache::Lease SharedBlockCache::leaseDsv41CheckpointForTransfer() {
    return dsv41_cache_.leaseOldestForTransfer();
}

void SharedBlockCache::commitDsv41CheckpointTransfer(const DSV41GpuCheckpointCache::Lease& lease) {
    dsv41_cache_.commitTransfer(lease);
}

size_t SharedBlockCache::evictDsv41AndFree(int group_id, size_t min_blocks) {
    size_t freed = 0;
    while (freed < min_blocks) {
        auto snapshots = dsv41_cache_.takeOldestJointEvictable(group_id);
        if (snapshots.empty())
            break;
        std::set<std::pair<size_t, int32_t>> blocks;
        for (const auto& snapshot : snapshots) {
            for (size_t group = 0; group < snapshot->data.blocks.size(); ++group) {
                if (group_id >= 0 && static_cast<size_t>(group_id) != group)
                    continue;
                for (auto block : snapshot->data.blocks[group])
                    blocks.emplace(group, block);
            }
        }
        freed += blocks.size();
    }
    return freed;
}

}  // namespace rtp_llm
