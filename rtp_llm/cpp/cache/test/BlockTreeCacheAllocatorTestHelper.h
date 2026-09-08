#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm::test {

// Mirrors KVCacheManager's mandatory allocator/BTC wiring for allocator unit tests.
template<typename Allocator>
class BlockTreeCacheTestAllocator: public Allocator {
public:
    template<typename... Args>
    explicit BlockTreeCacheTestAllocator(const CacheConfig& config, Args&&... args):
        Allocator(config, std::forward<Args>(args)...), config_(config) {}

    bool init() {
        if (!Allocator::init()) {
            return false;
        }
        auto block_tree_cache = createBlockTreeCache(
            config_, kv_cache_config_, this->shared_from_this(), ParallelismConfig{}, storage_backend_);
        if (!block_tree_cache) {
            return false;
        }
        this->attachBlockTreeCache(std::move(block_tree_cache));
        return true;
    }

    const BlockTreeCachePtr& blockTreeCacheOwner() const {
        return this->block_tree_cache_;
    }

    void setBlockTreeCacheConfigForTest(KVCacheConfig config) {
        kv_cache_config_ = std::move(config);
    }

    void setStorageBackendForTest(std::shared_ptr<StorageBackend> storage_backend) {
        storage_backend_ = std::move(storage_backend);
    }

    MallocStatus preparedReserveStatusForTest(const MallocInfo&              malloc_info,
                                              size_t                         reserve_blocks,
                                              std::vector<RequiredPositions> required_positions) const {
        typename Allocator::PreparedKVCache prepared;
        prepared.required_positions = std::move(required_positions);
        return this->evaluatePreparedInitCapacity(
            malloc_info, reserve_blocks, prepared, /*has_load_context=*/true);
    }

private:
    CacheConfig   config_;
    KVCacheConfig kv_cache_config_;
    std::shared_ptr<StorageBackend> storage_backend_;
};

struct BlockTreeSeedResult {
    bool                                              success{false};
    std::unordered_map<std::string, BlockIndicesType> blocks_by_tag;
};

// Seed a physically valid path through every reusable declarative group set.
// The request references are dropped after insertion; BlockTreeCache's own holders
// keep the seeded blocks alive until the path is reclaimed.
template<typename Allocator>
BlockTreeSeedResult seedCompleteBlockTreePath(const std::shared_ptr<BlockTreeCacheTestAllocator<Allocator>>& allocator,
                                              const CacheKeysType&                                           keys) {
    BlockTreeSeedResult result;
    if (!allocator || keys.empty()) {
        return result;
    }

    const BlockTreeCachePtr& cache = allocator->blockTreeCacheOwner();
    if (!cache) {
        return result;
    }

    const auto&                                group_sets = cache->groupSets();
    std::vector<std::vector<GroupSetResource>> slots(keys.size(), std::vector<GroupSetResource>(group_sets.size()));
    std::vector<std::tuple<size_t, DeviceBlockPoolPtr, BlockIndicesType>> request_holds;

    for (const auto& group_set : group_sets) {
        if (!group_set || group_set->groupSetId() >= group_sets.size()
            || group_set->groupIds().size() != group_set->devicePools().size()) {
            for (const auto& [group_id, pool, blocks] : request_holds) {
                (void)group_id;
                pool->decRef(blocks);
            }
            return result;
        }

        const size_t group_set_id = group_set->groupSetId();
        for (size_t pool_index = 0; pool_index < group_set->devicePools().size(); ++pool_index) {
            const auto& device_pool = group_set->devicePools()[pool_index];
            if (!device_pool) {
                for (const auto& [group_id, pool, blocks] : request_holds) {
                    (void)group_id;
                    pool->decRef(blocks);
                }
                return result;
            }

            auto allocated = device_pool->malloc(keys.size());
            if (!allocated.has_value() || allocated->size() != keys.size()) {
                if (allocated.has_value()) {
                    device_pool->incRef(*allocated);
                    device_pool->decRef(*allocated);
                }
                for (const auto& [group_id, pool, held_blocks] : request_holds) {
                    (void)group_id;
                    pool->decRef(held_blocks);
                }
                return result;
            }
            BlockIndicesType blocks = std::move(*allocated);
            device_pool->incRef(blocks);

            for (size_t path_index = 0; path_index < keys.size(); ++path_index) {
                auto& device_blocks = slots[path_index][group_set_id].device_blocks;
                device_blocks.resize(group_set->devicePools().size(), NULL_BLOCK_IDX);
                device_blocks[pool_index] = blocks[path_index];
            }
            result.blocks_by_tag.emplace(group_set->groupAt(pool_index).tag, blocks);
            request_holds.emplace_back(group_set->groupIds()[pool_index], device_pool, std::move(blocks));
        }
    }

    cache->insert(keys, slots, Tier::DEVICE);
    for (const auto& [group_id, pool, blocks] : request_holds) {
        (void)group_id;
        pool->decRef(blocks);
    }

    auto match     = cache->match(keys);
    result.success = match.matched_device_blocks == keys.size();
    block_tree_cache_test::releaseRequestRefsForTest(*cache, match.matched_device_resources);
    return result;
}

}  // namespace rtp_llm::test
