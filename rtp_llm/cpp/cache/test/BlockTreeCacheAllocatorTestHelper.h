#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

namespace rtp_llm::test {

// Mirrors KVCacheManager's mandatory allocator/BTC wiring for allocator unit tests.
// The wrapper owns the cache so the raw allocator observer never outlives it.
template<typename Allocator>
class BlockTreeCacheTestAllocator: public Allocator {
public:
    template<typename... Args>
    explicit BlockTreeCacheTestAllocator(const CacheConfig& config, Args&&... args):
        Allocator(config, std::forward<Args>(args)...), config_(config) {}

    ~BlockTreeCacheTestAllocator() override {
        this->setBlockTreeCache(nullptr);
        block_tree_cache_.reset();
    }

    bool init() {
        if (!Allocator::init()) {
            return false;
        }
        block_tree_cache_ = createBlockTreeCache(config_, kv_cache_config_, this->shared_from_this());
        if (!block_tree_cache_) {
            return false;
        }
        this->setBlockTreeCache(block_tree_cache_.get());
        return true;
    }

    const BlockTreeCachePtr& blockTreeCacheOwner() const {
        return block_tree_cache_;
    }

    void setBlockTreeCacheConfigForTest(KVCacheConfig config) {
        kv_cache_config_ = std::move(config);
    }

private:
    CacheConfig       config_;
    KVCacheConfig     kv_cache_config_;
    BlockTreeCachePtr block_tree_cache_;
};

struct BlockTreeSeedResult {
    bool                                              success{false};
    std::unordered_map<std::string, BlockIndicesType> blocks_by_tag;
};

// Seed a physically valid path through every reusable declarative component group.
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

    const auto&                         component_groups = cache->componentGroups();
    std::vector<std::vector<GroupSlot>> slots(keys.size(), std::vector<GroupSlot>(component_groups.size()));
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIndicesType>> request_holds;

    for (const auto& component_group : component_groups) {
        if (!component_group || component_group->component_group_id < 0
            || static_cast<size_t>(component_group->component_group_id) >= component_groups.size()
            || component_group->tags().size() != component_group->devicePools().size()) {
            for (const auto& [pool, blocks] : request_holds) {
                pool->decRef(blocks);
            }
            return result;
        }

        const size_t component_group_id = static_cast<size_t>(component_group->component_group_id);
        for (size_t pool_index = 0; pool_index < component_group->devicePools().size(); ++pool_index) {
            const auto& device_pool = component_group->devicePools()[pool_index];
            if (!device_pool) {
                for (const auto& [pool, blocks] : request_holds) {
                    pool->decRef(blocks);
                }
                return result;
            }

            auto allocated = device_pool->malloc(keys.size());
            if (!allocated.has_value() || allocated->size() != keys.size()) {
                if (allocated.has_value()) {
                    device_pool->free(*allocated);
                }
                for (const auto& [pool, held_blocks] : request_holds) {
                    pool->decRef(held_blocks);
                }
                return result;
            }
            BlockIndicesType blocks = std::move(*allocated);
            device_pool->incRef(blocks);

            for (size_t path_index = 0; path_index < keys.size(); ++path_index) {
                auto& device_blocks = slots[path_index][component_group_id].device_blocks;
                device_blocks.resize(component_group->devicePools().size(), NULL_BLOCK_IDX);
                device_blocks[pool_index] = blocks[path_index];
            }
            result.blocks_by_tag.emplace(component_group->tags()[pool_index], blocks);
            request_holds.emplace_back(device_pool, std::move(blocks));
        }
    }

    cache->insert(nullptr, keys, slots);
    for (const auto& [pool, blocks] : request_holds) {
        pool->decRef(blocks);
    }
    cache->onBlocksReleased();

    auto match     = cache->match(keys);
    result.success = match.matched_blocks == keys.size();
    cache->releaseMatchedBlocks(match.matched_block_sets);
    return result;
}

}  // namespace rtp_llm::test
