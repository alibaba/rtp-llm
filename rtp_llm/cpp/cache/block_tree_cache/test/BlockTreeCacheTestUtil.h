#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

namespace rtp_llm {

class BlockTreeCacheTestUtil {
public:
    static std::unique_ptr<BlockTreeCache> makeBlockTreeCache(std::unique_ptr<BlockTree>        tree,
                                                              std::vector<ComponentGroupPtr>    component_groups,
                                                              std::vector<Component>            components,
                                                              BlockTreeCacheConfig              config,
                                                              std::shared_ptr<StorageBackend>   storage_backend,
                                                              std::shared_ptr<BroadcastManager> broadcast_manager) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping = preparePerTagMapping(component_groups);
        std::vector<DeviceKVCacheGroupPtr>         per_tag_device_groups(per_tag_mapping.size());
        std::unique_ptr<BlockTreeCache>            cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                                                 std::move(component_groups),
                                                                                 std::move(components),
                                                                                 std::move(config),
                                                                                 std::move(storage_backend),
                                                                                 std::move(broadcast_manager),
                                                                                 std::move(per_tag_device_groups),
                                                                                 std::move(per_tag_mapping));
        if (!cache->init()) {
            return nullptr;
        }
        return cache;
    }

    static std::unique_ptr<BlockTreeCache> makeBlockTreeCache(std::unique_ptr<BlockTree>     tree,
                                                              std::vector<ComponentGroupPtr> component_groups,
                                                              std::vector<Component>         components,
                                                              BlockTreeCacheConfig           config) {
        return makeBlockTreeCache(std::move(tree),
                                  std::move(component_groups),
                                  std::move(components),
                                  std::move(config),
                                  std::shared_ptr<StorageBackend>{},
                                  std::shared_ptr<BroadcastManager>{});
    }

    static std::unique_ptr<BlockTreeCache> makeBlockTreeCache(std::unique_ptr<BlockTree>     tree,
                                                              std::vector<ComponentGroupPtr> component_groups,
                                                              std::vector<Component>         components) {
        return makeBlockTreeCache(
            std::move(tree), std::move(component_groups), std::move(components), BlockTreeCacheConfig{});
    }

    // Seeds component-group slots directly for Host/Disk transition tests.
    static bool insertComponentGroupSlots(BlockTreeCache&                            cache,
                                          TreeNode*                                  parent,
                                          const CacheKeysType&                       cache_keys,
                                          const std::vector<std::vector<GroupSlot>>& slots) {
        BlockTree* tree = cache.tree();
        if (tree == nullptr) {
            return false;
        }
        const BlockTreeInsertResult insert_result = tree->insertNode(parent, cache_keys, slots);
        cache.evictor_.onInsertCommitted(insert_result);
        return insert_result.leaf != nullptr;
    }

private:
    BlockTreeCacheTestUtil() = delete;

    static std::vector<BlockTreeCache::PerTagMapping>
    preparePerTagMapping(std::vector<ComponentGroupPtr>& component_groups) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping;
        for (ComponentGroupPtr& component_group : component_groups) {
            size_t device_pool_count = component_group->devicePoolCount();
            if (device_pool_count == 0) {
                component_group->setDevicePools({DeviceBlockPoolPtr{}});
                device_pool_count = 1;
            }
            for (size_t local_pool_index = 0; local_pool_index < device_pool_count; ++local_pool_index) {
                per_tag_mapping.push_back({component_group->component_group_id, static_cast<int>(local_pool_index)});
            }
        }
        return per_tag_mapping;
    }
};

}  // namespace rtp_llm
