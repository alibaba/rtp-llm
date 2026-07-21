#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class BlockTreeCacheTestUtil {
public:
    static std::unique_ptr<BlockTreeCache> makeBlockTreeCache(std::unique_ptr<BlockTree>        tree,
                                                              std::vector<ComponentGroupPtr>    component_groups,
                                                              std::vector<Component>            components,
                                                              BlockTreeCacheConfig              config,
                                                              std::shared_ptr<StorageBackend>   storage_backend,
                                                              std::shared_ptr<BroadcastManager> broadcast_manager) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping = preparePerTagMapping(component_groups, components);
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
    preparePerTagMapping(std::vector<ComponentGroupPtr>& component_groups, std::vector<Component>& components) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping;
        for (ComponentGroupPtr& component_group : component_groups) {
            size_t device_pool_count = component_group->devicePoolCount();
            if (!component_group->hasLayout()) {
                if (device_pool_count == 0) {
                    component_group->setDevicePools({DeviceBlockPoolPtr{}});
                    device_pool_count = 1;
                }
                size_t payload_bytes = device_pool_count;
                if (component_group->hostPool() != nullptr) {
                    payload_bytes = component_group->hostPool()->payloadBytes();
                } else if (component_group->diskPool() != nullptr) {
                    payload_bytes = component_group->diskPool()->payloadBytes();
                }
                RTP_LLM_CHECK(payload_bytes >= device_pool_count);

                std::vector<int> membership;
                membership.reserve(device_pool_count);
                size_t remaining_bytes = payload_bytes;
                for (size_t local_pool_index = 0; local_pool_index < device_pool_count; ++local_pool_index) {
                    Component component;
                    component.component_id       = static_cast<int>(components.size());
                    component.component_group_id = component_group->component_group_id;
                    component.tag                = "test_" + std::to_string(component_group->component_group_id) + "_"
                                    + std::to_string(local_pool_index);
                    component.model_layer_ids = {0};
                    const size_t layer_bytes  = local_pool_index + 1 == device_pool_count ? remaining_bytes : 1;
                    component.layer_bytes     = {layer_bytes};
                    remaining_bytes -= layer_bytes;
                    membership.push_back(component.component_id);
                    components.push_back(std::move(component));
                }
                RTP_LLM_CHECK(component_group->finalizeLayout(std::move(membership), components));
            }

            {
                const auto& component_indices = component_group->componentIndices();
                RTP_LLM_CHECK_WITH_INFO(device_pool_count == component_indices.size(),
                                        "sealed group %d device pool count %zu != membership count %zu",
                                        component_group->component_group_id,
                                        device_pool_count,
                                        component_indices.size());
                for (int component_index : component_indices) {
                    RTP_LLM_CHECK_WITH_INFO(component_index >= 0
                                                && static_cast<size_t>(component_index) < components.size(),
                                            "sealed group %d component index %d is outside registry size %zu",
                                            component_group->component_group_id,
                                            component_index,
                                            components.size());
                    RTP_LLM_CHECK_WITH_INFO(components[static_cast<size_t>(component_index)].component_group_id
                                                == component_group->component_group_id,
                                            "component %d belongs to group %d, expected %d",
                                            component_index,
                                            components[static_cast<size_t>(component_index)].component_group_id,
                                            component_group->component_group_id);
                }
            }
            for (size_t local_pool_index = 0; local_pool_index < device_pool_count; ++local_pool_index) {
                per_tag_mapping.push_back({component_group->component_group_id, static_cast<int>(local_pool_index)});
            }
        }
        return per_tag_mapping;
    }
};

}  // namespace rtp_llm
