#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockCacheTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
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
        std::vector<std::string>                   per_tag_tags = preparePerTagTags(component_groups, per_tag_mapping);
        std::vector<DeviceKVCacheGroupPtr>         per_tag_device_groups(per_tag_mapping.size());
        auto components_ptr  = std::make_shared<const std::vector<Component>>(std::move(components));
        auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(component_groups, components_ptr);
        std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
        if (broadcast_manager != nullptr) {
            multi_rank_engine =
                std::make_shared<MultiRankBlockTransferEngine>(component_groups, std::move(broadcast_manager));
        }
        auto transfer_dispatcher =
            std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
        auto task_pool = std::make_unique<BlockCacheTaskPool>(
            static_cast<size_t>(config.eviction_thread_pool_size), 1000, "BlockTreeEvictionPool");
        std::unique_ptr<BlockTreeCache> cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                                                 std::move(component_groups),
                                                                                 std::move(components_ptr),
                                                                                 std::move(config),
                                                                                 std::move(storage_backend),
                                                                                 std::move(transfer_dispatcher),
                                                                                 std::move(task_pool),
                                                                                 std::move(per_tag_tags),
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
        const BlockTreeInsertResult           insert_result    = tree->insertNode(parent, cache_keys, slots);
        const std::vector<ComponentGroupPtr>& component_groups = cache.componentGroups();
        for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
            TreeNode* node = inserted.node;
            if (node == nullptr) {
                continue;
            }
            for (const ComponentGroupPtr& group : component_groups) {
                if (group == nullptr || group->component_group_id < 0) {
                    continue;
                }
                const size_t gid = static_cast<size_t>(group->component_group_id);
                if (gid >= node->group_slots.size()) {
                    continue;
                }
                GroupSlot& slot = node->group_slots[gid];
                if (!slot.has_value(Tier::DEVICE)) {
                    continue;
                }
                const std::vector<BlockIdxType> blocks = group->getBlocks(slot, Tier::DEVICE);
                if (!blocks.empty()) {
                    group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}},
                                           BlockRefType::BLOCK_CACHE);
                }
            }
        }
        for (const BlockTreeAdoptedSlot& adopted : insert_result.adopted_slots) {
            if (adopted.node == nullptr || adopted.component_group_id < 0) {
                continue;
            }
            const size_t gid = static_cast<size_t>(adopted.component_group_id);
            if (gid >= component_groups.size() || component_groups[gid] == nullptr
                || gid >= adopted.node->group_slots.size()) {
                continue;
            }
            const ComponentGroupPtr& group  = component_groups[gid];
            const auto               blocks = group->getBlocks(adopted.node->group_slots[gid], Tier::DEVICE);
            if (!blocks.empty()) {
                group->referenceBlocks(GroupBlockSet{adopted.component_group_id, Tier::DEVICE, {blocks}},
                                       BlockRefType::BLOCK_CACHE);
            }
        }
        cache.evictor_.onInsertCommitted(insert_result);
        return insert_result.leaf != nullptr;
    }

private:
    BlockTreeCacheTestUtil() = delete;

    static DeviceBlockPoolPtr makeStructuralDevicePool(const std::string& tag) {
        constexpr size_t physical_block_count = 1024;
        constexpr size_t block_bytes          = 1;

        MemoryLayoutConfig layout;
        layout.layer_num                  = 1;
        layout.block_num                  = static_cast<uint32_t>(physical_block_count);
        layout.dtype                      = TYPE_INT8;
        layout.kv_cache_offset_bytes      = 0;
        layout.kv_block_stride_bytes      = block_bytes;
        layout.kv_block_pool_size_bytes   = physical_block_count * block_bytes;
        layout.block_stride_bytes         = block_bytes;
        layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;

        auto config                     = std::make_shared<DeviceBlockPoolConfig>();
        config->pool_type               = BlockPoolType::DEVICE;
        config->pool_name               = "block_tree_cache_test_" + tag;
        config->physical_block_count    = physical_block_count;
        config->total_size_bytes        = layout.total_size_bytes;
        config->memory_layouts          = {layout};
        config->use_cuda_malloc_backing = false;

        auto pool = std::make_shared<DeviceBlockPool>(config);
        RTP_LLM_CHECK(pool->init());
        auto structural_blocks = pool->malloc(physical_block_count - 1);
        RTP_LLM_CHECK(structural_blocks.has_value());
        // Reserve every literal structural id as allocated at refCount 0. Tree
        // insertion takes the sole cache hold, preserving refCount==1 eviction.
        return pool;
    }

    static std::vector<BlockTreeCache::PerTagMapping>
    preparePerTagMapping(std::vector<ComponentGroupPtr>& component_groups, std::vector<Component>& components) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping;
        for (ComponentGroupPtr& component_group : component_groups) {
            size_t device_pool_count = component_group->devicePoolCount();
            if (device_pool_count == 0) {
                const std::string tag = "tag_" + std::to_string(per_tag_mapping.size());
                component_group->setDevicePools({makeStructuralDevicePool(tag)}, {tag});
                device_pool_count = 1;
            }
            if (!component_group->hasLayout()) {
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
                    component.tag                = component_group->tags().empty() ?
                                                       "test_" + std::to_string(component_group->component_group_id) + "_"
                                            + std::to_string(local_pool_index) :
                                                       component_group->tags()[local_pool_index];
                    component.model_layer_ids    = {0};
                    const size_t layer_bytes     = local_pool_index + 1 == device_pool_count ? remaining_bytes : 1;
                    component.layer_bytes        = {layer_bytes};
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

    static std::vector<std::string>
    preparePerTagTags(const std::vector<ComponentGroupPtr>&             component_groups,
                      const std::vector<BlockTreeCache::PerTagMapping>& per_tag_mapping) {
        std::vector<std::string> per_tag_tags;
        per_tag_tags.reserve(per_tag_mapping.size());
        for (const auto& mapping : per_tag_mapping) {
            RTP_LLM_CHECK(mapping.component_group_id >= 0);
            RTP_LLM_CHECK(static_cast<size_t>(mapping.component_group_id) < component_groups.size());
            const auto& group = component_groups[static_cast<size_t>(mapping.component_group_id)];
            RTP_LLM_CHECK(group != nullptr);
            RTP_LLM_CHECK(mapping.local_pool_index >= 0);
            RTP_LLM_CHECK(static_cast<size_t>(mapping.local_pool_index) < group->tags().size());
            per_tag_tags.push_back(group->tags()[static_cast<size_t>(mapping.local_pool_index)]);
        }
        return per_tag_tags;
    }
};

}  // namespace rtp_llm
