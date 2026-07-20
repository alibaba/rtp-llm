#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

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
        std::vector<std::string>                   per_tag_tags = preparePerTagTags(component_groups, per_tag_mapping);
        std::vector<DeviceKVCacheGroupPtr>         per_tag_device_groups(per_tag_mapping.size());
        std::unique_ptr<BlockTreeCache>            cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                                                 std::move(component_groups),
                                                                                 std::move(components),
                                                                                 std::move(config),
                                                                                 std::move(storage_backend),
                                                                                 std::move(broadcast_manager),
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
                    group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}});
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
                group->referenceBlocks(GroupBlockSet{adopted.component_group_id, Tier::DEVICE, {blocks}});
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

        BlockPoolConfig config;
        config.pool_name        = "block_tree_cache_test_" + tag;
        config.block_num        = static_cast<uint32_t>(physical_block_count);
        config.total_size_bytes = layout.total_size_bytes;
        config.memory_layouts   = {layout};

        auto backing_pool = std::make_shared<BlockPool>(config, AllocationType::HOST);
        RTP_LLM_CHECK(backing_pool->init());
        auto device_pool = std::make_shared<DeviceBlockPool>(std::move(backing_pool));
        RTP_LLM_CHECK(device_pool->init());
        return device_pool;
    }

    static std::vector<BlockTreeCache::PerTagMapping>
    preparePerTagMapping(std::vector<ComponentGroupPtr>& component_groups) {
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping;
        for (ComponentGroupPtr& component_group : component_groups) {
            size_t device_pool_count = component_group->devicePoolCount();
            if (device_pool_count == 0) {
                const std::string tag = "tag_" + std::to_string(per_tag_mapping.size());
                component_group->setDevicePools({makeStructuralDevicePool(tag)}, {tag});
                device_pool_count = 1;
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
