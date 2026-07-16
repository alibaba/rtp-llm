#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"

#include <algorithm>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BlockTreeEvictor::EvictionPlan::needsCopy() const {
    return primary.target_tier != Tier::NONE
           || std::any_of(cascade_moves.begin(), cascade_moves.end(), [](const EvictionMove& cascade_move) {
                  return cascade_move.target_tier != Tier::NONE;
              });
}

BlockTreeEvictor::BlockTreeEvictor(std::vector<ComponentGroupPtr>& component_groups,
                                   ExecuteTransferFn               execute_transfer,
                                   bool                            enable_reverse_eviction):
    component_groups_(component_groups),
    execute_transfer_(std::move(execute_transfer)),
    enable_reverse_eviction_(enable_reverse_eviction) {}

void BlockTreeEvictor::init(const std::vector<Component>& components) {
    buildGroupLayerTagSlots(components);
}

void BlockTreeEvictor::buildGroupLayerTagSlots(const std::vector<Component>& components) {
    size_t group_count = component_groups_.size();
    for (const auto& group : component_groups_) {
        if (group->component_group_id >= 0) {
            group_count = std::max(group_count, static_cast<size_t>(group->component_group_id) + 1);
        }
    }

    group_layer_tag_slots_.clear();
    group_layer_tag_slots_.resize(group_count);

    for (const auto& group : component_groups_) {
        if (group->component_group_id < 0)
            continue;

        auto& layer_slots = group_layer_tag_slots_[static_cast<size_t>(group->component_group_id)];
        for (int component_index : group->component_indices) {
            if (component_index < 0 || static_cast<size_t>(component_index) >= components.size())
                continue;
            const auto& component = components[static_cast<size_t>(component_index)];
            layer_slots.insert(layer_slots.end(),
                               component.memory_block_layer_tag_slots.begin(),
                               component.memory_block_layer_tag_slots.end());
        }
        std::sort(layer_slots.begin(),
                  layer_slots.end(),
                  [](const MemoryBlockLayerTagSlot& a, const MemoryBlockLayerTagSlot& b) {
                      return a.layer_id < b.layer_id;
                  });
    }
}

std::optional<EvictionMove> BlockTreeEvictor::chooseVictim(Tier tier) {
    for (auto& group : component_groups_) {
        auto eviction_move = group->driveEviction(1, tier);
        if (!eviction_move.has_value()) {
            continue;
        }

        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::chooseVictim: selected candidate, "
                          "group[%d] type=%s tier=%s target=%s node_key=%ld",
                          eviction_move->component_group_id,
                          cacheGroupTypeName(group->group_type),
                          tierName(eviction_move->source_tier),
                          tierName(eviction_move->target_tier),
                          eviction_move->node ? eviction_move->node->cache_key : 0);
        return eviction_move;
    }

    return std::nullopt;
}

std::vector<EvictionMove>
BlockTreeEvictor::chooseWatermarkVictims(ComponentGroup& group, Tier tier, double watermark_ratio) {
    std::vector<EvictionMove> victims;
    if (watermark_ratio <= 0.0)
        return victims;

    size_t excess = computeGroupExcess(group, tier, watermark_ratio);
    if (excess == 0)
        return victims;

    RTP_LLM_LOG_INFO("BlockTreeEvictor::chooseWatermarkVictims: tier=%s group[%d] "
                     "excess=%zu (ratio=%.2f), evicting",
                     tierName(tier),
                     group.component_group_id,
                     excess,
                     watermark_ratio);

    victims.reserve(excess);
    for (size_t i = 0; i < excess; ++i) {
        auto eviction_move = group.driveEviction(1, tier);
        if (eviction_move.has_value())
            victims.push_back(*eviction_move);
        else
            break;
    }
    return victims;
}

std::optional<BlockTreeEvictor::EvictionPlan> BlockTreeEvictor::buildPlan(EvictionMove eviction_move) {
    EvictionPlan plan;
    if (eviction_move.node == nullptr)
        return std::nullopt;

    if (!prepareMove(eviction_move)) {
        restoreSourceHeap(eviction_move);
        releaseTargetBlocks(eviction_move);
        return std::nullopt;
    }
    plan.primary = eviction_move;

    for (int cascade_group_id : selectCascadeGroups(eviction_move.node,
                                                    eviction_move.component_group_id,
                                                    eviction_move.source_tier,
                                                    enable_reverse_eviction_)) {
        auto cascade_move =
            makeMove(eviction_move.node, cascade_group_id, eviction_move.source_tier, eviction_move.target_tier);
        if (cascade_move.source_blocks.empty())
            continue;

        if (!prepareMove(cascade_move)) {
            restoreSourceHeap(cascade_move);
            releaseTargetBlocks(cascade_move);
            RTP_LLM_LOG_WARNING("BlockTreeEvictor::buildPlan: cascade target alloc failed "
                                "group[%d] tier %s->%s node_key=%ld, skipping",
                                cascade_group_id,
                                tierName(cascade_move.source_tier),
                                tierName(cascade_move.target_tier),
                                eviction_move.node->cache_key);
            continue;
        }
        plan.cascade_moves.push_back(std::move(cascade_move));
    }

    return plan;
}

BlockTreeEvictor::CopyResultSet BlockTreeEvictor::performCopy(const EvictionPlan& plan) {
    CopyResultSet results;
    results.primary_success = true;

    if (plan.primary.target_tier != Tier::NONE) {
        results.primary_success = executeTierCopy(plan.primary);
        if (!results.primary_success) {
            RTP_LLM_LOG_WARNING("BlockTreeEvictor::performCopy: primary copy FAILED "
                                "group[%d] node_key=%ld %s->%s",
                                plan.primary.component_group_id,
                                plan.primary.node ? plan.primary.node->cache_key : 0,
                                tierName(plan.primary.source_tier),
                                tierName(plan.primary.target_tier));
            results.cascade_success.assign(plan.cascade_moves.size(), false);
            return results;
        } else {
            RTP_LLM_LOG_DEBUG("BlockTreeEvictor::performCopy: primary copy OK "
                              "group[%d] node_key=%ld %s->%s",
                              plan.primary.component_group_id,
                              plan.primary.node ? plan.primary.node->cache_key : 0,
                              tierName(plan.primary.source_tier),
                              tierName(plan.primary.target_tier));
        }
    }

    results.cascade_success.reserve(plan.cascade_moves.size());
    for (const auto& cascade_move : plan.cascade_moves) {
        bool copy_ok = true;
        if (cascade_move.target_tier != Tier::NONE) {
            copy_ok = executeTierCopy(cascade_move);
        }
        results.cascade_success.push_back(copy_ok);

        if (!copy_ok) {
            RTP_LLM_LOG_WARNING("BlockTreeEvictor::performCopy: cascade copy FAILED "
                                "group[%d] node_key=%ld %s->%s",
                                cascade_move.component_group_id,
                                cascade_move.node ? cascade_move.node->cache_key : 0,
                                tierName(cascade_move.source_tier),
                                tierName(cascade_move.target_tier));
        } else if (cascade_move.target_tier != Tier::NONE) {
            RTP_LLM_LOG_DEBUG("BlockTreeEvictor::performCopy: cascade copy OK "
                              "group[%d] node_key=%ld %s->%s",
                              cascade_move.component_group_id,
                              cascade_move.node ? cascade_move.node->cache_key : 0,
                              tierName(cascade_move.source_tier),
                              tierName(cascade_move.target_tier));
        }
    }
    return results;
}

void BlockTreeEvictor::complete(BlockTree& tree, const EvictionPlan& plan, const CopyResultSet& results) {
    if (plan.primary.node == nullptr)
        return;

    if (!results.primary_success) {
        rollbackPreparedPlan(plan);
        return;
    }

    auto primary_gid = static_cast<size_t>(plan.primary.component_group_id);
    if (primary_gid < component_groups_.size() && primary_gid < plan.primary.node->group_slots.size()) {
        auto& group = component_groups_[primary_gid];
        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::complete: primary group[%d] node_key=%ld source=%s target=%s",
                          plan.primary.component_group_id,
                          plan.primary.node->cache_key,
                          tierName(plan.primary.source_tier),
                          tierName(plan.primary.target_tier));
        applyMoveCompletion(group, plan.primary);
    }

    for (size_t i = 0; i < plan.cascade_moves.size(); ++i) {
        const auto& cascade_move = plan.cascade_moves[i];
        const bool  ok   = i < results.cascade_success.size() && results.cascade_success[i];
        if (!ok) {
            releaseTargetBlocks(cascade_move);
            restoreSourceHeap(cascade_move);
            continue;
        }

        auto gid = static_cast<size_t>(cascade_move.component_group_id);
        if (gid >= component_groups_.size() || cascade_move.node == nullptr ||
            gid >= cascade_move.node->group_slots.size()) {
            releaseTargetBlocks(cascade_move);
            continue;
        }

        auto& group = component_groups_[gid];

        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::complete: cascade group[%d] node_key=%ld source=%s target=%s",
                          cascade_move.component_group_id,
                          cascade_move.node->cache_key,
                          tierName(cascade_move.source_tier),
                          tierName(cascade_move.target_tier));

        applyMoveCompletion(group, cascade_move);
    }

    finalizeEviction(tree, plan.primary.node);
}

// Move source blocks out of the slot, install target blocks (if demoting), and
// release the source cache reference. Source blocks were held only by cache.
void BlockTreeEvictor::applyMoveCompletion(ComponentGroupPtr& group, const EvictionMove& move) {
    auto& slot         = move.node->group_slots[static_cast<size_t>(move.component_group_id)];
    auto  source_block = group->getBlocks(slot, move.source_tier);
    // Release source cache-hold (saved ids) before clearing the slot.
    group->unreferenceBlocks(GroupBlockSet{move.component_group_id, move.source_tier, {source_block}});
    group->evictFromTier(move.node, slot, move.source_tier);

    if (move.target_tier != Tier::NONE) {
        group->setBlocks(slot, move.target_tier, move.target_blocks);
        group->tryAddToHeap(move.node, move.target_tier);
    }
}

void BlockTreeEvictor::rollbackPreparedPlan(const EvictionPlan& plan) {
    releaseTargetBlocks(plan.primary);
    restoreSourceHeap(plan.primary);
    for (const auto& cascade_move : plan.cascade_moves) {
        releaseTargetBlocks(cascade_move);
        restoreSourceHeap(cascade_move);
    }
}

void BlockTreeEvictor::writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                                          CacheKeyType                           cache_key,
                                          int                                    component_group_id) {
    if (!storage_backend)
        return;

    auto key = std::to_string(cache_key) + "_g" + std::to_string(component_group_id);
    std::vector<std::pair<std::string, std::vector<char>>> items;
    items.emplace_back(std::move(key), std::vector<char>{});
    if (!items.back().second.empty()) {
        storage_backend->batchWrite(items);
        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::writeRemoteThrough: remote write-through "
                          "group[%d] node_key=%ld",
                          component_group_id,
                          cache_key);
    } else {
        RTP_LLM_LOG_WARNING("BlockTreeEvictor::writeRemoteThrough: remote write-through SKIPPED "
                            "(no data serialization yet) group[%d] node_key=%ld",
                            component_group_id,
                            cache_key);
    }
}

bool BlockTreeEvictor::executeTierCopy(const EvictionMove& eviction_move) {
    if (!execute_transfer_) {
        return false;
    }

    TransferDescriptor descriptor;
    if (!buildTransferDescriptor(eviction_move, descriptor)) {
        return false;
    }

    return execute_transfer_(descriptor) == CopyStatus::OK;
}

bool BlockTreeEvictor::buildTransferDescriptor(const EvictionMove& eviction_move, TransferDescriptor& descriptor) {
    if (eviction_move.source_blocks.empty() || eviction_move.target_blocks.empty()
        || isNullBlockIdx(eviction_move.target_blocks[0])) {
        return false;
    }

    const BlockIdxType target = eviction_move.target_blocks[0];
    if (eviction_move.source_tier == Tier::DEVICE && eviction_move.target_tier == Tier::HOST) {
        descriptor =
            TransferDescriptor::deviceToHost(eviction_move.component_group_id, eviction_move.source_blocks, target);
    } else if (eviction_move.source_tier == Tier::HOST && eviction_move.target_tier == Tier::DISK) {
        if (isNullBlockIdx(eviction_move.source_blocks[0])) {
            return false;
        }
        descriptor =
            TransferDescriptor::hostToDisk(eviction_move.component_group_id, eviction_move.source_blocks[0], target);
    } else {
        return false;
    }

    return true;
}

EvictionMove BlockTreeEvictor::makeMove(TreeNode* node,
                                         int       component_group_id,
                                         Tier      source_tier,
                                         Tier      target_tier) const {
    EvictionMove eviction_move;
    eviction_move.node               = node;
    eviction_move.component_group_id = component_group_id;
    eviction_move.source_tier        = source_tier;
    eviction_move.target_tier        = target_tier;

    auto gid = static_cast<size_t>(component_group_id);
    if (node == nullptr || gid >= node->group_slots.size())
        return eviction_move;

    const auto& slot = node->group_slots[gid];
    switch (source_tier) {
        case Tier::DEVICE:
            if (slot.has_value(Tier::DEVICE))
                eviction_move.source_blocks = slot.device_blocks;
            break;
        case Tier::HOST:
            if (slot.has_value(Tier::HOST))
                eviction_move.source_blocks = {slot.host_block};
            break;
        case Tier::DISK:
            if (slot.has_value(Tier::DISK))
                eviction_move.source_blocks = {slot.disk_slot};
            break;
        default:
            break;
    }
    return eviction_move;
}

bool BlockTreeEvictor::prepareMove(EvictionMove& eviction_move) {
    if (eviction_move.node == nullptr || eviction_move.source_blocks.empty())
        return false;

    reserveSourceHeap(eviction_move);
    if (eviction_move.target_tier != Tier::NONE) {
        auto gid = static_cast<size_t>(eviction_move.component_group_id);
        if (gid >= component_groups_.size())
            return false;
        // cache self-allocated path: malloc + incRef, not yet written to slot/heap.
        BlockIdxType target = component_groups_[gid]->allocateSingleBlock(eviction_move.target_tier);
        if (isNullBlockIdx(target))
            return false;
        eviction_move.target_blocks = {target};
    }

    return true;
}

void BlockTreeEvictor::reserveSourceHeap(const EvictionMove& eviction_move) {
    auto gid = static_cast<size_t>(eviction_move.component_group_id);
    if (gid >= component_groups_.size() || eviction_move.node == nullptr ||
        gid >= eviction_move.node->group_slots.size())
        return;

    auto& group = component_groups_[gid];
    auto& slot  = eviction_move.node->group_slots[gid];
    if (auto* heap = group->heapForTier(eviction_move.source_tier)) {
        heap->invalidate(eviction_move.node);
    }

    switch (eviction_move.source_tier) {
        case Tier::DEVICE:
            slot.in_device_heap = false;
            break;
        case Tier::HOST:
            slot.in_host_heap = false;
            break;
        case Tier::DISK:
            slot.in_disk_heap = false;
            break;
        default:
            break;
    }
}

void BlockTreeEvictor::restoreSourceHeap(const EvictionMove& eviction_move) {
    auto gid = static_cast<size_t>(eviction_move.component_group_id);
    if (gid >= component_groups_.size() || eviction_move.node == nullptr) {
        return;
    }

    auto& group = component_groups_[gid];
    switch (eviction_move.source_tier) {
        case Tier::DEVICE:
            group->tryAddToDeviceHeap(eviction_move.node);
            break;
        case Tier::HOST:
            group->tryAddToHostHeap(eviction_move.node);
            break;
        case Tier::DISK:
            group->tryAddToDiskHeap(eviction_move.node);
            break;
        default:
            break;
    }
}

void BlockTreeEvictor::releaseTargetBlocks(const EvictionMove& eviction_move) {
    if (eviction_move.target_blocks.empty())
        return;
    auto gid = static_cast<size_t>(eviction_move.component_group_id);
    if (gid >= component_groups_.size())
        return;
    auto& group = component_groups_[gid];
    for (auto block : eviction_move.target_blocks)
        group->releaseSingleBlock(eviction_move.target_tier, block);
}

void BlockTreeEvictor::finalizeEviction(BlockTree& tree, TreeNode* node) {
    if (shouldDeleteNode(tree, node)) {
        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::finalizeEviction: deleting empty node key=%ld", node->cache_key);
        TreeNode* parent = node->parent;
        tree.removeNode(node);
        tree.removeEmptyAncestors(parent, reusableGroupIds());
        if (parent && parent != tree.root() && parent->parent != nullptr) {
            for (auto& g : component_groups_) {
                g->tryAddToDeviceHeap(parent);
            }
        }
    } else if (node->parent && node->parent != tree.root()) {
        TreeNode* parent = node->parent;
        for (auto& g : component_groups_) {
            g->tryAddToDeviceHeap(parent);
        }
    }
}

bool BlockTreeEvictor::shouldDeleteNode(const BlockTree& tree, const TreeNode* node) const {
    if (node == nullptr || node == tree.root() || !node->children.empty())
        return false;
    for (const auto& group : component_groups_) {
        auto gidx = static_cast<size_t>(group->component_group_id);
        if (gidx < node->group_slots.size() && !node->group_slots[gidx].is_empty()) {
            return false;
        }
    }
    return true;
}

std::vector<int> BlockTreeEvictor::reusableGroupIds() const {
    std::vector<int> ids;
    for (const auto& group : component_groups_) {
        ids.push_back(group->component_group_id);
    }
    return ids;
}

std::vector<int>
BlockTreeEvictor::selectCascadeGroups(const TreeNode* node,
                                      int             source_group_id,
                                      Tier            tier,
                                      bool            enable_reverse_eviction) const {
    std::vector<int> result;

    const ComponentGroupPtr* source_group = nullptr;
    for (const auto& group : component_groups_) {
        if (group->component_group_id == source_group_id) {
            source_group = &group;
            break;
        }
    }

    if (enable_reverse_eviction && source_group != nullptr &&
        (*source_group)->isLeafAtTier(node, source_group_id, tier)) {
        for (const auto& group : component_groups_) {
            if (group->component_group_id != source_group_id)
                result.push_back(group->component_group_id);
        }
        return result;
    }

    CacheGroupType source_type = source_group != nullptr ? (*source_group)->group_type : CacheGroupType::FULL;
    for (const auto& group : component_groups_) {
        bool below = false;
        switch (source_type) {
            case CacheGroupType::FULL:
                below = (group->group_type == CacheGroupType::SWA || group->group_type == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::SWA:
                below = (group->group_type == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::LINEAR:
                below = false;
                break;
        }
        if (below)
            result.push_back(group->component_group_id);
    }
    return result;
}

size_t BlockTreeEvictor::computeGroupExcess(const ComponentGroup& group, Tier tier, double ratio) const {
    if (tier == Tier::DEVICE) {
        return group.devicePoolMaxExcess(ratio);
    }
    size_t capacity = (tier == Tier::HOST) ? group.hostPoolCapacity() : group.diskPoolCapacity();
    if (capacity == 0)
        return 0;
    size_t used      = (tier == Tier::HOST) ? group.hostPoolUsed() : group.diskPoolUsed();
    size_t threshold = static_cast<size_t>(capacity * ratio);
    return (used > threshold) ? (used - threshold) : 0;
}

}  // namespace rtp_llm
