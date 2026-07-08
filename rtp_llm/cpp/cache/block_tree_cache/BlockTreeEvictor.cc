#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"

#include <algorithm>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BlockTreeEvictor::EvictionPlan::needsCopy() const {
    return primary.target_tier != Tier::NONE
           || std::any_of(cascade_moves.begin(), cascade_moves.end(), [](const EvictionMove& move) {
                  return move.target_tier != Tier::NONE;
              });
}

BlockTreeEvictor::BlockTreeEvictor(std::vector<ComponentGroupPtr>& component_groups,
                                   CopyEnginePtr                   copy_engine,
                                   bool                            enable_reverse_eviction):
    component_groups_(component_groups),
    copy_engine_(std::move(copy_engine)),
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
        auto er = group->driveEviction(1, tier);
        if (!er.has_value()) {
            continue;
        }

        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::chooseVictim: selected candidate, "
                          "group[%d] type=%s tier=%s target=%s node_key=%ld",
                          er->component_group_id,
                          cacheGroupTypeName(group->group_type),
                          tierName(er->source_tier),
                          tierName(er->target_tier),
                          er->node ? er->node->cache_key : 0);
        return er;
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
        auto er = group.driveEviction(1, tier);
        if (er.has_value())
            victims.push_back(*er);
        else
            break;
    }
    return victims;
}

std::optional<BlockTreeEvictor::EvictionPlan> BlockTreeEvictor::buildPlan(EvictionMove er) {
    EvictionPlan plan;
    if (er.node == nullptr)
        return std::nullopt;

    if (!prepareMove(er)) {
        restoreSourceHeap(er);
        releaseTargetBlock(er);
        return std::nullopt;
    }
    plan.primary = er;

    for (int cascade_group_id : selectCascadeGroups(er.node,
                                                    er.component_group_id,
                                                    er.source_tier,
                                                    enable_reverse_eviction_)) {
        auto move = makeMove(er.node, cascade_group_id, er.source_tier, er.target_tier);
        if (move.blocks_to_release.empty())
            continue;

        if (!prepareMove(move)) {
            restoreSourceHeap(move);
            releaseTargetBlock(move);
            RTP_LLM_LOG_WARNING("BlockTreeEvictor::buildPlan: cascade target alloc failed "
                                "group[%d] tier %s->%s node_key=%ld, skipping",
                                cascade_group_id,
                                tierName(move.source_tier),
                                tierName(move.target_tier),
                                er.node->cache_key);
            continue;
        }
        plan.cascade_moves.push_back(std::move(move));
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
            results.cascade_success = std::vector<bool>(plan.cascade_moves.size(), false);
            return results;
        } else {
            RTP_LLM_LOG_DEBUG("BlockTreeEvictor::performCopy: primary copy OK "
                              "group[%d] node_key=%ld %s->%s target_block=%d",
                              plan.primary.component_group_id,
                              plan.primary.node ? plan.primary.node->cache_key : 0,
                              tierName(plan.primary.source_tier),
                              tierName(plan.primary.target_tier),
                              plan.primary.target_block);
        }
    }

    results.cascade_success.reserve(plan.cascade_moves.size());
    for (const auto& move : plan.cascade_moves) {
        bool copy_ok = true;
        if (move.target_tier != Tier::NONE) {
            copy_ok = executeTierCopy(move);
        }
        results.cascade_success.push_back(copy_ok);

        if (!copy_ok) {
            RTP_LLM_LOG_WARNING("BlockTreeEvictor::performCopy: cascade copy FAILED "
                                "group[%d] node_key=%ld %s->%s",
                                move.component_group_id,
                                move.node ? move.node->cache_key : 0,
                                tierName(move.source_tier),
                                tierName(move.target_tier));
        } else if (move.target_tier != Tier::NONE) {
            RTP_LLM_LOG_DEBUG("BlockTreeEvictor::performCopy: cascade copy OK "
                              "group[%d] node_key=%ld %s->%s target_block=%d",
                              move.component_group_id,
                              move.node ? move.node->cache_key : 0,
                              tierName(move.source_tier),
                              tierName(move.target_tier),
                              move.target_block);
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
        auto& slot  = plan.primary.node->group_slots[primary_gid];
        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::complete: primary group[%d] node_key=%ld source=%s target=%s",
                          plan.primary.component_group_id,
                          plan.primary.node->cache_key,
                          tierName(plan.primary.source_tier),
                          tierName(plan.primary.target_tier));
        group->evictFromTier(plan.primary.node, slot, plan.primary.source_tier);
        releaseBlocks(plan.primary.component_group_id, plan.primary.source_tier, plan.primary.blocks_to_release);
        setTargetSlot(group, slot, plan.primary.node, plan.primary.target_tier, plan.primary.target_block);
    }

    for (size_t i = 0; i < plan.cascade_moves.size(); ++i) {
        const auto& move = plan.cascade_moves[i];
        const bool  ok   = i < results.cascade_success.size() && results.cascade_success[i];
        if (!ok) {
            releaseTargetBlock(move);
            restoreSourceHeap(move);
            continue;
        }

        auto gid = static_cast<size_t>(move.component_group_id);
        if (gid >= component_groups_.size() || move.node == nullptr || gid >= move.node->group_slots.size()) {
            releaseTargetBlock(move);
            continue;
        }

        auto& group = component_groups_[gid];
        auto& slot  = move.node->group_slots[gid];

        RTP_LLM_LOG_DEBUG("BlockTreeEvictor::complete: cascade group[%d] node_key=%ld source=%s target=%s",
                          move.component_group_id,
                          move.node->cache_key,
                          tierName(move.source_tier),
                          tierName(move.target_tier));

        group->evictFromTier(move.node, slot, move.source_tier);
        releaseBlocks(move.component_group_id, move.source_tier, move.blocks_to_release);
        setTargetSlot(group, slot, move.node, move.target_tier, move.target_block);
    }

    finalizeEviction(tree, plan.primary.node);
}

void BlockTreeEvictor::rollbackPreparedPlan(const EvictionPlan& plan) {
    releaseTargetBlock(plan.primary);
    restoreSourceHeap(plan.primary);
    for (const auto& move : plan.cascade_moves) {
        releaseTargetBlock(move);
        restoreSourceHeap(move);
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

bool BlockTreeEvictor::executeTierCopy(const EvictionMove& move) {
    if (!copy_engine_ || move.blocks_to_release.empty() || isNullBlockIdx(move.target_block))
        return false;

    TransferDescriptor desc;
    if (move.source_tier == Tier::DEVICE && move.target_tier == Tier::HOST) {
        desc = TransferDescriptor::deviceToHost(
            move.node, move.component_group_id, move.blocks_to_release, move.target_block);
    } else if (move.source_tier == Tier::HOST && move.target_tier == Tier::DISK) {
        if (isNullBlockIdx(move.blocks_to_release[0]))
            return false;
        desc = TransferDescriptor::hostToDisk(
            move.node, move.component_group_id, move.blocks_to_release[0], move.target_block);
    } else if (move.source_tier == Tier::DISK && move.target_tier == Tier::HOST) {
        if (isNullBlockIdx(move.blocks_to_release[0]))
            return false;
        desc = TransferDescriptor::diskToHost(
            move.node, move.component_group_id, move.blocks_to_release[0], move.target_block);
    } else {
        return false;
    }

    return copy_engine_->submit(desc).result().ok();
}

EvictionMove BlockTreeEvictor::makeMove(TreeNode* node,
                                          int       component_group_id,
                                          Tier      source_tier,
                                          Tier      target_tier) const {
    EvictionMove move;
    move.node               = node;
    move.component_group_id = component_group_id;
    move.source_tier        = source_tier;
    move.target_tier        = target_tier;

    auto gid = static_cast<size_t>(component_group_id);
    if (node == nullptr || gid >= node->group_slots.size())
        return move;

    const auto& slot = node->group_slots[gid];
    switch (source_tier) {
        case Tier::DEVICE:
            if (slot.has_device_value())
                move.blocks_to_release = slot.device_blocks;
            break;
        case Tier::HOST:
            if (slot.has_host_value())
                move.blocks_to_release = {slot.host_block};
            break;
        case Tier::DISK:
            if (slot.has_disk_value())
                move.blocks_to_release = {slot.disk_slot};
            break;
        default:
            break;
    }
    return move;
}

bool BlockTreeEvictor::prepareMove(EvictionMove& move) {
    if (move.node == nullptr || move.blocks_to_release.empty())
        return false;

    reserveSourceHeap(move);
    if (move.target_tier != Tier::NONE) {
        move.target_block = allocateBlock(move.component_group_id, move.target_tier);
        if (isNullBlockIdx(move.target_block))
            return false;
    }

    return true;
}

void BlockTreeEvictor::reserveSourceHeap(const EvictionMove& move) {
    auto gid = static_cast<size_t>(move.component_group_id);
    if (gid >= component_groups_.size() || move.node == nullptr || gid >= move.node->group_slots.size())
        return;

    auto& group = component_groups_[gid];
    auto& slot  = move.node->group_slots[gid];
    if (auto* heap = group->heapForTier(move.source_tier)) {
        heap->invalidate(move.node);
    }

    switch (move.source_tier) {
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

void BlockTreeEvictor::restoreSourceHeap(const EvictionMove& move) {
    auto gid = static_cast<size_t>(move.component_group_id);
    if (gid >= component_groups_.size() || move.node == nullptr) {
        return;
    }

    auto& group = component_groups_[gid];
    switch (move.source_tier) {
        case Tier::DEVICE:
            group->tryAddToDeviceHeap(move.node);
            break;
        case Tier::HOST:
            group->tryAddToHostHeap(move.node);
            break;
        case Tier::DISK:
            group->tryAddToDiskHeap(move.node);
            break;
        default:
            break;
    }
}

void BlockTreeEvictor::releaseTargetBlock(const EvictionMove& move) {
    releaseBlocks(move.component_group_id, move.target_tier, {move.target_block});
}

void BlockTreeEvictor::releaseBlocks(int component_group_id,
                                     Tier tier,
                                     const std::vector<BlockIdxType>& blocks) {
    if (blocks.empty())
        return;

    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= component_groups_.size())
        return;

    auto& group = component_groups_[gid];
    if (tier == Tier::DEVICE) {
        group->releaseDeviceBlocks(blocks);
    } else if (tier == Tier::HOST) {
        if (auto hp = group->hostPool()) {
            for (auto b : blocks)
                if (!isNullBlockIdx(b))
                    hp->free(b);
        }
    } else if (tier == Tier::DISK) {
        if (auto dp = group->diskPool()) {
            for (auto b : blocks)
                if (!isNullBlockIdx(b))
                    dp->free(b);
        }
    }
}

void BlockTreeEvictor::setTargetSlot(ComponentGroupPtr& group,
                                     GroupSlot&         slot,
                                     TreeNode*          node,
                                     Tier               target_tier,
                                     BlockIdxType       target_block) {
    if (isNullBlockIdx(target_block))
        return;
    if (target_tier == Tier::HOST) {
        slot.host_block = target_block;
        if (auto hp = group->hostPool())
            hp->incRef(target_block);
        group->tryAddToHostHeap(node);
    } else if (target_tier == Tier::DISK) {
        slot.disk_slot = target_block;
        if (auto dp = group->diskPool())
            dp->incRef(target_block);
        group->tryAddToDiskHeap(node);
    }
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

std::shared_ptr<HostBlockPool> BlockTreeEvictor::hostPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->hostPool() : nullptr;
}

std::shared_ptr<DiskBlockPool> BlockTreeEvictor::diskPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->diskPool() : nullptr;
}

BlockIdxType BlockTreeEvictor::allocateBlock(int component_group_id, Tier tier) {
    if (tier == Tier::HOST) {
        auto pool = hostPoolForGroup(component_group_id);
        if (!pool)
            return NULL_BLOCK_IDX;
        auto slot = pool->malloc();
        return slot.has_value() ? slot.value() : NULL_BLOCK_IDX;
    }

    if (tier == Tier::DISK) {
        auto pool = diskPoolForGroup(component_group_id);
        if (!pool)
            return NULL_BLOCK_IDX;
        auto slot = pool->malloc();
        return slot.has_value() ? slot.value() : NULL_BLOCK_IDX;
    }

    return NULL_BLOCK_IDX;
}

}  // namespace rtp_llm
