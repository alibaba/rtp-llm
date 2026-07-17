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

bool BlockTreeEvictor::init(const std::vector<Component>& components,
                            EvictionPolicy               device_policy,
                            EvictionPolicy               host_policy,
                            EvictionPolicy               disk_policy) {
    // Own one heap per (group, tier), using the cache-wide per-tier policies.
    heaps_.clear();
    heaps_.resize(component_groups_.size());
    group_layer_tag_slots_.clear();
    for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
        const auto& group = component_groups_[group_index];
        if (group == nullptr) {
            RTP_LLM_LOG_ERROR("BlockTreeEvictor::init: component group must not be null");
            heaps_.clear();
            return false;
        }
        const int gid = group->component_group_id;
        if (gid < 0 || static_cast<size_t>(gid) >= component_groups_.size()) {
            RTP_LLM_LOG_ERROR("BlockTreeEvictor::init: invalid component_group_id=%d, group_count=%zu",
                              gid,
                              component_groups_.size());
            heaps_.clear();
            return false;
        }
        if (static_cast<size_t>(gid) != group_index) {
            RTP_LLM_LOG_ERROR("BlockTreeEvictor::init: component_group_id=%d must equal vector index=%zu",
                              gid,
                              group_index);
            heaps_.clear();
            return false;
        }
        auto& tier_heaps = heaps_[static_cast<size_t>(gid)];
        if (tier_heaps.device || tier_heaps.host || tier_heaps.disk) {
            RTP_LLM_LOG_ERROR("BlockTreeEvictor::init: duplicate component_group_id=%d", gid);
            heaps_.clear();
            return false;
        }
        tier_heaps.device = std::make_unique<EvictionHeap>(device_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(host_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(disk_policy);
    }

    buildGroupLayerTagSlots(components);
    return true;
}

void BlockTreeEvictor::buildGroupLayerTagSlots(const std::vector<Component>& components) {
    group_layer_tag_slots_.clear();
    group_layer_tag_slots_.resize(component_groups_.size());

    for (const auto& group : component_groups_) {
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

EvictionHeap* BlockTreeEvictor::heapFor(int group_id, Tier tier) {
    if (group_id < 0 || static_cast<size_t>(group_id) >= heaps_.size())
        return nullptr;
    auto& tier_heaps = heaps_[static_cast<size_t>(group_id)];
    switch (tier) {
        case Tier::DEVICE:
            return tier_heaps.device.get();
        case Tier::HOST:
            return tier_heaps.host.get();
        case Tier::DISK:
            return tier_heaps.disk.get();
        default:
            return nullptr;
    }
}

const EvictionHeap* BlockTreeEvictor::heapFor(int group_id, Tier tier) const {
    if (group_id < 0 || static_cast<size_t>(group_id) >= heaps_.size())
        return nullptr;
    const auto& tier_heaps = heaps_[static_cast<size_t>(group_id)];
    switch (tier) {
        case Tier::DEVICE:
            return tier_heaps.device.get();
        case Tier::HOST:
            return tier_heaps.host.get();
        case Tier::DISK:
            return tier_heaps.disk.get();
        default:
            return nullptr;
    }
}

Tier BlockTreeEvictor::defaultTargetTier(Tier source) {
    switch (source) {
        case Tier::DEVICE:
            return Tier::HOST;
        case Tier::HOST:
            return Tier::DISK;
        default:
            return Tier::NONE;
    }
}

// ---- Candidate eligibility gate (design section 4.3) ----
void BlockTreeEvictor::refreshCandidate(ComponentGroup& group, TreeNode* node, Tier tier) {
    if (node == nullptr || tier == Tier::NONE)
        return;
    EvictionHeap* heap = heapFor(group.component_group_id, tier);
    if (heap == nullptr)
        return;

    auto gid = static_cast<size_t>(group.component_group_id);
    if (gid >= node->group_slots.size()) {
        heap->erase(node);
        return;
    }
    auto& slot = node->group_slots[gid];

    if (slot.transfer_state != SlotTransferState::IDLE || !group.isSlotEvictable(*node, tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, slot.candidate_meta);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInsertCommitted(const BlockTreeInsertResult& result) {
    // Newly created nodes enter their current serving tier now; stamp meta clocks.
    for (const auto& inserted : result.inserted_nodes) {
        TreeNode* node = inserted.node;
        if (node == nullptr)
            continue;
        const uint64_t access = ++access_seq_;
        const uint64_t admit  = ++admission_seq_;
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            auto& slot                          = node->group_slots[gid];
            slot.candidate_meta.last_access_seq = access;
            slot.candidate_meta.admission_seq   = admit;
            slot.candidate_meta.hit_count       = 0;
        }
    }

    // Every newly inserted node is offered to every group. FULL's topology
    // predicate filters interior nodes; SWA/LINEAR admit every ready node.
    for (const auto& inserted : result.inserted_nodes) {
        TreeNode* node = inserted.node;
        if (node == nullptr)
            continue;
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            refreshCandidate(*group, node, group->getTopTier(node->group_slots[gid]));
        }
    }

    // inserted_nodes contains only newly created nodes. If a new suffix is
    // attached below an existing FULL leaf, its direct parent is not in that
    // list and must be refreshed once. Higher ancestors keep the same direct
    // children, and root never participates in eviction.
    TreeNode* existing_parent =
        result.inserted_nodes.empty() || result.inserted_nodes.front().node == nullptr
            ? nullptr
            : result.inserted_nodes.front().node->parent;
    if (existing_parent != nullptr && existing_parent->parent != nullptr) {
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid >= existing_parent->group_slots.size())
                continue;
            refreshCandidate(*group, existing_parent, group->getTopTier(existing_parent->group_slots[gid]));
        }
    }
}

void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t access = ++access_seq_;
    for (TreeNode* node : path) {
        if (node == nullptr)
            continue;
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            auto&      slot = node->group_slots[gid];
            const Tier top  = group->getTopTier(slot);
            if (top == Tier::NONE)
                continue;
            slot.candidate_meta.last_access_seq = access;
            slot.candidate_meta.hit_count++;
            // Only re-sort entries that are already tracked; matching never admits
            // a node on its own (it is protected by the match reference instead).
            EvictionHeap* heap = heapFor(group->component_group_id, top);
            if (heap != nullptr && heap->contains(node)) {
                heap->upsert(node, slot.candidate_meta);
            }
        }
    }
}

void BlockTreeEvictor::refreshCandidatesAfterRelease(const GroupBlockSet& set) {
    auto gid = static_cast<size_t>(set.component_group_id);
    if (gid >= component_groups_.size())
        return;
    auto& group = component_groups_[gid];
    for (TreeNode* node : set.nodes) {
        if (node == nullptr || gid >= node->group_slots.size())
            continue;
        refreshCandidate(*group, node, group->getTopTier(node->group_slots[gid]));
    }
}

void BlockTreeEvictor::onTopologyChanged(TreeNode* parent) {
    if (parent == nullptr)
        return;
    for (auto& group : component_groups_) {
        auto gid = static_cast<size_t>(group->component_group_id);
        if (gid >= parent->group_slots.size())
            continue;
        refreshCandidate(*group, parent, group->getTopTier(parent->group_slots[gid]));
    }
}

void BlockTreeEvictor::onNodeAboutToRemove(TreeNode* node) {
    if (node == nullptr)
        return;
    for (auto& group : component_groups_) {
        for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            if (auto* heap = heapFor(group->component_group_id, tier)) {
                heap->erase(node);
            }
        }
    }
}

CandidateStats BlockTreeEvictor::candidateStats() const {
    CandidateStats stats;
    for (const auto& tier_heaps : heaps_) {
        if (tier_heaps.device)
            stats.device_candidates += tier_heaps.device->size();
        if (tier_heaps.host)
            stats.host_candidates += tier_heaps.host->size();
        if (tier_heaps.disk)
            stats.disk_candidates += tier_heaps.disk->size();
    }
    return stats;
}

// ---- Eviction selection ----
std::optional<EvictionMove> BlockTreeEvictor::chooseVictimInGroup(ComponentGroup& group, Tier tier) {
    EvictionHeap* heap = heapFor(group.component_group_id, tier);
    if (heap == nullptr)
        return std::nullopt;

    // Exact-update heaps only contain ready candidates. The one remaining race is
    // a node referenced/started after admission: verify then drop (lazy ref) if
    // stale; the release path re-admits it via refreshCandidatesAfterRelease.
    while (true) {
        auto entry = heap->takeBest();
        if (!entry.has_value())
            return std::nullopt;

        TreeNode* node = entry->node;
        auto      gid  = static_cast<size_t>(group.component_group_id);
        if (node == nullptr || gid >= node->group_slots.size())
            continue;

        auto& slot = node->group_slots[gid];
        if (slot.transfer_state != SlotTransferState::IDLE || !group.isSlotEvictable(*node, tier)) {
            continue;  // dropped from heap; will be refreshed on release
        }

        return makeMove(node, group.component_group_id, tier, defaultTargetTier(tier));
    }
}

std::optional<EvictionMove> BlockTreeEvictor::chooseVictim(Tier tier) {
    for (auto& group : component_groups_) {
        auto eviction_move = chooseVictimInGroup(*group, tier);
        if (!eviction_move.has_value())
            continue;

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

std::optional<EvictionMove> BlockTreeEvictor::chooseVictim(int component_group_id, Tier tier) {
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= component_groups_.size()) {
        return std::nullopt;
    }
    const auto& group = component_groups_[static_cast<size_t>(component_group_id)];
    if (group == nullptr || group->component_group_id != component_group_id) {
        return std::nullopt;
    }
    return chooseVictimInGroup(*group, tier);
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
        auto eviction_move = chooseVictimInGroup(group, tier);
        if (eviction_move.has_value())
            victims.push_back(*eviction_move);
        else
            break;
    }
    return victims;
}

// ---- Migration pipeline (begin -> copy -> finish) ----
std::optional<BlockTreeEvictor::EvictionPlan> BlockTreeEvictor::buildPlan(EvictionMove eviction_move) {
    EvictionPlan plan;
    if (eviction_move.node == nullptr)
        return std::nullopt;

    if (!prepareMove(eviction_move)) {
        restoreSource(eviction_move);
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
            restoreSource(cascade_move);
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
        const bool  ok           = i < results.cascade_success.size() && results.cascade_success[i];
        if (!ok) {
            releaseTargetBlocks(cascade_move);
            restoreSource(cascade_move);
            continue;
        }

        auto gid = static_cast<size_t>(cascade_move.component_group_id);
        if (gid >= component_groups_.size() || cascade_move.node == nullptr
            || gid >= cascade_move.node->group_slots.size()) {
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

// Move source blocks out of the slot, install target blocks (if demoting), clear
// the transfer state, and re-admit the node at its new tier. Source blocks were
// held only by cache.
void BlockTreeEvictor::applyMoveCompletion(ComponentGroupPtr& group, const EvictionMove& move) {
    auto& slot = move.node->group_slots[static_cast<size_t>(move.component_group_id)];
    // Release source cache-hold (prepare-time snapshot) before clearing the slot.
    group->unreferenceBlocks(GroupBlockSet{move.component_group_id, move.source_tier, {move.source_blocks}});
    group->evictFromTier(move.node, slot, move.source_tier);
    slot.transfer_state = SlotTransferState::IDLE;

    if (move.target_tier != Tier::NONE) {
        group->setBlocks(slot, move.target_tier, move.target_blocks);
        // Section 7.5: keep last_access_seq / hit_count, refresh the admission clock.
        slot.candidate_meta.admission_seq = ++admission_seq_;
        refreshCandidate(*group, move.node, move.target_tier);
    }
}

void BlockTreeEvictor::rollbackPreparedPlan(const EvictionPlan& plan) {
    releaseTargetBlocks(plan.primary);
    restoreSource(plan.primary);
    for (const auto& cascade_move : plan.cascade_moves) {
        releaseTargetBlocks(cascade_move);
        restoreSource(cascade_move);
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

// ---- Load-back transitions ----
bool BlockTreeEvictor::beginLoadBack(TreeNode* node, int group_id, Tier source) {
    auto gid = static_cast<size_t>(group_id);
    if (node == nullptr || gid >= component_groups_.size() || gid >= node->group_slots.size())
        return false;
    auto& group = component_groups_[gid];
    auto& slot  = node->group_slots[gid];
    if (group == nullptr || (source != Tier::HOST && source != Tier::DISK) || group->getTopTier(slot) != source)
        return false;
    if (slot.transfer_state != SlotTransferState::IDLE)
        return false;  // in-flight protection (section 7.3): do not double-initiate
    if (auto* heap = heapFor(group_id, source)) {
        heap->erase(node);
    }
    slot.transfer_state = SlotTransferState::LOADING_BACK;
    return true;
}

void BlockTreeEvictor::finishLoadBack(TreeNode* node, int group_id, Tier source, bool copy_ok) {
    auto gid = static_cast<size_t>(group_id);
    if (node == nullptr || gid >= component_groups_.size() || gid >= node->group_slots.size())
        return;
    auto& group         = component_groups_[gid];
    auto& slot          = node->group_slots[gid];
    slot.transfer_state = SlotTransferState::IDLE;
    if (copy_ok) {
        slot.candidate_meta.admission_seq = ++admission_seq_;
        refreshCandidate(*group, node, Tier::DEVICE);
    } else {
        refreshCandidate(*group, node, source);
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

EvictionMove
BlockTreeEvictor::makeMove(TreeNode* node, int component_group_id, Tier source_tier, Tier target_tier) const {
    EvictionMove eviction_move;
    eviction_move.node               = node;
    eviction_move.component_group_id = component_group_id;
    eviction_move.source_tier        = source_tier;
    eviction_move.target_tier        = target_tier;

    auto gid = static_cast<size_t>(component_group_id);
    if (node == nullptr || gid >= node->group_slots.size() || gid >= component_groups_.size())
        return eviction_move;

    // getBlocks encapsulates the tier->slot-field mapping and returns empty for
    // absent values, so the source_blocks.empty() guard still holds.
    eviction_move.source_blocks = component_groups_[gid]->getBlocks(node->group_slots[gid], source_tier);
    return eviction_move;
}

bool BlockTreeEvictor::prepareMove(EvictionMove& eviction_move) {
    if (eviction_move.node == nullptr || eviction_move.source_blocks.empty())
        return false;

    reserveSource(eviction_move);
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

// Reserve the source: exclude it from all heaps and mark the in-flight state so
// no other selector can pick it. Idempotent (heap.erase / state assignment).
void BlockTreeEvictor::reserveSource(const EvictionMove& eviction_move) {
    auto gid = static_cast<size_t>(eviction_move.component_group_id);
    if (eviction_move.node == nullptr || gid >= eviction_move.node->group_slots.size())
        return;
    eviction_move.node->group_slots[gid].transfer_state = SlotTransferState::DEMOTING;
    if (auto* heap = heapFor(eviction_move.component_group_id, eviction_move.source_tier)) {
        heap->erase(eviction_move.node);
    }
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
void BlockTreeEvictor::restoreSource(const EvictionMove& eviction_move) {
    auto gid = static_cast<size_t>(eviction_move.component_group_id);
    if (eviction_move.node == nullptr || gid >= component_groups_.size()
        || gid >= eviction_move.node->group_slots.size())
        return;
    eviction_move.node->group_slots[gid].transfer_state = SlotTransferState::IDLE;
    refreshCandidate(*component_groups_[gid], eviction_move.node, eviction_move.source_tier);
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
        onNodeAboutToRemove(node);  // drop from all heaps before the pointer dies
        tree.removeNode(node);
        TreeNode* surviving_ancestor = tree.removeEmptyAncestors(parent, reusableGroupIds());
        if (surviving_ancestor != nullptr && surviving_ancestor != tree.root()) {
            onTopologyChanged(surviving_ancestor);
        }
    } else if (node->parent && node->parent != tree.root()) {
        onTopologyChanged(node->parent);
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

std::vector<int> BlockTreeEvictor::selectCascadeGroups(const TreeNode* node,
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

    if (enable_reverse_eviction && source_group != nullptr
        && (*source_group)->isLeafAtTier(node, source_group_id, tier)) {
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
