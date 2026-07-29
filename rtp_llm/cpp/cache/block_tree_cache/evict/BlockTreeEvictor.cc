#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

#include <algorithm>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

bool BlockTreeEvictor::EvictionPlan::needsCopy() const {
    return primary.target_tier != Tier::NONE
           || std::any_of(cascade_moves.begin(), cascade_moves.end(), [](const EvictionMove& cascade_move) {
                  return cascade_move.target_tier != Tier::NONE;
              });
}

BlockTreeEvictor::BlockTreeEvictor(std::vector<GroupSetPtr>& group_sets,
                                   ExecuteTransferFn         execute_transfer,
                                   bool                      enable_reverse_eviction):
    group_sets_(group_sets),
    task_runner_(std::make_unique<EvictionTaskRunner>(std::move(execute_transfer))),
    enable_reverse_eviction_(enable_reverse_eviction) {}

BlockTreeEvictor::BlockTreeEvictor(std::vector<GroupSetPtr>&      group_sets,
                                   ExecuteTransferFn              execute_transfer,
                                   bool                           enable_reverse_eviction,
                                   BlockTree*                     tree,
                                   const BlockTransferDispatcher* transfer_dispatcher,
                                   BlockTreeTaskPool*             task_pool,
                                   BlockTreeCacheMetricsReporter& metrics_reporter,
                                   std::mutex&                    mutex,
                                   int                            memory_timeout_ms,
                                   int                            disk_timeout_ms,
                                   IsTierEnabledFn                is_tier_enabled,
                                   CreditsFn                      reserve_credits,
                                   CreditsFn                      settle_credits,
                                   SettledFn                      settled,
                                   RemoteWriteFn                  remote_write):
    group_sets_(group_sets),
    task_runner_(std::make_unique<EvictionTaskRunner>(std::move(execute_transfer),
                                                      group_sets,
                                                      tree,
                                                      transfer_dispatcher,
                                                      task_pool,
                                                      metrics_reporter,
                                                      mutex,
                                                      memory_timeout_ms,
                                                      disk_timeout_ms,
                                                      std::move(is_tier_enabled),
                                                      std::move(reserve_credits),
                                                      std::move(settle_credits),
                                                      std::move(settled),
                                                      std::move(remote_write))),
    enable_reverse_eviction_(enable_reverse_eviction) {}

BlockTreeEvictor::~BlockTreeEvictor() = default;

EvictionTaskRunner& BlockTreeEvictor::taskRunner() {
    return *task_runner_;
}

const EvictionTaskRunner& BlockTreeEvictor::taskRunner() const {
    return *task_runner_;
}

bool BlockTreeEvictor::submitLocked(EvictionMove& eviction_move, std::vector<EvictionReleaseCredit>* release_credits) {
    return task_runner_->submitLocked(*this, eviction_move, release_credits);
}

void BlockTreeEvictor::init(EvictionPolicy device_policy, EvictionPolicy host_policy, EvictionPolicy disk_policy) {
    // GroupSetFactory has already validated that group_set_id equals the vector
    // position. Own one heap per (group set, tier).
    heaps_.clear();
    heaps_.resize(group_sets_.size());
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        auto& tier_heaps  = heaps_[group_set_id];
        tier_heaps.device = std::make_unique<EvictionHeap>(device_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(host_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(disk_policy);
    }
}

EvictionHeap* BlockTreeEvictor::heapFor(size_t group_set_id, Tier tier) {
    if (group_set_id >= heaps_.size()) {
        return nullptr;
    }
    auto& tier_heaps = heaps_[group_set_id];
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

const EvictionHeap* BlockTreeEvictor::heapFor(size_t group_set_id, Tier tier) const {
    if (group_set_id >= heaps_.size()) {
        return nullptr;
    }
    const auto& tier_heaps = heaps_[group_set_id];
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
void BlockTreeEvictor::refreshCandidate(GroupSet& group_set, TreeNode* node, Tier tier) {
    if (node == nullptr || tier == Tier::NONE) {
        return;
    }
    EvictionHeap* heap = heapFor(group_set.groupSetId(), tier);
    if (heap == nullptr) {
        return;
    }

    const size_t group_set_id = group_set.groupSetId();
    if (group_set_id >= node->group_set_resources.size()) {
        heap->erase(node);
        return;
    }
    auto& resource = node->group_set_resources[group_set_id];

    if (resource.transfer_state != GroupSetTransferState::IDLE || !group_set.isEvictable(*node, tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, resource.candidate_meta);
}

void BlockTreeEvictor::refreshCandidate(TreeNode* node, size_t group_set_id) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return;
    }
    const GroupSetPtr& group_set = group_sets_[group_set_id];
    if (group_set == nullptr) {
        return;
    }
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (EvictionHeap* heap = heapFor(group_set_id, tier)) {
            heap->erase(node);
        }
    }
    refreshCandidate(*group_set, node, group_set->getTopTier(node->group_set_resources[group_set_id]));
}

void BlockTreeEvictor::onTierEntered(TreeNode* node, size_t group_set_id, Tier tier) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return;
    }
    const GroupSetPtr& group_set = group_sets_[group_set_id];
    GroupSetResource&  resource  = node->group_set_resources[group_set_id];
    if (group_set == nullptr || group_set->getTopTier(resource) != tier) {
        return;
    }
    resource.candidate_meta.admission_seq      = ++admission_seq_;
    resource.candidate_meta.tier_enter_time_us = currentTimeUs();
    refreshCandidate(node, group_set_id);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInsertCommitted(const BlockTreeInsertResult& result) {
    // An existing empty GroupSetResource may be repopulated independently from the
    // node topology. Existing fills precede the newly created suffix in tree
    // traversal order, so admit them first and preserve that ordering in the
    // eviction policy clocks.
    for (const auto& adopted : result.adopted_resources) {
        if (adopted.node == nullptr || adopted.group_set_id >= group_sets_.size()
            || adopted.group_set_id >= adopted.node->group_set_resources.size()) {
            continue;
        }
        const size_t group_set_id = adopted.group_set_id;
        GroupSetPtr& group_set    = group_sets_[group_set_id];
        if (group_set == nullptr) {
            continue;
        }
        GroupSetResource& resource                 = adopted.node->group_set_resources[group_set_id];
        resource.candidate_meta.last_access_seq    = ++access_seq_;
        resource.candidate_meta.admission_seq      = ++admission_seq_;
        resource.candidate_meta.hit_count          = 0;
        resource.candidate_meta.tier_enter_time_us = currentTimeUs();
        refreshCandidate(*group_set, adopted.node, group_set->getTopTier(resource));

        // A group-set fill can make its direct FULL parent cease to be a leaf
        // at this tier. For other group types the refresh remains idempotent.
        TreeNode* parent = adopted.node->parent;
        if (parent != nullptr && parent->parent != nullptr && group_set_id < parent->group_set_resources.size()) {
            refreshCandidate(*group_set, parent, group_set->getTopTier(parent->group_set_resources[group_set_id]));
        }
    }

    // Every newly inserted node is offered to every group set. FULL's topology
    // predicate filters interior nodes; SWA/LINEAR admit every ready node.
    for (const auto& inserted : result.inserted_nodes) {
        TreeNode* node = inserted.node;
        if (node == nullptr) {
            continue;
        }
        const uint64_t access             = ++access_seq_;
        const uint64_t admit              = ++admission_seq_;
        const int64_t  tier_enter_time_us = currentTimeUs();
        for (auto& group_set : group_sets_) {
            const size_t group_set_id = group_set->groupSetId();
            if (group_set_id >= node->group_set_resources.size()) {
                continue;
            }
            GroupSetResource& resource                 = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq    = access;
            resource.candidate_meta.admission_seq      = admit;
            resource.candidate_meta.hit_count          = 0;
            resource.candidate_meta.tier_enter_time_us = tier_enter_time_us;
            refreshCandidate(*group_set, node, group_set->getTopTier(resource));
        }
    }

    // inserted_nodes contains only newly created nodes. If a new suffix is
    // attached below an existing FULL leaf, its direct parent is not in that
    // list and must be refreshed once. Higher ancestors keep the same direct
    // children, and root never participates in eviction.
    TreeNode* existing_parent = result.inserted_nodes.empty() || result.inserted_nodes.front().node == nullptr ?
                                    nullptr :
                                    result.inserted_nodes.front().node->parent;
    if (existing_parent != nullptr && existing_parent->parent != nullptr) {
        for (auto& group_set : group_sets_) {
            const size_t group_set_id = group_set->groupSetId();
            if (group_set_id >= existing_parent->group_set_resources.size()) {
                continue;
            }
            refreshCandidate(
                *group_set, existing_parent, group_set->getTopTier(existing_parent->group_set_resources[group_set_id]));
        }
    }
}

void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t access = ++access_seq_;
    for (TreeNode* node : path) {
        if (node == nullptr) {
            continue;
        }
        for (auto& group_set : group_sets_) {
            const size_t group_set_id = group_set->groupSetId();
            if (group_set_id >= node->group_set_resources.size()) {
                continue;
            }
            auto&      resource = node->group_set_resources[group_set_id];
            const Tier top      = group_set->getTopTier(resource);
            if (top == Tier::NONE) {
                continue;
            }
            resource.candidate_meta.last_access_seq = access;
            resource.candidate_meta.hit_count++;
            // Only re-sort entries that are already tracked; matching never admits
            // a node on its own (it is protected by the match reference instead).
            EvictionHeap* heap = heapFor(group_set->groupSetId(), top);
            if (heap != nullptr && heap->contains(node)) {
                heap->upsert(node, resource.candidate_meta);
            }
        }
    }
}

void BlockTreeEvictor::refreshCandidatesAfterRelease(const MultiNodeResource& set) {
    const size_t group_set_id = set.group_set_id;
    if (group_set_id >= group_sets_.size()) {
        return;
    }
    auto& group_set = group_sets_[group_set_id];
    for (TreeNode* node : set.tree_nodes) {
        if (node == nullptr || group_set_id >= node->group_set_resources.size()) {
            continue;
        }
        refreshCandidate(*group_set, node, group_set->getTopTier(node->group_set_resources[group_set_id]));
    }
}

void BlockTreeEvictor::refreshAllCandidates(const BlockTree& tree) {
    for (const auto& node_ptr : tree.nodes()) {
        TreeNode* node = node_ptr.get();
        if (node == nullptr) {
            continue;
        }
        for (auto& group_set : group_sets_) {
            const size_t group_set_id = group_set->groupSetId();
            if (group_set_id >= node->group_set_resources.size()) {
                continue;
            }
            refreshCandidate(*group_set, node, group_set->getTopTier(node->group_set_resources[group_set_id]));
        }
    }
}

void BlockTreeEvictor::onTopologyChanged(TreeNode* parent) {
    if (parent == nullptr) {
        return;
    }
    for (auto& group_set : group_sets_) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set_id >= parent->group_set_resources.size()) {
            continue;
        }
        refreshCandidate(*group_set, parent, group_set->getTopTier(parent->group_set_resources[group_set_id]));
    }
}

void BlockTreeEvictor::onNodeAboutToRemove(TreeNode* node) {
    if (node == nullptr) {
        return;
    }
    for (auto& group_set : group_sets_) {
        for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            if (auto* heap = heapFor(group_set->groupSetId(), tier)) {
                heap->erase(node);
            }
        }
    }
}

CandidateStats BlockTreeEvictor::candidateStats() const {
    CandidateStats stats;
    for (const auto& tier_heaps : heaps_) {
        if (tier_heaps.device) {
            stats.device_candidates += tier_heaps.device->size();
        }
        if (tier_heaps.host) {
            stats.host_candidates += tier_heaps.host->size();
        }
        if (tier_heaps.disk) {
            stats.disk_candidates += tier_heaps.disk->size();
        }
    }
    return stats;
}

size_t BlockTreeEvictor::candidateCount(size_t group_set_id, Tier tier) const {
    const EvictionHeap* heap = heapFor(group_set_id, tier);
    return heap == nullptr ? 0 : heap->size();
}

std::vector<TreeNode*> BlockTreeEvictor::candidateNodes(size_t group_set_id, Tier tier) const {
    const EvictionHeap* heap = heapFor(group_set_id, tier);
    return heap == nullptr ? std::vector<TreeNode*>{} : heap->nodes();
}

// ---- Eviction selection ----
std::optional<EvictionMove> BlockTreeEvictor::chooseVictimInGroupSet(GroupSet& group_set, Tier tier) {
    EvictionHeap* heap = heapFor(group_set.groupSetId(), tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    // Exact-update heaps only contain ready candidates. The one remaining race is
    // a node referenced/started after admission: verify then drop (lazy ref) if
    // stale; the release path re-admits it via refreshCandidatesAfterRelease.
    while (true) {
        auto entry = heap->takeBest();
        if (!entry.has_value()) {
            return std::nullopt;
        }

        TreeNode*    node         = entry->node;
        const size_t group_set_id = group_set.groupSetId();
        if (node == nullptr || group_set_id >= node->group_set_resources.size()) {
            continue;
        }

        auto& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_state != GroupSetTransferState::IDLE || !group_set.isEvictable(*node, tier)) {
            continue;  // dropped from heap; will be refreshed on release
        }

        return makeMove(node, group_set.groupSetId(), tier, defaultTargetTier(tier));
    }
}

std::optional<EvictionMove> BlockTreeEvictor::chooseVictim(Tier tier) {
    for (auto& group_set : group_sets_) {
        auto eviction_move = chooseVictimInGroupSet(*group_set, tier);
        if (!eviction_move.has_value()) {
            continue;
        }

        RTP_LLM_LOG_DEBUG("selected candidate, "
                          "group_set[%zu] type=%s tier=%s target=%s node_key=%ld",
                          eviction_move->group_set_id,
                          cacheGroupTypeName(group_set->groupType()),
                          tierName(eviction_move->source_tier),
                          tierName(eviction_move->target_tier),
                          eviction_move->node ? eviction_move->node->cache_key : 0);
        return eviction_move;
    }
    return std::nullopt;
}

std::optional<EvictionMove> BlockTreeEvictor::chooseVictim(size_t group_set_id, Tier tier) {
    if (group_set_id >= group_sets_.size()) {
        return std::nullopt;
    }
    const auto& group_set = group_sets_[group_set_id];
    if (group_set == nullptr || group_set->groupSetId() != group_set_id) {
        return std::nullopt;
    }
    return chooseVictimInGroupSet(*group_set, tier);
}

std::vector<EvictionMove>
BlockTreeEvictor::chooseWatermarkVictims(GroupSet& group_set, Tier tier, double watermark_ratio) {
    std::vector<EvictionMove> victims;
    if (watermark_ratio <= 0.0) {
        return victims;
    }

    size_t excess = computeGroupSetExcess(group_set, tier, watermark_ratio);
    if (excess == 0) {
        return victims;
    }

    RTP_LLM_LOG_INFO("tier=%s group_set[%zu] "
                     "excess=%zu (ratio=%.2f), evicting",
                     tierName(tier),
                     group_set.groupSetId(),
                     excess,
                     watermark_ratio);

    victims.reserve(excess);
    for (size_t i = 0; i < excess; ++i) {
        auto eviction_move = chooseVictimInGroupSet(group_set, tier);
        if (eviction_move.has_value()) {
            victims.push_back(*eviction_move);
        } else {
            break;
        }
    }
    return victims;
}

// ---- Migration pipeline (begin -> copy -> finish) ----
std::optional<BlockTreeEvictor::EvictionPlan> BlockTreeEvictor::buildPlan(EvictionMove eviction_move) {
    EvictionPlan plan;
    if (eviction_move.node == nullptr) {
        return std::nullopt;
    }

    if (!prepareMove(eviction_move)) {
        return std::nullopt;
    }
    plan.primary = eviction_move;

    for (size_t cascade_group_set_id : selectCascadeGroupSets(
             eviction_move.node, eviction_move.group_set_id, eviction_move.source_tier, enable_reverse_eviction_)) {
        auto cascade_move =
            makeMove(eviction_move.node, cascade_group_set_id, eviction_move.source_tier, eviction_move.target_tier);

        const bool had_source = !cascade_move.source_blocks.empty();
        if (!prepareMove(cascade_move)) {
            if (had_source) {
                RTP_LLM_LOG_WARNING("cascade move rejected "
                                    "group_set[%zu] tier %s->%s node_key=%ld, skipping",
                                    cascade_group_set_id,
                                    tierName(cascade_move.source_tier),
                                    tierName(cascade_move.target_tier),
                                    eviction_move.node->cache_key);
            }
            continue;
        }
        plan.cascade_moves.push_back(std::move(cascade_move));
    }

    return plan;
}

void BlockTreeEvictor::complete(BlockTree& tree, const EvictionPlan& plan, const CopyResultSet& results) {
    if (plan.primary.node == nullptr) {
        return;
    }

    if (!results.primary_success) {
        rollbackPreparedPlan(plan);
        return;
    }

    auto primary_group_set_id = plan.primary.group_set_id;
    if (primary_group_set_id < group_sets_.size()
        && primary_group_set_id < plan.primary.node->group_set_resources.size()) {
        auto& group_set = group_sets_[primary_group_set_id];
        RTP_LLM_LOG_DEBUG("primary group_set[%zu] node_key=%ld source=%s target=%s",
                          plan.primary.group_set_id,
                          plan.primary.node->cache_key,
                          tierName(plan.primary.source_tier),
                          tierName(plan.primary.target_tier));
        applyMoveCompletion(group_set, plan.primary);
    }

    for (size_t i = 0; i < plan.cascade_moves.size(); ++i) {
        const auto& cascade_move = plan.cascade_moves[i];
        const bool  ok           = i < results.cascade_success.size() && results.cascade_success[i];
        if (!ok) {
            releaseTargetBlocks(cascade_move);
            restoreSource(cascade_move);
            continue;
        }

        auto group_set_id = cascade_move.group_set_id;
        if (group_set_id >= group_sets_.size() || cascade_move.node == nullptr
            || group_set_id >= cascade_move.node->group_set_resources.size()) {
            releaseTargetBlocks(cascade_move);
            continue;
        }

        auto& group_set = group_sets_[group_set_id];
        RTP_LLM_LOG_DEBUG("cascade group_set[%zu] node_key=%ld source=%s target=%s",
                          cascade_move.group_set_id,
                          cascade_move.node->cache_key,
                          tierName(cascade_move.source_tier),
                          tierName(cascade_move.target_tier));
        applyMoveCompletion(group_set, cascade_move);
    }

    finalizeEviction(tree, plan.primary.node);
}

// Move source blocks out of the resource, install target blocks (if demoting), clear
// the transfer state, and re-admit the node at its new tier. Source blocks were
// held only by cache.
bool BlockTreeEvictor::applyMoveCompletion(GroupSetPtr& group_set, const EvictionMove& move) {
    auto& resource = move.node->group_set_resources[move.group_set_id];
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", move.group_set_id, move.node->cache_key);
        releaseTargetBlocks(move);
        return false;
    }

    if (move.target_tier != Tier::NONE) {
        MultiNodeResource target_holder{move.group_set_id, move.target_tier, {move.target_blocks}};
        group_set->setBlocks(resource, move.target_tier, move.target_blocks);
        group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
        group_set->unreferenceBlocks(target_holder, BlockRefType::EVICTION);
    }

    // DEMOTING is the operation's ownership token. Release its saved source
    // cache hold before clearing the corresponding resource tier. The target is
    // installed while the state is still non-IDLE, then IDLE is published last.
    group_set->unreferenceBlocks(MultiNodeResource{move.group_set_id, move.source_tier, {move.source_blocks}},
                                 BlockRefType::BLOCK_CACHE);
    group_set->evictFromTier(move.node, resource, move.source_tier);
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(group_set->isValidSteadyState(resource),
                            "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            move.group_set_id,
                            move.node->cache_key);

    if (move.target_tier != Tier::NONE) {
        // Section 7.5: keep last_access_seq / hit_count, refresh the admission clock.
        resource.candidate_meta.admission_seq      = ++admission_seq_;
        resource.candidate_meta.tier_enter_time_us = currentTimeUs();
        refreshCandidate(*group_set, move.node, move.target_tier);
    }
    return true;
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
                                          size_t                                 group_set_id) {
    if (!storage_backend) {
        return;
    }

    auto key = std::to_string(cache_key) + "_g" + std::to_string(group_set_id);
    std::vector<std::pair<std::string, std::vector<char>>> items;
    items.emplace_back(std::move(key), std::vector<char>{});
    if (!items.back().second.empty()) {
        storage_backend->batchWrite(items);
        RTP_LLM_LOG_DEBUG("remote write-through "
                          "group_set[%zu] node_key=%ld",
                          group_set_id,
                          cache_key);
    } else {
        RTP_LLM_LOG_WARNING("remote write-through SKIPPED "
                            "(no data serialization yet) group_set[%zu] node_key=%ld",
                            group_set_id,
                            cache_key);
    }
}

EvictionMove BlockTreeEvictor::makeMove(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) const {
    EvictionMove eviction_move;
    eviction_move.node         = node;
    eviction_move.group_set_id = group_set_id;
    eviction_move.source_tier  = source_tier;
    eviction_move.target_tier  = target_tier;

    if (node == nullptr || group_set_id >= node->group_set_resources.size() || group_set_id >= group_sets_.size()) {
        return eviction_move;
    }

    // getBlocks encapsulates the tier-to-resource-field mapping and returns empty for
    // absent values, so the source_blocks.empty() guard still holds.
    eviction_move.source_tier_enter_time_us = node->group_set_resources[group_set_id].candidate_meta.tier_enter_time_us;
    eviction_move.source_blocks =
        group_sets_[group_set_id]->getBlocks(node->group_set_resources[group_set_id], source_tier);
    return eviction_move;
}

bool BlockTreeEvictor::prepareMove(EvictionMove& eviction_move) {
    TreeNode* const node = eviction_move.node;
    if (node == nullptr) {
        return false;
    }

    const size_t group_set_id       = eviction_move.group_set_id;
    auto         erase_stale_source = [&]() {
        if (auto* heap = heapFor(group_set_id, eviction_move.source_tier)) {
            heap->erase(node);
        }
    };

    if (group_set_id >= group_sets_.size() || group_sets_[group_set_id] == nullptr
        || group_sets_[group_set_id]->groupSetId() != group_set_id
        || group_set_id >= node->group_set_resources.size()) {
        erase_stale_source();
        return false;
    }

    auto& group_set         = *group_sets_[group_set_id];
    auto& resource          = node->group_set_resources[group_set_id];
    auto  reject_stale_move = [&]() {
        erase_stale_source();
        const Tier current_tier = group_set.getTopTier(resource);
        if (current_tier == Tier::DEVICE || current_tier == Tier::HOST || current_tier == Tier::DISK) {
            refreshCandidate(group_set, node, current_tier);
        }
        return false;
    };

    const bool source_tier_valid = eviction_move.source_tier == Tier::DEVICE || eviction_move.source_tier == Tier::HOST
                                   || eviction_move.source_tier == Tier::DISK;
    if (!source_tier_valid || eviction_move.source_blocks.empty()
        || resource.transfer_state != GroupSetTransferState::IDLE
        || group_set.getTopTier(resource) != eviction_move.source_tier
        || group_set.getBlocks(resource, eviction_move.source_tier) != eviction_move.source_blocks
        || !group_set.isEvictable(*node, eviction_move.source_tier)) {
        return reject_stale_move();
    }

    reserveSource(eviction_move);
    if (eviction_move.target_tier != Tier::NONE) {
        // The target is an in-flight transfer holder until it is installed.
        BlockIdxType target = group_set.allocateSingleBlock(eviction_move.target_tier, BlockRefType::EVICTION);
        if (isNullBlockIdx(target)) {
            restoreSource(eviction_move);
            return false;
        }
        eviction_move.target_blocks = {target};
    }

    return true;
}

// Reserve the source: exclude it from all heaps and mark the in-flight state so
// no other selector can pick it. The caller must first verify that the resource
// is IDLE; assigning DEMOTING is not safe for an already reserved resource.
void BlockTreeEvictor::reserveSource(const EvictionMove& eviction_move) {
    auto group_set_id = eviction_move.group_set_id;
    if (eviction_move.node == nullptr || group_set_id >= eviction_move.node->group_set_resources.size()) {
        return;
    }
    eviction_move.node->group_set_resources[group_set_id].transfer_state = GroupSetTransferState::DEMOTING;
    if (auto* heap = heapFor(eviction_move.group_set_id, eviction_move.source_tier)) {
        heap->erase(eviction_move.node);
    }
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
bool BlockTreeEvictor::restoreSource(const EvictionMove& eviction_move) {
    auto group_set_id = eviction_move.group_set_id;
    if (eviction_move.node == nullptr || group_set_id >= group_sets_.size()
        || group_set_id >= eviction_move.node->group_set_resources.size()) {
        return false;
    }
    GroupSetResource& resource = eviction_move.node->group_set_resources[group_set_id];
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING(
            "state mismatch, group_set=%zu node_key=%ld", eviction_move.group_set_id, eviction_move.node->cache_key);
        return false;
    }
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(group_sets_[group_set_id]->isValidSteadyState(resource),
                            "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                            eviction_move.group_set_id,
                            eviction_move.node->cache_key);
    refreshCandidate(*group_sets_[group_set_id], eviction_move.node, eviction_move.source_tier);
    return true;
}

void BlockTreeEvictor::releaseTargetBlocks(const EvictionMove& eviction_move) {
    if (eviction_move.target_blocks.empty()) {
        return;
    }
    auto group_set_id = eviction_move.group_set_id;
    if (group_set_id >= group_sets_.size()) {
        return;
    }
    auto& group_set = group_sets_[group_set_id];
    for (auto block : eviction_move.target_blocks) {
        group_set->releaseSingleBlock(eviction_move.target_tier, block, BlockRefType::EVICTION);
    }
}

void BlockTreeEvictor::finalizeEviction(BlockTree& tree, TreeNode* node) {
    if (shouldDeleteNode(tree, node)) {
        RTP_LLM_LOG_DEBUG("deleting empty node key=%ld", node->cache_key);
        TreeNode* parent = node->parent;
        onNodeAboutToRemove(node);  // drop from all heaps before the pointer dies
        tree.removeNode(node);
        TreeNode* surviving_ancestor = tree.removeEmptyAncestors(parent, reusableGroupSetIds());
        if (surviving_ancestor != nullptr && surviving_ancestor != tree.root()) {
            onTopologyChanged(surviving_ancestor);
        }
    } else if (node->parent && node->parent != tree.root()) {
        onTopologyChanged(node->parent);
    }
}

bool BlockTreeEvictor::shouldDeleteNode(const BlockTree& tree, const TreeNode* node) const {
    if (node == nullptr || node == tree.root() || !node->children.empty()) {
        return false;
    }
    for (const auto& group_set : group_sets_) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set_id >= node->group_set_resources.size()
            || !node->group_set_resources[group_set_id].is_removable()) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> BlockTreeEvictor::reusableGroupSetIds() const {
    std::vector<size_t> ids;
    for (const auto& group_set : group_sets_) {
        ids.push_back(group_set->groupSetId());
    }
    return ids;
}

std::vector<size_t> BlockTreeEvictor::selectCascadeGroupSets(const TreeNode* node,
                                                             size_t          source_group_set_id,
                                                             Tier            tier,
                                                             bool            enable_reverse_eviction) const {
    std::vector<size_t> result;

    const GroupSetPtr* source_group_set = nullptr;
    for (const auto& group_set : group_sets_) {
        if (group_set->groupSetId() == source_group_set_id) {
            source_group_set = &group_set;
            break;
        }
    }

    if (source_group_set == nullptr) {
        return result;
    }

    if (enable_reverse_eviction && (*source_group_set)->isLeafAtTier(node, tier)) {
        for (const auto& group_set : group_sets_) {
            if (group_set->groupSetId() != source_group_set_id) {
                result.push_back(group_set->groupSetId());
            }
        }
        return result;
    }

    // Forward cascading is intrinsic to group-set priority: when a high-priority
    // resource is evicted, lower-priority resources on the same node follow.
    CacheGroupType source_type = (*source_group_set)->groupType();
    for (const auto& group_set : group_sets_) {
        bool below = false;
        switch (source_type) {
            case CacheGroupType::FULL:
                below =
                    (group_set->groupType() == CacheGroupType::SWA || group_set->groupType() == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::SWA:
                below = (group_set->groupType() == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::LINEAR:
                below = false;
                break;
        }
        if (below) {
            result.push_back(group_set->groupSetId());
        }
    }
    return result;
}

size_t BlockTreeEvictor::computeGroupSetExcess(const GroupSet& group_set, Tier tier, double ratio) const {
    if (tier == Tier::DEVICE) {
        return group_set.devicePoolMaxExcess(ratio);
    }
    size_t capacity = (tier == Tier::HOST) ? group_set.hostPoolCapacity() : group_set.diskPoolCapacity();
    if (capacity == 0) {
        return 0;
    }
    size_t used      = (tier == Tier::HOST) ? group_set.hostPoolUsed() : group_set.diskPoolUsed();
    size_t threshold = static_cast<size_t>(capacity * ratio);
    return (used > threshold) ? (used - threshold) : 0;
}

}  // namespace rtp_llm
