#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

#include <algorithm>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

bool BlockTreeEvictor::EvictionPlan::needsCopy() const {
    return primary_desc.target_tier != Tier::NONE
           || std::any_of(cascade_descs.begin(), cascade_descs.end(), [](const TransferDescriptor& cascade_desc) {
                  return cascade_desc.target_tier != Tier::NONE;
              });
}

BlockTreeEvictor::BlockTreeEvictor(BlockTree*                     tree,
                                   ExecuteTransferFn              execute_transfer,
                                   bool                           enable_reverse_eviction,
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
    tree_(tree),
    task_runner_(std::make_unique<EvictionTaskRunner>(std::move(execute_transfer),
                                                      tree->groupSets(),
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

bool BlockTreeEvictor::submitLocked(TransferDescriptor&                 eviction_desc,
                                    std::vector<EvictionReleaseCredit>* release_credits) {
    return task_runner_->submitLocked(*this, eviction_desc, release_credits);
}

void BlockTreeEvictor::init(EvictionPolicy device_policy, EvictionPolicy host_policy, EvictionPolicy disk_policy) {
    // GroupSetFactory has already validated that group_set_id equals the vector
    // position. Own one heap per (group set, tier).
    heaps_.clear();
    heaps_.resize(tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
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

bool BlockTreeEvictor::isEvictable(const GroupSet& group_set, const TreeNode* node, Tier tier) const {
    const size_t            group_set_id = group_set.groupSetId();
    const GroupSetResource& resource     = node->group_set_resources[group_set_id];
    return group_set.isEvictable(resource, tier)
           && (group_set.groupType() != CacheGroupType::FULL || tree_->isLeafAtTier(node, group_set_id, tier));
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

    if (resource.transfer_state != GroupSetTransferState::IDLE || !isEvictable(group_set, node, tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, resource.candidate_meta);
}

void BlockTreeEvictor::refreshCandidate(TreeNode* node, size_t group_set_id) {
    if (node == nullptr || group_set_id >= tree_->groupSets().size() || group_set_id >= node->group_set_resources.size()) {
        return;
    }
    const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
    if (group_set == nullptr) {
        return;
    }
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (EvictionHeap* heap = heapFor(group_set_id, tier)) {
            heap->erase(node);
        }
    }
    refreshCandidate(*group_set, node, node->group_set_resources[group_set_id].getTopTier());
}

void BlockTreeEvictor::onTierEntered(TreeNode* node, size_t group_set_id, Tier tier) {
    if (node == nullptr || group_set_id >= tree_->groupSets().size() || group_set_id >= node->group_set_resources.size()) {
        return;
    }
    const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
    GroupSetResource&  resource  = node->group_set_resources[group_set_id];
    if (group_set == nullptr || resource.getTopTier() != tier) {
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
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
            GroupSetResource& resource                 = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq    = ++access_seq_;
            resource.candidate_meta.admission_seq      = ++admission_seq_;
            resource.candidate_meta.hit_count          = 0;
            resource.candidate_meta.tier_enter_time_us = currentTimeUs();
            refreshCandidate(*group_set, adopted.first, resource.getTopTier());

            TreeNode* parent = adopted.first->parent;
            if (parent->parent != nullptr) {
                refreshCandidate(*group_set, parent, parent->group_set_resources[group_set_id].getTopTier());
            }
        }
    }

    // Every newly inserted node is offered to every group set. FULL's topology
    // predicate filters interior nodes; SWA/LINEAR admit every ready node.
    for (TreeNode* node : result.inserted_nodes) {
        const uint64_t access             = ++access_seq_;
        const uint64_t admit              = ++admission_seq_;
        const int64_t  tier_enter_time_us = currentTimeUs();
        for (auto& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            GroupSetResource& resource                 = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq    = access;
            resource.candidate_meta.admission_seq      = admit;
            resource.candidate_meta.hit_count          = 0;
            resource.candidate_meta.tier_enter_time_us = tier_enter_time_us;
            refreshCandidate(*group_set, node, resource.getTopTier());
        }
    }

    // inserted_nodes contains only newly created nodes. If a new suffix is
    // attached below an existing FULL leaf, its direct parent is not in that
    // list and must be refreshed once. Higher ancestors keep the same direct
    // children, and root never participates in eviction.
    TreeNode* existing_parent =
        result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent;
    if (existing_parent != nullptr && existing_parent->parent != nullptr) {
        for (auto& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            refreshCandidate(
                *group_set, existing_parent, existing_parent->group_set_resources[group_set_id].getTopTier());
        }
    }
}

void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t access = ++access_seq_;
    for (TreeNode* node : path) {
        if (node == nullptr) {
            continue;
        }
        for (auto& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            if (group_set_id >= node->group_set_resources.size()) {
                continue;
            }
            auto&      resource = node->group_set_resources[group_set_id];
            const Tier top      = resource.getTopTier();
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
    if (group_set_id >= tree_->groupSets().size()) {
        return;
    }
    auto& group_set = tree_->groupSets()[group_set_id];
    for (const auto& [node, _] : set.node_blocks) {
        if (group_set_id >= node->group_set_resources.size()) {
            continue;
        }
        refreshCandidate(*group_set, node, node->group_set_resources[group_set_id].getTopTier());
    }
}

void BlockTreeEvictor::onTopologyChanged(TreeNode* parent) {
    if (parent == nullptr) {
        return;
    }
    for (auto& group_set : tree_->groupSets()) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set_id >= parent->group_set_resources.size()) {
            continue;
        }
        refreshCandidate(*group_set, parent, parent->group_set_resources[group_set_id].getTopTier());
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
std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictimInGroupSet(GroupSet& group_set, Tier tier) {
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
        if (resource.transfer_state != GroupSetTransferState::IDLE || !isEvictable(group_set, node, tier)) {
            continue;  // dropped from heap; will be refreshed on release
        }

        return makeDesc(node, group_set.groupSetId(), tier, defaultTargetTier(tier));
    }
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(Tier tier) {
    for (auto& group_set : tree_->groupSets()) {
        auto eviction_desc = chooseVictimInGroupSet(*group_set, tier);
        if (!eviction_desc.has_value()) {
            continue;
        }

        RTP_LLM_LOG_DEBUG("selected candidate, "
                          "group_set[%zu] type=%s tier=%s target=%s node_key=%ld",
                          eviction_desc->group_set_id,
                          cacheGroupTypeName(group_set->groupType()),
                          tierName(eviction_desc->source_tier),
                          tierName(eviction_desc->target_tier),
                          eviction_desc->node ? eviction_desc->node->cache_key : 0);
        return eviction_desc;
    }
    return std::nullopt;
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(size_t group_set_id, Tier tier) {
    if (group_set_id >= tree_->groupSets().size()) {
        return std::nullopt;
    }
    const auto& group_set = tree_->groupSets()[group_set_id];
    if (group_set == nullptr || group_set->groupSetId() != group_set_id) {
        return std::nullopt;
    }
    return chooseVictimInGroupSet(*group_set, tier);
}

std::vector<TransferDescriptor>
BlockTreeEvictor::chooseWatermarkVictims(GroupSet& group_set, Tier tier, double watermark_ratio) {
    std::vector<TransferDescriptor> victims;
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
        auto eviction_desc = chooseVictimInGroupSet(group_set, tier);
        if (eviction_desc.has_value()) {
            victims.push_back(*eviction_desc);
        } else {
            break;
        }
    }
    return victims;
}

// ---- Migration pipeline (begin -> copy -> finish) ----
std::optional<BlockTreeEvictor::EvictionPlan> BlockTreeEvictor::buildPlan(TransferDescriptor eviction_desc) {
    EvictionPlan plan;
    if (eviction_desc.node == nullptr) {
        return std::nullopt;
    }

    if (!prepareDesc(eviction_desc)) {
        return std::nullopt;
    }
    plan.primary_desc = eviction_desc;

    for (size_t cascade_group_set_id : selectCascadeGroupSets(
             eviction_desc.node, eviction_desc.group_set_id, eviction_desc.source_tier, enable_reverse_eviction_)) {
        auto cascade_desc =
            makeDesc(eviction_desc.node, cascade_group_set_id, eviction_desc.source_tier, eviction_desc.target_tier);

        const bool had_source = !cascade_desc.source_blocks.empty();
        if (!prepareDesc(cascade_desc)) {
            if (had_source) {
                RTP_LLM_LOG_WARNING("cascade move rejected "
                                    "group_set[%zu] tier %s->%s node_key=%ld, skipping",
                                    cascade_group_set_id,
                                    tierName(cascade_desc.source_tier),
                                    tierName(cascade_desc.target_tier),
                                    eviction_desc.node->cache_key);
            }
            continue;
        }
        plan.cascade_descs.push_back(std::move(cascade_desc));
    }

    return plan;
}

void BlockTreeEvictor::complete(const EvictionPlan& plan, const CopyResultSet& results) {
    if (plan.primary_desc.node == nullptr) {
        return;
    }

    if (!results.primary_success) {
        rollbackPreparedPlan(plan);
        return;
    }

    auto primary_group_set_id = plan.primary_desc.group_set_id;
    if (primary_group_set_id < tree_->groupSets().size()
        && primary_group_set_id < plan.primary_desc.node->group_set_resources.size()) {
        auto& group_set = tree_->groupSets()[primary_group_set_id];
        RTP_LLM_LOG_DEBUG("primary group_set[%zu] node_key=%ld source=%s target=%s",
                          plan.primary_desc.group_set_id,
                          plan.primary_desc.node->cache_key,
                          tierName(plan.primary_desc.source_tier),
                          tierName(plan.primary_desc.target_tier));
        applyDescCompletion(group_set, plan.primary_desc);
    }

    for (size_t i = 0; i < plan.cascade_descs.size(); ++i) {
        const auto& cascade_desc = plan.cascade_descs[i];
        const bool  ok           = i < results.cascade_success.size() && results.cascade_success[i];
        if (!ok) {
            releaseTargetBlocks(cascade_desc);
            restoreSource(cascade_desc);
            continue;
        }

        auto group_set_id = cascade_desc.group_set_id;
        if (group_set_id >= tree_->groupSets().size() || cascade_desc.node == nullptr
            || group_set_id >= cascade_desc.node->group_set_resources.size()) {
            releaseTargetBlocks(cascade_desc);
            continue;
        }

        auto& group_set = tree_->groupSets()[group_set_id];
        RTP_LLM_LOG_DEBUG("cascade group_set[%zu] node_key=%ld source=%s target=%s",
                          cascade_desc.group_set_id,
                          cascade_desc.node->cache_key,
                          tierName(cascade_desc.source_tier),
                          tierName(cascade_desc.target_tier));
        applyDescCompletion(group_set, cascade_desc);
    }

    finalizeEviction(plan.primary_desc.node);
}

// Move source blocks out of the resource, install target blocks (if demoting), clear
// the transfer state, and re-admit the node at its new tier. Source blocks were
// held only by cache.
bool BlockTreeEvictor::applyDescCompletion(const GroupSetPtr& group_set, const TransferDescriptor& eviction_desc) {
    auto& resource = eviction_desc.node->group_set_resources[eviction_desc.group_set_id];
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", eviction_desc.group_set_id, eviction_desc.node->cache_key);
        releaseTargetBlocks(eviction_desc);
        return false;
    }

    if (eviction_desc.target_tier != Tier::NONE) {
        MultiNodeResource target_holder{
            eviction_desc.group_set_id, eviction_desc.target_tier, {{eviction_desc.node, eviction_desc.target_blocks}}};
        resource.setBlocks(eviction_desc.target_tier, eviction_desc.target_blocks);
        if (eviction_desc.target_tier == Tier::DEVICE) {
            group_set->mapDeviceBlocksToTreeNode(target_holder);
        }
        group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
        group_set->unreferenceBlocks(target_holder, BlockRefType::EVICTION);
    }

    // DEMOTING is the operation's ownership token. Release its saved source
    // cache hold before clearing the corresponding resource tier. The target is
    // installed while the state is still non-IDLE, then IDLE is published last.
    const MultiNodeResource source_holder{
        eviction_desc.group_set_id,
        eviction_desc.source_tier,
        {{eviction_desc.node, eviction_desc.source_blocks}}};
    if (eviction_desc.source_tier == Tier::DEVICE) {
        group_set->unmapDeviceBlocksFromTreeNode(source_holder);
    }
    group_set->unreferenceBlocks(source_holder, BlockRefType::BLOCK_CACHE);
    resource.evictFromTier(eviction_desc.source_tier);
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            eviction_desc.group_set_id,
                            eviction_desc.node->cache_key);

    if (eviction_desc.target_tier != Tier::NONE) {
        // Section 7.5: keep last_access_seq / hit_count, refresh the admission clock.
        resource.candidate_meta.admission_seq      = ++admission_seq_;
        resource.candidate_meta.tier_enter_time_us = currentTimeUs();
        refreshCandidate(*group_set, eviction_desc.node, eviction_desc.target_tier);
    }
    return true;
}

void BlockTreeEvictor::rollbackPreparedPlan(const EvictionPlan& plan) {
    releaseTargetBlocks(plan.primary_desc);
    restoreSource(plan.primary_desc);
    for (const auto& cascade_desc : plan.cascade_descs) {
        releaseTargetBlocks(cascade_desc);
        restoreSource(cascade_desc);
    }
}

void BlockTreeEvictor::writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                                          CacheKeyType                           cache_key,
                                          size_t                                 group_set_id) {
    if (!storage_backend) {
        return;
    }

    auto key = std::to_string(cache_key) + "_g" + std::to_string(group_set_id);
    std::vector<std::pair<std::string, std::vector<char>>> entries;
    entries.emplace_back(std::move(key), std::vector<char>{});
    if (!entries.back().second.empty()) {
        storage_backend->batchWrite(entries);
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

TransferDescriptor
BlockTreeEvictor::makeDesc(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) const {
    TransferDescriptor eviction_desc;
    eviction_desc.node         = node;
    eviction_desc.group_set_id = group_set_id;
    eviction_desc.source_tier  = source_tier;
    eviction_desc.target_tier  = target_tier;

    if (node == nullptr || group_set_id >= node->group_set_resources.size() || group_set_id >= tree_->groupSets().size()) {
        return eviction_desc;
    }

    // getBlocks encapsulates the tier-to-resource-field mapping and returns empty for
    // absent values, so the source_blocks.empty() guard still holds.
    eviction_desc.source_tier_enter_time_us =
        node->group_set_resources[group_set_id].candidate_meta.tier_enter_time_us;
    eviction_desc.source_blocks = node->group_set_resources[group_set_id].getBlocks(source_tier);
    return eviction_desc;
}

bool BlockTreeEvictor::prepareDesc(TransferDescriptor& eviction_desc) {
    TreeNode* const node = eviction_desc.node;
    if (node == nullptr) {
        return false;
    }

    const size_t group_set_id       = eviction_desc.group_set_id;
    auto         erase_stale_source = [&]() {
        if (auto* heap = heapFor(group_set_id, eviction_desc.source_tier)) {
            heap->erase(node);
        }
    };

    if (group_set_id >= tree_->groupSets().size() || tree_->groupSets()[group_set_id] == nullptr
        || tree_->groupSets()[group_set_id]->groupSetId() != group_set_id
        || group_set_id >= node->group_set_resources.size()) {
        erase_stale_source();
        return false;
    }

    auto& group_set         = *tree_->groupSets()[group_set_id];
    auto& resource          = node->group_set_resources[group_set_id];
    auto  reject_stale_desc = [&]() {
        erase_stale_source();
        const Tier current_tier = resource.getTopTier();
        if (current_tier == Tier::DEVICE || current_tier == Tier::HOST || current_tier == Tier::DISK) {
            refreshCandidate(group_set, node, current_tier);
        }
        return false;
    };

    const bool source_tier_valid = eviction_desc.source_tier == Tier::DEVICE || eviction_desc.source_tier == Tier::HOST
                                   || eviction_desc.source_tier == Tier::DISK;
    if (!source_tier_valid || eviction_desc.source_blocks.empty()
        || resource.transfer_state != GroupSetTransferState::IDLE
        || resource.getTopTier() != eviction_desc.source_tier
        || resource.getBlocks(eviction_desc.source_tier) != eviction_desc.source_blocks
        || !isEvictable(group_set, node, eviction_desc.source_tier)) {
        return reject_stale_desc();
    }

    reserveSource(eviction_desc);
    if (eviction_desc.target_tier != Tier::NONE) {
        // The target is an in-flight transfer holder until it is installed.
        BlockIdxType target = group_set.allocateSingleBlock(eviction_desc.target_tier, BlockRefType::EVICTION);
        if (isNullBlockIdx(target)) {
            restoreSource(eviction_desc);
            return false;
        }
        eviction_desc.target_blocks = {target};
    }

    return true;
}

// Reserve the source: exclude it from all heaps and mark the in-flight state so
// no other selector can pick it. The caller must first verify that the resource
// is IDLE; assigning DEMOTING is not safe for an already reserved resource.
void BlockTreeEvictor::reserveSource(const TransferDescriptor& eviction_desc) {
    auto group_set_id = eviction_desc.group_set_id;
    if (eviction_desc.node == nullptr || group_set_id >= eviction_desc.node->group_set_resources.size()) {
        return;
    }
    eviction_desc.node->group_set_resources[group_set_id].transfer_state = GroupSetTransferState::DEMOTING;
    if (auto* heap = heapFor(eviction_desc.group_set_id, eviction_desc.source_tier)) {
        heap->erase(eviction_desc.node);
    }
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
bool BlockTreeEvictor::restoreSource(const TransferDescriptor& eviction_desc) {
    auto group_set_id = eviction_desc.group_set_id;
    if (eviction_desc.node == nullptr || group_set_id >= tree_->groupSets().size()
        || group_set_id >= eviction_desc.node->group_set_resources.size()) {
        return false;
    }
    GroupSetResource& resource = eviction_desc.node->group_set_resources[group_set_id];
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING(
            "state mismatch, group_set=%zu node_key=%ld", eviction_desc.group_set_id, eviction_desc.node->cache_key);
        return false;
    }
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE)
                                || resource.hasCompleteDeviceValue(),
                            "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                            eviction_desc.group_set_id,
                            eviction_desc.node->cache_key);
    refreshCandidate(*tree_->groupSets()[group_set_id], eviction_desc.node, eviction_desc.source_tier);
    return true;
}

void BlockTreeEvictor::releaseTargetBlocks(const TransferDescriptor& eviction_desc) {
    if (eviction_desc.target_blocks.empty()) {
        return;
    }
    auto group_set_id = eviction_desc.group_set_id;
    if (group_set_id >= tree_->groupSets().size()) {
        return;
    }
    auto& group_set = tree_->groupSets()[group_set_id];
    for (auto block : eviction_desc.target_blocks) {
        group_set->releaseSingleBlock(eviction_desc.target_tier, block, BlockRefType::EVICTION);
    }
}

void BlockTreeEvictor::finalizeEviction(TreeNode* node) {
    if (tree_->isRemovable(node)) {
        RTP_LLM_LOG_DEBUG("deleting empty node key=%ld", node->cache_key);
        for (const auto& group_set : tree_->groupSets()) {
            for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
                heapFor(group_set->groupSetId(), tier)->erase(node);
            }
        }
        TreeNode* surviving_ancestor = tree_->removeNodeAndEmptyAncestors(node);
        if (surviving_ancestor != tree_->root()) {
            onTopologyChanged(surviving_ancestor);
        }
    } else if (node->parent && node->parent != tree_->root()) {
        onTopologyChanged(node->parent);
    }
}

std::vector<size_t> BlockTreeEvictor::selectCascadeGroupSets(const TreeNode* node,
                                                             size_t          source_group_set_id,
                                                             Tier            tier,
                                                             bool            enable_reverse_eviction) const {
    std::vector<size_t> result;

    const GroupSetPtr* source_group_set = nullptr;
    for (const auto& group_set : tree_->groupSets()) {
        if (group_set->groupSetId() == source_group_set_id) {
            source_group_set = &group_set;
            break;
        }
    }

    if (source_group_set == nullptr) {
        return result;
    }

    if (enable_reverse_eviction && tree_->isLeafAtTier(node, source_group_set_id, tier)) {
        for (const auto& group_set : tree_->groupSets()) {
            if (group_set->groupSetId() != source_group_set_id) {
                result.push_back(group_set->groupSetId());
            }
        }
        return result;
    }

    // Forward cascading is intrinsic to group-set priority: when a high-priority
    // resource is evicted, lower-priority resources on the same node follow.
    CacheGroupType source_type = (*source_group_set)->groupType();
    for (const auto& group_set : tree_->groupSets()) {
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
