#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

namespace rtp_llm {
namespace {
struct WatermarkNeed {
    bool high;
    size_t to_low;
};

WatermarkNeed watermarkNeed(size_t capacity, size_t used, size_t pending, TierWatermark mark) {
    const size_t effective = used - std::min(pending, used);
    const auto low = static_cast<size_t>(std::floor(capacity * mark.low_ratio));
    return {mark.reachedHigh(capacity, effective), effective > low ? effective - low : 0};
}

size_t evictionDemand(WatermarkNeed need, bool has_pending, bool& triggered) {
    if (!need.to_low && !has_pending)
        triggered = false;
    else if (need.high)
        triggered = true;
    return triggered ? need.to_low : 0;
}
}  // namespace

BlockTreeEvictor::BlockTreeEvictor(BlockTree*                     tree,
                                   EvictionPolicy                 device_policy,
                                   EvictionPolicy                 host_policy,
                                   EvictionPolicy                 disk_policy,
                                   const BlockTransferDispatcher* transfer_dispatcher,
                                   BlockTreeTaskPool*             task_pool,
                                   CacheObserver& metrics_reporter,
                                   std::mutex&                    mutex,
                                   int                            host_timeout_ms,
                                   int                            disk_timeout_ms,
                                   size_t                         max_descriptors_per_batch,
                                   IsTierEnabledFn                is_tier_enabled,
                                   SettledFn                      settled,
                                   size_t                         disk_cache_evict_max_pending_bytes,
                                   std::shared_ptr<StorageBackend> host_writeback_backend):
    tree_(tree),
    host_writeback_backend_(std::move(host_writeback_backend)),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    is_tier_enabled_(std::move(is_tier_enabled)),
    settled_(std::move(settled)),
    task_runner_(std::make_unique<EvictionTaskRunner>(tree->groupSets(), transfer_dispatcher)),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    disk_cache_evict_max_pending_bytes_(disk_cache_evict_max_pending_bytes) {
    // GroupSetFactory has already validated that group_set_id equals the vector
    // position. Own one heap per (group resource, tier).
    heaps_.resize(tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        auto& tier_heaps  = heaps_[group_set_id];
        tier_heaps.device = std::make_unique<EvictionHeap>(device_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(host_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(disk_policy);
        tier_heaps.next_device_reclaim_log.resize(tree_->groupSets()[group_set_id]->pools().device.size());
    }
}

BlockTreeEvictor::~BlockTreeEvictor() {
    std::lock_guard<std::mutex> lock(pending_release_mutex_);
    for (const auto& [pool, count] : pending_release_counts_) {
        BTC_CHECK_WITH_INFO(count == 0,
                                "pending eviction releases remain: pool=%s address=%p count=%zu",
                                pool ? pool->poolName().c_str() : "<null>",
                                static_cast<const void*>(pool),
                                count);
    }
}

EvictionHeap* BlockTreeEvictor::heapFor(size_t group_set_id, Tier tier) const {
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

bool BlockTreeEvictor::isEvictable(TreeNode* node, size_t group_set_id, Tier source_tier) const {
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    return !node->is_resident && resource.transfer_state == GroupSetTransferState::IDLE
           && (resource.getTopTier() == source_tier || (source_tier == Tier::HOST && resource.retained_host_replica)
               || (source_tier == Tier::DISK && resource.retained_disk_replica))
           && (tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL
               || tree_->isLeafAtTier(node, group_set_id, source_tier));
}

void BlockTreeEvictor::suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier) {
    if (source_tier != Tier::NONE) {
        heapFor(group_set_id, source_tier)->erase(node);
    }
}

std::optional<std::pair<size_t, EvictionEntry>>
BlockTreeEvictor::bestHostVictimLocked(const HostBlockPool& pool, const HostVictimPredicate& eligible) const {
    std::optional<std::pair<size_t, EvictionEntry>> candidate;
    // A shared physical pool can reclaim a peer group's Host page. Per-group
    // quotas have distinct pools and keep their own eviction candidates.
    for (const auto& peer : tree_->groupSets()) {
        if (peer->pools().host.get() != &pool) {
            continue;
        }
        const auto peer_id = peer->groupSetId();
        const auto victim = heapFor(peer_id, Tier::HOST)->best([&](TreeNode* node) {
            return (!eligible || eligible(peer_id, node)) && isEvictable(node, peer_id, Tier::HOST)
                && pool.canReclaimCacheBlock(node->group_set_resources[peer_id].host_block);
        });
        if (victim && (!candidate || EvictionHeap::EntryLess{}(*victim, candidate->second))) {
            candidate.emplace(peer_id, *victim);
        }
    }
    return candidate;
}

bool BlockTreeEvictor::dropColdHostPageLocked(size_t group_id, const char* trigger) {
    const auto& pool = tree_->groupSets().at(group_id)->pools().host;
    if (!pool) return false;
    const auto candidate = bestHostVictimLocked(*pool);
    if (!candidate) return false;
    const auto& [victim_group, entry] = *candidate;
    auto* node = entry.node;
    TransferDescriptor drop(node, victim_group, node->depth - 1, Tier::HOST, Tier::NONE,
                            node->group_set_resources[victim_group].getBlocks(Tier::HOST), true);
    runDropTask(std::move(drop), false, trigger, nullptr, DropScope::PRIMARY_ONLY);
    return true;
}

size_t BlockTreeEvictor::pendingHostReleases(const HostBlockPool& pool) const {
    const auto retired = pool.retiredBlocksNum();
    std::lock_guard lock(pending_release_mutex_);
    const auto pending = pending_release_counts_.find(&pool);
    return retired + (pending == pending_release_counts_.end() ? 0 : pending->second);
}

size_t BlockTreeEvictor::scheduleHostReleaseLocked(const HostBlockPool& pool, size_t deficit,
                                                   const HostVictimPredicate& eligible) {
    size_t scheduled = pendingHostReleases(pool);
    if (deficit <= scheduled || watermarkTargetTier(Tier::HOST) != Tier::REMOTE) {
        return scheduled;
    }
    const auto batch_limit = std::min(deficit - scheduled, max_descriptors_per_batch_);
    std::vector<TransferDescriptor> descriptors;
    std::vector<EvictionTimingMetrics> timings;
    const auto submit = [&] {
        if (descriptors.empty()) {
            return true;
        }
        const auto count = descriptors.size();
        const bool accepted = submitEvictionTask(std::move(descriptors), std::move(timings));
        descriptors.clear();
        timings.clear();
        scheduled += accepted ? count : 0;
        return accepted;
    };
    while (scheduled + descriptors.size() < deficit) {
        const auto victim = bestHostVictimLocked(pool, eligible);
        if (!victim) {
            break;
        }
        const auto& [group, entry] = *victim;
        if (!descriptors.empty() && (descriptors.size() == batch_limit || descriptors.front().group_set_id != group)
            && !submit()) {
            return scheduled;
        }
        if (descriptors.empty()) {
            descriptors.reserve(batch_limit);
            timings.reserve(batch_limit);
        }
        auto* node = entry.node;
        const auto& resource = node->group_set_resources[group];
        descriptors.emplace_back(node, group, node->depth - 1, Tier::HOST, Tier::REMOTE,
                                 resource.getBlocks(Tier::HOST), true);
        timings.push_back(prepareEvictionSource(descriptors.back()));
    }
    submit();
    return scheduled;
}

void BlockTreeEvictor::admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier) {
    if (node->parent == nullptr || node->is_resident || target_tier == Tier::NONE
        || node->group_set_resources[group_set_id].transfer_state != GroupSetTransferState::IDLE) {
        return;
    }
    const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
    EvictionHeap*      heap      = heapFor(group_set_id, target_tier);
    if (group_set->groupType() == CacheGroupType::FULL && !tree_->isLeafAtTier(node, group_set_id, target_tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, node->group_set_resources[group_set_id].candidate_meta);
}

void BlockTreeEvictor::updateFullCandidate(TreeNode* parent) {
    if (parent == nullptr || parent->parent == nullptr) {
        return;
    }
    for (auto& group_set : tree_->groupSets()) {
        updateFullCandidate(parent, group_set->groupSetId());
    }
}

void BlockTreeEvictor::updateFullCandidate(TreeNode* node, size_t group_set_id) {
    if (node == nullptr || node->parent == nullptr
        || tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL) {
        return;
    }
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier              tier     = resource.getTopTier();
    if (resource.transfer_state != GroupSetTransferState::IDLE) {
        suspendCandidate(node, group_set_id, tier);
        return;
    }
    admitCandidate(node, group_set_id, tier);
    if (resource.retained_host_replica) {
        admitCandidate(node, group_set_id, Tier::HOST);
    }
    if (resource.retained_disk_replica) {
        admitCandidate(node, group_set_id, Tier::DISK);
    }
}

void BlockTreeEvictor::onLoaded(TreeNode* node, size_t group_set_id) {
    GroupSetResource& resource                    = node->group_set_resources[group_set_id];
    const int64_t     tier_enter_time_us          = cacheNowUs();
    resource.candidate_meta.admission_seq         = ++admission_seq_;
    resource.candidate_meta.tier_enter_time_us    = tier_enter_time_us;
    resource.candidate_meta.last_access_time_us   = tier_enter_time_us;
    resource.candidate_meta.tier_last_hit_time_us = 0;
    if (resource.retained_host_replica) {
        admitCandidate(node, group_set_id, Tier::HOST);
    }
    if (resource.retained_disk_replica) {
        admitCandidate(node, group_set_id, Tier::DISK);
    }
    admitCandidate(node, group_set_id, Tier::DEVICE);
    updateFullCandidate(node->parent, group_set_id);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInserted(const BlockTreeInsertResult& result) {
    updatePendingReleaseBlocks(result.revoked_evictions, /*reserve=*/false, /*revoked=*/true);
    for (TreeNode* node : result.newly_resident_nodes) {
        eraseNodeFromAllHeaps(node);
    }
    for (const auto& candidate : result.replaced_candidates) {
        suspendCandidate(candidate.node, candidate.group_set_id, candidate.source_tier);
    }
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const int64_t     insert_time_us              = cacheNowUs();
            GroupSetResource& resource                    = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq       = nextAccessSeq();
            resource.candidate_meta.admission_seq         = ++admission_seq_;
            resource.candidate_meta.hit_count             = 0;
            resource.candidate_meta.tier_last_hit_time_us = 0;
            resource.candidate_meta.last_access_time_us   = insert_time_us;
            resource.candidate_meta.tier_enter_time_us    = insert_time_us;
            admitCandidate(adopted.first, group_set_id, resource.getTopTier());
            updateFullCandidate(adopted.first->parent, group_set_id);
        }
    }

    for (TreeNode* node : result.inserted_nodes) {
        const uint64_t access         = nextAccessSeq();
        const uint64_t admit          = ++admission_seq_;
        const int64_t  insert_time_us = cacheNowUs();
        for (auto& group_set : tree_->groupSets()) {
            const size_t      group_set_id                = group_set->groupSetId();
            GroupSetResource& resource                    = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq       = access;
            resource.candidate_meta.admission_seq         = admit;
            resource.candidate_meta.hit_count             = 0;
            resource.candidate_meta.tier_last_hit_time_us = 0;
            resource.candidate_meta.last_access_time_us   = insert_time_us;
            resource.candidate_meta.tier_enter_time_us    = insert_time_us;
            admitCandidate(node, group_set_id, resource.getTopTier());
        }
    }

    updateFullCandidate(result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent);
}

uint64_t BlockTreeEvictor::nextAccessSeq() {
    return ++access_seq_;
}

void BlockTreeEvictor::onMatched(TreeNode* node, size_t group_set_id) {
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier              tier     = resource.getTopTier();
    if (tier != Tier::NONE) {
        heapFor(group_set_id, tier)->updateIfPresent(node, resource.candidate_meta);
    }
}

CandidateStats BlockTreeEvictor::candidateStats() const {
    CandidateStats stats;
    for (const auto& tier_heaps : heaps_) {
        stats.device_candidates += tier_heaps.device->size();
        stats.host_candidates += tier_heaps.host->size();
        stats.disk_candidates += tier_heaps.disk->size();
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

EvictionVictimMetrics BlockTreeEvictor::getAndResetEvictionVictimMetrics() const {
    return eviction_victim_metrics_.getAndReset();
}

size_t BlockTreeEvictor::reclaimDeviceBlocksLocked(size_t group_set_id, size_t member_group_id, size_t num_blocks) {
    const GroupSetPtr&   group_set          = tree_->groupSets()[group_set_id];
    const auto&          device_pool        = group_set->pools().device[member_group_id];
    const size_t         initial_free       = device_pool->freeBlocksNum();
    size_t               reclaimed          = 0;
    bool                 eviction_triggered = false;
    auto                 scope              = DropScope::CASCADE;
    EvictionBatchContext eviction_batch_context;
    while (reclaimed < num_blocks) {
        if (!dropLocked(group_set_id,
                        Tier::DEVICE,
                        /*notify_settled=*/true,
                        EvictionVictimTrigger::FORCE_DROP,
                        &eviction_batch_context,
                        scope)) {
            const auto type = group_set->groupType();
            if (scope == DropScope::PRIMARY_ONLY || (type != CacheGroupType::SWA && type != CacheGroupType::LINEAR)) {
                break;
            }
            // Retry with a narrower plan and reconsider candidates rejected by the cascade pass.
            scope                  = DropScope::PRIMARY_ONLY;
            eviction_batch_context = {};
            continue;
        }
        eviction_triggered        = true;
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    if (eviction_triggered) {
        metrics_reporter_->reportEvictionTriggered(Tier::DEVICE, group_set->groupType(), /*force_drop=*/true);
    }
    metrics_reporter_->reportEvictionBlocks(Tier::DEVICE,
                                            group_set->groupType(),
                                            /*force_drop=*/true,
                                            num_blocks,
                                            std::min(reclaimed, num_blocks));
    if (reclaimed == 0 && num_blocks > 0) {
        logZeroDeviceReclaimLocked(
            group_set_id, member_group_id, num_blocks, eviction_batch_context.rejected_candidates.size());
    }
    return reclaimed;
}

void BlockTreeEvictor::logZeroDeviceReclaimLocked(size_t group_set_id,
                                                  size_t member_group_id,
                                                  size_t required_blocks,
                                                  size_t rejected_candidates) {
    auto&      tier_heaps = heaps_[group_set_id];
    auto&      next_log   = tier_heaps.next_device_reclaim_log[member_group_id];
    const auto now        = std::chrono::steady_clock::now();
    if (now < next_log) {
        return;
    }
    next_log = now + std::chrono::seconds(5);

    const auto& group_set  = tree_->groupSets()[group_set_id];
    const auto& pool       = group_set->pools().device[member_group_id];
    const auto  stats      = pool->referenceStats();
    const auto& heap       = *tier_heaps.device;
    const auto  candidates = heap.size();
    // Frozen state is read from every member pool after the diagnostic rate limit.
    const bool has_unfrozen_candidate =
        heap.best([&group_set, group_set_id](TreeNode* node) {
                const auto& blocks = node->group_set_resources[group_set_id].device_blocks;
                for (size_t i = 0; i < blocks.size(); ++i) {
                    if (!isNullBlockIdx(blocks[i]) && group_set->pools().device[i]->isFrozen(blocks[i])) {
                        return false;
                    }
                }
                return true;
            })
            .has_value();
    // A candidate covers at most one block in the requested member pool. The
    // difference is only a lower bound, since candidates may share blocks or hold other references.
    const auto cache_only_outside_heap_min =
        stats.tree.cache_only_blocks - std::min(stats.tree.cache_only_blocks, candidates);
    size_t pending_eviction_releases = 0;
    {
        std::lock_guard<std::mutex> lock(pending_release_mutex_);
        const auto                  pending = pending_release_counts_.find(pool.get());
        if (pending != pending_release_counts_.end()) {
            pending_eviction_releases = pending->second;
        }
    }

    // Reference counts can overlap. Candidate counts are tree nodes, not physical blocks.
    BTC_LOG_WARNING(
        "BTC on-demand DEVICE reclaim returned zero: pool=%s group_set=%zu group_type=%s required_blocks=%zu "
        "total_blocks=%zu free_blocks=%zu active_blocks=%zu cache_only_blocks=%zu cache_only_outside_heap_min=%zu "
        "device_candidate_nodes=%zu has_unfrozen_candidate=%d rejected_candidate_nodes=%zu pending_eviction_releases=%zu "
        "cache_ref_blocks=%zu load_ref_blocks=%zu eviction_ref_blocks=%zu "
        "frozen_ref_blocks=%zu request_ref_blocks=%zu remote_connector_ref_blocks=%zu pd_connector_ref_blocks=%zu",
        pool->poolName().c_str(),
        group_set_id,
        metricCacheGroupTypeName(group_set->groupType()),
        required_blocks,
        stats.tree.total_blocks,
        stats.tree.free_blocks,
        stats.tree.active_blocks,
        stats.tree.cache_only_blocks,
        cache_only_outside_heap_min,
        candidates,
        has_unfrozen_candidate,
        rejected_candidates,
        pending_eviction_releases,
        stats.tree.blocksWithRef(BlockTreeRefType::CACHE),
        stats.tree.blocksWithRef(BlockTreeRefType::LOAD),
        stats.tree.blocksWithRef(BlockTreeRefType::EVICT),
        stats.tree.blocksWithRef(BlockTreeRefType::FROZEN),
        stats.request_ref_blocks,
        stats.remote_connector_ref_blocks,
        stats.pd_connector_ref_blocks);
}

bool BlockTreeEvictor::dropLocked(size_t                group_set_id,
                                  Tier                  source_tier,
                                  bool                  notify_settled,
                                  EvictionVictimTrigger trigger,
                                  EvictionBatchContext* eviction_batch_context,
                                  DropScope             scope) {
    std::optional<TransferDescriptor> eviction_desc =
        chooseVictim(group_set_id, source_tier, /*force_drop=*/true, eviction_batch_context, trigger, scope);
    if (!eviction_desc.has_value()) {
        return false;
    }
    const auto& resource = eviction_desc->node->group_set_resources[group_set_id];
    if ((source_tier != Tier::DEVICE && eviction_desc->node->opaque)
        || resource.retained_host_replica || resource.retained_disk_replica) {
        scope = DropScope::PRIMARY_ONLY;
    }
    runDropTask(std::move(*eviction_desc),
                notify_settled,
                trigger == EvictionVictimTrigger::WATERMARK ? "watermark" : "force_drop",
                eviction_batch_context,
                scope);
    return true;
}

bool BlockTreeEvictor::batchDropLocked(size_t  group_set_id,
                                       Tier    source_tier,
                                       size_t  max_victim_count,
                                       size_t& scheduled_count) {
    std::vector<IBlockPool*> source_pools;
    const GroupSetPtr&       group_set = tree_->groupSets()[group_set_id];
    if (source_tier == Tier::DEVICE) {
        for (const auto& pool : group_set->pools().device) {
            source_pools.push_back(pool.get());
        }
    } else if (source_tier == Tier::HOST && group_set->pools().host != nullptr) {
        source_pools.push_back(group_set->pools().host.get());
    } else if (source_tier == Tier::DISK && group_set->pools().disk != nullptr) {
        source_pools.push_back(group_set->pools().disk.get());
    }

    scheduled_count                      = 0;
    bool                 processed       = false;
    size_t               remaining_count = max_victim_count;
    EvictionBatchContext eviction_batch_context;
    while (remaining_count > 0) {
        std::vector<size_t> used_before;
        used_before.reserve(source_pools.size());
        for (IBlockPool* pool : source_pools) {
            used_before.push_back(pool->usedBlocksNum());
        }
        if (!dropLocked(group_set_id, source_tier, false, EvictionVictimTrigger::WATERMARK, &eviction_batch_context)) {
            break;
        }
        processed                  = true;
        size_t physically_released = std::numeric_limits<size_t>::max();
        for (size_t pool_index = 0; pool_index < source_pools.size(); ++pool_index) {
            const size_t used_after = source_pools[pool_index]->usedBlocksNum();
            physically_released =
                std::min(physically_released, used_before[pool_index] - std::min(used_before[pool_index], used_after));
        }
        scheduled_count += std::min(physically_released, remaining_count);
        const size_t progress = std::min(std::max<size_t>(1, physically_released), remaining_count);
        remaining_count -= progress;
    }
    if (processed) {
        settled_(false);
    }
    return processed;
}

size_t BlockTreeEvictor::releaseDevicePagesLocked(const IBlockPool& pool, const std::vector<BlockIdList>& slots,
                                                  size_t requested_slots, const BlockIdList& eligible,
                                                  size_t pressure, bool copy_only,
                                                  bool demote_to_host, bool cascade) {
    using SiteKey = std::pair<uint64_t, size_t>;
    struct Victim {
        size_t slot;
        EvictionEntry rank;
        std::map<SiteKey, TreeNode*> roots;
    };
    if (!requested_slots) {
        return 0;
    }
    const auto& groups = tree_->groupSets();
    const std::unordered_set<BlockIdxType> allowed(eligible.begin(), eligible.end());
    std::unordered_map<BlockIdxType, size_t> slot_for_block;
    std::vector<Victim> victims;
    victims.reserve(requested_slots);
    for (size_t slot = 0; slot < slots.size(); ++slot) {
        if (slot < requested_slots) {
            victims.push_back({slot, {}, {}});
        }
        for (auto block : slots[slot]) {
            slot_for_block.emplace(block, slot);
        }
    }
    const EvictionHeap::EntryLess less;
    for (const auto& group : groups) {
        const auto group_id = group->groupSetId();
        const auto& pools = group->pools().device;
        const auto* heap = heapFor(group_id, Tier::DEVICE);
        for (auto* node : heap->nodes()) {
            const auto& resource = node->group_set_resources[group_id];
            if (!resource.hasCompleteDeviceValue()) {
                continue;
            }
            for (size_t member = 0; member < pools.size(); ++member) {
                const auto selected = slot_for_block.find(resource.device_blocks[member]);
                if (pools[member].get() != &pool || selected == slot_for_block.end() || selected->second >= requested_slots
                    || !isEvictable(node, group_id, Tier::DEVICE)) {
                    continue;
                }
                auto& victim = victims[selected->second];
                const auto& rank = heap->entry(node);
                if (victim.roots.empty() || less(victim.rank, rank)) {
                    victim.rank = rank;
                }
                victim.roots.emplace(SiteKey{node->identity, group_id}, node);
            }
        }
    }
    std::sort(victims.begin(), victims.end(), [&](const Victim& a, const Victim& b) {
        return less(a.rank, b.rank);
    });

    std::set<SiteKey> selected;
    std::vector<std::vector<TreeNode*>> copies(groups.size());
    std::vector<bool> retired(slots.size(), false);
    std::unordered_map<BlockIdxType, size_t> planned_moves;
    size_t progress = 0;
    // Related reclaim may remove an eligible sibling at the selected node.
    // It does not grant permission to the native ancestor-pruning path.
    const auto can_drop = [&](TreeNode* node, size_t group) {
        const auto& resource = node->group_set_resources[group];
        if (resource.transfer_detached || !isEvictable(node, group, Tier::DEVICE)
            || !resource.hasCompleteDeviceValue() || (demote_to_host && !groups[group]->pools().host)) {
            return false;
        }
        const auto& pools = groups[group]->pools().device;
        for (size_t member = 0; member < pools.size(); ++member) {
            const auto block = resource.device_blocks[member];
            if (!pools[member]->unpinnedCacheReferences(block)
                || (!allowed.empty() && (pools[member].get() != &pool || !allowed.count(block)))) {
                return false;
            }
        }
        return true;
    };
    for (const auto& victim : victims) {
        if (progress >= pressure) {
            break;
        }
        if (retired[victim.slot] || victim.roots.empty()) {
            continue;
        }
        std::map<SiteKey, TreeNode*> drop;
        std::unordered_map<BlockIdxType, size_t> root_refs;
        bool roots_available = true;
        for (const auto& [site, node] : victim.roots) {
            if (selected.count(site)) {
                continue;
            }
            const auto group = site.second;
            if (!can_drop(node, group)) {
                roots_available = false;
                break;
            }
            drop.emplace(site, node);
            const auto& resource = node->group_set_resources[group];
            const auto& pools = groups[group]->pools().device;
            for (size_t member = 0; member < pools.size(); ++member) {
                if (pools[member].get() == &pool) {
                    ++root_refs[resource.device_blocks[member]];
                }
            }
        }
        if (!roots_available || drop.empty()
            || !std::all_of(slots[victim.slot].begin(), slots[victim.slot].end(), [&](auto block) {
                   const auto refs = pool.unpinnedCacheReferences(block);
                   const auto planned = planned_moves.find(block);
                   const size_t already_moving = planned == planned_moves.end() ? 0 : planned->second;
                   return refs && root_refs[block] + already_moving == refs;
               })) {
            continue;
        }
        if (cascade) {
            for (const auto& [site, node] : victim.roots) {
                if (selected.count(site)) {
                    continue;
                }
                const bool leaf = tree_->isLeafAtTier(node, site.second, Tier::DEVICE);
                for (const auto& group : groups) {
                    const SiteKey sibling{site.first, group->groupSetId()};
                    if (sibling != site && !selected.count(sibling)
                        && (leaf || group->groupType() == CacheGroupType::LINEAR)
                        && can_drop(node, sibling.second)) {
                        drop.emplace(sibling, node);
                    }
                }
            }
        }
        bool accepted = false;
        std::set<size_t> touched_slots;
        for (const auto& [site, node] : drop) {
            const auto group = site.second;
            const auto& resource = node->group_set_resources[group];
            const bool retained = resource.retained_host_replica || resource.retained_disk_replica;
            if (copy_only && retained) {
                continue;
            }
            selected.insert(site);
            accepted = true;
            const bool moving = demote_to_host && !retained;
            for (size_t member = 0; member < groups[group]->pools().device.size(); ++member) {
                const auto block = resource.device_blocks[member];
                const auto slot = slot_for_block.find(block);
                if (groups[group]->pools().device[member].get() == &pool && slot != slot_for_block.end()) {
                    touched_slots.insert(slot->second);
                    if (moving && !copy_only)
                        ++planned_moves[block];
                }
            }
            if (moving) {
                copies[group].push_back(node);
            } else {
                TransferDescriptor desc(node, group, 0, Tier::DEVICE, Tier::NONE, resource.device_blocks, true);
                runDropTask(std::move(desc), false, "force_drop", nullptr, DropScope::PRIMARY_ONLY);
            }
        }
        size_t retired_here = 0;
        for (auto slot : touched_slots) {
            if (!retired[slot] && std::all_of(slots[slot].begin(), slots[slot].end(), [&](auto block) {
                    if (!pool.pageIdentity(block))
                        return true;
                    const auto planned = planned_moves.find(block);
                    return planned != planned_moves.end() && planned->second == pool.unpinnedCacheReferences(block);
                })) {
                retired[slot] = true;
                ++retired_here;
            }
        }
        progress += std::max<size_t>(retired_here, accepted);
    }
    // One Host slot holds a whole group's payload, even for multi-member Device groups.
    std::map<const HostBlockPool*, std::pair<size_t, size_t>> shortages;
    for (size_t group = 0; group < copies.size(); ++group) {
        if (copies[group].empty())
            continue;
        const auto scheduled = submitHostCopiesLocked(group, copies[group], copy_only);
        if (scheduled < copies[group].size()) {
            auto& [group_id, needed] = shortages[groups[group]->pools().host.get()];
            group_id = group;
            needed += copies[group].size() - scheduled;
        }
    }
    for (const auto& [_, shortage] : shortages)
        relieveHostPressureLocked(shortage.first, shortage.second);
    return progress;
}

bool BlockTreeEvictor::invalidateDevicePagesLocked(const IBlockPool& pool, const BlockIdList& blocks) {
    if (blocks.empty()) {
        return true;
    }
    const std::unordered_set<BlockIdxType> invalid(blocks.begin(), blocks.end());
    std::vector<std::pair<TreeNode*, size_t>> affected;
    bool busy = false;
    for (const auto& node : tree_->node_pool_) {
        for (const auto& group : tree_->groupSets()) {
            const auto group_id = group->groupSetId();
            const auto& resource = node->group_set_resources[group_id];
            const auto& pools = group->pools().device;
            bool matches = false;
            for (size_t member = 0; member < resource.device_blocks.size(); ++member) {
                if (pools[member].get() == &pool && invalid.count(resource.device_blocks[member])) {
                    matches = true;
                    busy |= resource.transfer_state != GroupSetTransferState::IDLE
                            || pool.isFrozen(resource.device_blocks[member]);
                }
            }
            if (matches) {
                affected.emplace_back(node.get(), group_id);
            }
        }
    }
    // The cache mutex covers both passes. Dropping only the selected tier
    // leaves every later (node, group) alive through its own Device resource.
    for (const auto& [node, group_id] : affected) {
        invalidateSource(node, group_id, Tier::DEVICE, DropScope::PRIMARY_ONLY);
    }
    return !busy;
}

size_t BlockTreeEvictor::reclaimLocalPagesLocked(Tier tier, size_t group_id, size_t pressure) {
    if ((tier != Tier::HOST && tier != Tier::DISK) || group_id >= heaps_.size()) {
        return 0;
    }
    return batchEvictLocked(group_id, tier, pressure).scheduled_count;
}

void BlockTreeEvictor::setLocalEvictionPolicyLocked(Tier tier, EvictionPolicy policy) {
    if (tier != Tier::HOST && tier != Tier::DISK) {
        throw std::invalid_argument("local eviction policy requires Host or Disk");
    }
    for (size_t group_id = 0; group_id < heaps_.size(); ++group_id) {
        auto& old = tier == Tier::HOST ? heaps_[group_id].host : heaps_[group_id].disk;
        auto next = std::make_unique<EvictionHeap>(policy);
        for (auto* node : old->nodes()) {
            next->upsert(node, node->group_set_resources[group_id].candidate_meta);
        }
        old = std::move(next);
    }
}

void BlockTreeEvictor::relieveHostPressureLocked(size_t group_id, size_t needed) {
    const auto& host = tree_->groupSets()[group_id]->pools().host;
    if (!needed || !host || host->freeBlocksNum() != 0)
        return;
    const auto target = watermarkTargetTier(Tier::HOST);
    if (target == Tier::REMOTE) {
        scheduleHostReleaseLocked(*host, needed);
        return;
    }
    if (target == Tier::NONE) {
        const size_t pending = pendingHostReleases(*host);
        for (size_t i = pending; i < needed; ++i) {
            if (!dropColdHostPageLocked(group_id, "host_capacity"))
                break;
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pending_release_mutex_);
        const auto found = pending_release_counts_.find(host.get());
        if (found != pending_release_counts_.end() && found->second != 0) {
            return;
        }
    }
    batchEvictLocked(group_id, Tier::HOST, std::min(needed, max_descriptors_per_batch_));
}

size_t BlockTreeEvictor::submitHostCopiesLocked(size_t group_id, const std::vector<TreeNode*>& nodes,
                                               bool retain_source) {
    const auto& group = *tree_->groupSets()[group_id];
    if (!group.pools().host) {
        return 0;
    }
    size_t scheduled = 0;
    std::vector<TransferDescriptor> descriptors;
    std::vector<EvictionTimingMetrics> timings;
    auto submit = [&] {
        const auto count = descriptors.size();
        if (count && submitEvictionTask(std::move(descriptors), std::move(timings), retain_source)) {
            scheduled += count;
        }
        descriptors.clear();
        timings.clear();
    };
    for (TreeNode* node : nodes) {
        auto& resource = node->group_set_resources[group_id];
        if (!resource.hasCompleteDeviceValue() || resource.servingTierCount() != 1
            || resource.transfer_state != GroupSetTransferState::IDLE) {
            continue;
        }
        auto host = group.pools().host->malloc();
        if (!host) {
            break;
        }
        group.pools().host->incTreeRef(*host, BlockTreeRefType::EVICT);
        timings.emplace_back(resource.candidate_meta);
        descriptors.emplace_back(node, group_id, 0, Tier::DEVICE, Tier::HOST, resource.device_blocks, true);
        descriptors.back().target_blocks.push_back(*host);
        suspendCandidate(node, group_id, Tier::DEVICE);
        resource.eviction_access_seq = resource.candidate_meta.last_access_seq;
        resource.transfer_state = GroupSetTransferState::EVICTING;
        tree_->refreshPublishedState(node);
        if (descriptors.size() >= max_descriptors_per_batch_) {
            submit();
        }
    }
    submit();
    return scheduled;
}

bool BlockTreeEvictor::submitEvictionTask(std::vector<TransferDescriptor>    descriptors,
                                          std::vector<EvictionTimingMetrics> timings,
                                          bool                               retain_source) {
    const TransferDescriptor& descriptor = descriptors.front();
    const auto                timeout    = std::chrono::milliseconds(
        descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK ? disk_timeout_ms_ :
                                                                                       host_timeout_ms_);
    auto task_ptr =
        std::make_shared<EvictionTransferTask>(
            TransferTask(std::move(descriptors), timeout), std::move(timings), retain_source);
    for (size_t i = 0; i < task_ptr->descriptors().size(); ++i) {
        const auto& desc = task_ptr->descriptors()[i];
        heaps_[desc.group_set_id].pending_timings.emplace(
            std::make_pair(desc.node, desc.generation),
            GroupSetTierHeaps::PendingEviction{task_ptr->timings[i], retain_source, desc.source_tier});
    }
    if (!retain_source) {
        updatePendingRelease(task_ptr->descriptors(), true);
    }
    auto queue_wait =
        TransferStageTimer::create(CacheStageContext::forTransfer(*metrics_reporter_, task_ptr->transfer_task, "evict"),
                                   "business_queue_wait",
                                   "business_task",
                                   AsyncProfileKind::Wait);
    block_tree_cache_detail::ScopeRollback discard_wait([&] { queue_wait.discard(); });
    auto                                   on_timeout = [this, task_ptr, queue_wait]() {
        queue_wait.finish(false);
        BTC_LOG_WARNING("eviction expired in business queue, source=%s target=%s descriptors=%zu",
                            tierName(task_ptr->descriptors().front().source_tier),
                            tierName(task_ptr->descriptors().front().target_tier),
                            task_ptr->descriptors().size());
        scheduleEvictionSettlement(task_ptr, false);
    };
    const auto deadline  = task_ptr->transfer_task.deadline();
    const bool submitted = task_pool_->submit(
        BlockTreeTaskClass::BACKGROUND,
        [this, task_ptr, queue_wait]() {
            queue_wait.finish(true);
            runEvictionTask(task_ptr);
        },
        deadline,
        std::move(on_timeout));
    if (!submitted) {
        const auto& first  = task_ptr->descriptors().front();
        const auto  queues = task_pool_->queueSizes();
        BTC_INTERVAL_LOG_WARNING(5,
                             "eviction submission rejected: group_set=%zu source=%s target=%s blocks=%zu "
                             "queue_load=%zu queue_background=%zu queue_completion=%zu",
                             first.group_set_id,
                             tierName(first.source_tier),
                             tierName(first.target_tier),
                             task_ptr->descriptors().size(),
                             queues.load,
                             queues.background,
                             queues.completion);
        for (const auto& desc : task_ptr->descriptors()) {
            heaps_[desc.group_set_id].pending_timings.erase({desc.node, desc.generation});
        }
        if (!retain_source) {
            updatePendingRelease(task_ptr->descriptors(), false);
        }
        rollbackTransferLocked(task_ptr->descriptors());
        metrics_reporter_->reportEvictionFinished(*task_ptr, tree_->groupSets(), false, {}, 0);
        return false;
    }
    discard_wait.dismiss();
    return true;
}

BlockTreeEvictor::BatchEvictResult
BlockTreeEvictor::batchEvictLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count) {
    if (max_victim_count == 0) {
        return {};
    }
    const Tier target_tier = watermarkTargetTier(source_tier);
    if (target_tier == Tier::NONE) {
        BatchEvictResult result;
        result.direct_progress = batchDropLocked(group_set_id, source_tier, max_victim_count, result.scheduled_count);
        if (result.scheduled_count != 0 && task_pool_ != nullptr) {
            try {
                // Continue only after physical reclamation, and let LOAD run between batches.
                result.async_submitted =
                    task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [this, group_set_id, source_tier] {
                        CacheLockGuard lock(
                            *mutex_,
                            "watermark",
                            CacheStageContext::forLock(metrics_reporter_, "watermark", source_tier, Tier::NONE));
                        --heaps_[group_set_id].watermark_rounds[static_cast<size_t>(source_tier)].pending_count;
                        settled_(true);
                        finishWatermarkRoundLocked(group_set_id, source_tier);
                    });
            } catch (const std::exception& error) {
                BTC_LOG_ERROR("failed to enqueue drop settlement: %s", error.what());
            } catch (...) {
                BTC_LOG_ERROR("failed to enqueue drop settlement with unknown exception");
            }
            if (result.async_submitted) {
                // The callback must acquire our cache mutex before consuming this count.
                ++heaps_[group_set_id].watermark_rounds[static_cast<size_t>(source_tier)].pending_count;
            }
        }
        return result;
    }

    std::vector<TransferDescriptor>    descriptors;
    std::vector<EvictionTimingMetrics> timings;
    EvictionBatchContext               eviction_batch_context;
    for (size_t victim_count = 0; victim_count < max_victim_count; ++victim_count) {
        auto eviction_desc = chooseVictim(group_set_id,
                                          source_tier,
                                          /*force_drop=*/false,
                                          &eviction_batch_context,
                                          EvictionVictimTrigger::WATERMARK);
        if (!eviction_desc.has_value()) {
            break;
        }
        if (descriptors.empty()) {
            descriptors.reserve(max_victim_count);
            timings.reserve(max_victim_count);
        }
        auto& desc     = *eviction_desc;
        auto& resource = desc.node->group_set_resources[desc.group_set_id];
        const bool has_lower_replica =
            (source_tier == Tier::DEVICE && (resource.retained_host_replica || resource.retained_disk_replica))
            || (source_tier == Tier::HOST && target_tier == Tier::DISK && resource.retained_disk_replica);
        if (has_lower_replica) {
            if (!descriptors.empty()) {
                break;
            }
            desc.target_tier = Tier::NONE;
            runDropTask(std::move(desc), false, "watermark", nullptr, DropScope::PRIMARY_ONLY);
            return {true, false, 1};
        }
        timings.push_back(prepareEvictionSource(desc));
        descriptors.push_back(std::move(desc));
    }
    if (descriptors.empty()) {
        return {/*direct_progress=*/false,
                /*async_submitted=*/false,
                /*scheduled_count=*/0};
    }

    if (target_tier == Tier::REMOTE) {
        const auto count = descriptors.size();
        const auto submitted = submitEvictionTask(std::move(descriptors), std::move(timings));
        return {false, submitted, submitted ? count : 0};
    }

    const auto& group_set     = tree_->groupSets()[group_set_id];
    auto        target_blocks = group_set->allocateBlocks(descriptors.size(), target_tier, BlockTreeRefType::EVICT);
    if (!target_blocks.has_value()) {
        const IBlockPool* pool = target_tier == Tier::HOST ?
                                     group_set->pools().host.get() :
                                     group_set->pools().disk.get();
        BTC_INTERVAL_LOG_WARNING(5,
                             "eviction target allocation failed: group_set=%zu source=%s target=%s blocks=%zu "
                             "target_pool=%s",
                             group_set_id,
                             tierName(source_tier),
                             tierName(target_tier),
                             descriptors.size(),
                             pool ? pool->debugString().c_str() : "<null>");
        rollbackTransferLocked(descriptors);
        metrics_reporter_->reportEvictionFinished(
            EvictionTransferTask(TransferTask(std::move(descriptors), std::chrono::milliseconds(0)),
                                 std::move(timings)),
            tree_->groupSets(),
            false,
            {},
            0);
        return {/*direct_progress=*/false,
                /*async_submitted=*/false,
                /*scheduled_count=*/0};
    }
    for (size_t desc_index = 0; desc_index < descriptors.size(); ++desc_index) {
        descriptors[desc_index].target_blocks = {(*target_blocks)[desc_index]};
    }
    const size_t async_scheduled_count = descriptors.size();
    const bool   submitted             = submitEvictionTask(std::move(descriptors), std::move(timings));
    return {/*direct_progress=*/false,
            /*async_submitted=*/submitted,
            /*scheduled_count=*/submitted ? async_scheduled_count : 0};
}

EvictionTimingMetrics BlockTreeEvictor::prepareEvictionSource(const TransferDescriptor& desc) {
    auto& resource = desc.node->group_set_resources[desc.group_set_id];
    const EvictionTimingMetrics timing(resource.candidate_meta);
    suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
    resource.eviction_access_seq = resource.candidate_meta.last_access_seq;
    resource.transfer_state = GroupSetTransferState::EVICTING;
    tree_->refreshPublishedState(desc.node);
    return timing;
}

void BlockTreeEvictor::runEvictionTask(std::shared_ptr<const EvictionTransferTask> task) noexcept {
    try {
        if (task->descriptors().front().target_tier == Tier::REMOTE) {
            runHostWriteback(task);
            return;
        }
        task_runner_->runTransfer(task, *metrics_reporter_, [this, task](ErrorInfo error) noexcept {
            completeEvictionTask(task, std::move(error));
        });
        return;
    } catch (const std::exception& error) {
        BTC_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        BTC_LOG_ERROR("eviction copy failed with unknown exception");
    }
    completeEvictionTask(std::move(task), ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "eviction task runner exception"));
}

void BlockTreeEvictor::runHostWriteback(const std::shared_ptr<const EvictionTransferTask>& task) {
    StorageRequest request;
    auto keys = std::make_shared<BlockTreeKeys>();
    const auto group_id = task->descriptors().front().group_set_id;
    const auto& group = *tree_->groupSets()[group_id];
    keys->reserve(task->descriptors().size());
    request.handles.reserve(task->descriptors().size());
    for (const auto& desc : task->descriptors()) {
        if (desc.logical_key.empty()) {
            scheduleEvictionSettlement(task, false);
            return;
        }
        keys->emplace_back(std::string_view(desc.logical_key));
        request.handles.push_back({{group.groupTags().front(), desc.source_blocks.front(), Tier::HOST,
                                    (desc.path_index + 1) * tree_->keyTokens()}});
    }
    request.opaque_keys = std::move(keys);
    request.positional_replicas = true;
    request.transfer_task = task->transfer_task;
    auto write = host_writeback_backend_->prepareWrite(std::move(request));
    if (!write) {
        scheduleEvictionSettlement(task, false);
        return;
    }
    host_writeback_backend_->write(std::move(write), [this, task](StorageTransferOutcome outcome) {
        scheduleEvictionSettlement(task, outcome.safe && outcome.error.ok());
    });
}

void BlockTreeEvictor::completeEvictionTask(std::shared_ptr<const EvictionTransferTask> task,
                                            ErrorInfo                                   error) noexcept {
    scheduleEvictionSettlement(std::move(task), error.ok());
}

void BlockTreeEvictor::scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task,
                                                  bool                                        success) noexcept {
    auto stage_context = CacheStageContext::forTransfer(*metrics_reporter_, task->transfer_task, "evict");
    auto timers        = CompletionStageTimers::create(stage_context);
    auto settle        = [this, task = std::move(task), success, stage_context, timers]() noexcept {
        timers.finishQueue(success);
        block_tree_cache_detail::ScopeRollback finish([&] { timers.finish(success); });
        auto execute = TransferStageTimer::scope(stage_context, "completion_execute", "business_completion", success);
        size_t            cancelled_blocks           = 0;
        bool              has_valid_eviction         = false;
        std::vector<bool> skip_eviction_metrics;
        skip_eviction_metrics.reserve(task->descriptors().size());
        {
            CacheLockGuard lock(*mutex_, "watermark", stage_context);
            if (!task->retain_source) {
                updatePendingRelease(task->descriptors(), false);
            }
            for (const TransferDescriptor& desc : task->descriptors()) {
                heaps_[desc.group_set_id].pending_timings.erase({desc.node, desc.generation});
                const auto& resource          = desc.node->group_set_resources[desc.group_set_id];
                const bool  is_valid_eviction = desc.isCurrentOwner() && !resource.transfer_detached;
                const bool  cancelled_by_access =
                    is_valid_eviction && success && !task->retain_source
                    && resource.candidate_meta.last_access_seq != resource.eviction_access_seq;
                cancelled_blocks += cancelled_by_access;
                skip_eviction_metrics.push_back(!is_valid_eviction || cancelled_by_access);
                has_valid_eviction |= is_valid_eviction;
            }

            if (success) {
                {
                    auto update = TransferStageTimer::scope(
                        stage_context, "completion_tree_update", "business_completion", success);
                    completeEvict(task->descriptors(), task->retain_source);
                }
                {
                    auto cleanup =
                        TransferStageTimer::scope(stage_context, "completion_cleanup", "business_completion", success);
                    settleEviction(task->descriptors());
                }
            } else {
                auto rollback =
                    TransferStageTimer::scope(stage_context, "completion_rollback", "business_completion", success);
                rollbackTransferLocked(task->descriptors());
            }
            const auto& descriptor = task->descriptors().front();
            if (!success && descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::REMOTE
                && host_release_failed_) {
                host_release_failed_(*tree_->groupSets()[descriptor.group_set_id]->pools().host);
            }

            {
                auto watermark =
                    TransferStageTimer::scope(stage_context, "completion_watermark", "business_completion", success);
                const bool is_host_disk = descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK;
                auto&      round =
                    heaps_[descriptor.group_set_id].watermark_rounds[static_cast<size_t>(descriptor.source_tier)];
                // Keep the round active while settled_ rechecks watermarks.
                settled_(success && has_valid_eviction);
                if (is_host_disk) {
                    if (!success) {
                        round.watermark.reset();
                    }
                    scheduleWatermarkEvictionsLocked(Tier::HOST);
                }
                if (!task->retain_source && (!is_host_disk || !round.watermark.has_value())) {
                    finishWatermarkRoundLocked(descriptor.group_set_id, descriptor.source_tier);
                }
            }
        }
        if (has_valid_eviction) {
            auto metrics =
                TransferStageTimer::scope(stage_context, "completion_metrics", "business_completion", success);
            metrics_reporter_->reportEvictionFinished(
                *task, tree_->groupSets(), success, skip_eviction_metrics, cancelled_blocks);
        }
    };

    bool submitted = false;
    try {
        submitted = task_pool_->submitCompletion(settle);
    } catch (const std::exception& error) {
        BTC_LOG_ERROR("failed to enqueue eviction settlement: %s", error.what());
    } catch (...) {
        BTC_LOG_ERROR("failed to enqueue eviction settlement with unknown exception");
    }
    if (!submitted) {
        BTC_LOG_WARNING("eviction completion queue is closed; settling inline");
        settle();
    }
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(size_t                group_set_id,
                                                                 Tier                  tier,
                                                                 bool                  force_drop,
                                                                 EvictionBatchContext* eviction_batch_context,
                                                                 EvictionVictimTrigger trigger,
                                                                 DropScope             scope) {
    EvictionHeap* heap = heapFor(group_set_id, tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    // Instrumentation adds no allocation, atomics, extra locks, or reporting in the selection loop.
    auto* counters =
        metrics_reporter_->enabled() ?
            &eviction_victim_metrics_.at(trigger, tier, tree_->groupSets()[group_set_id]->groupType(), force_drop) :
            nullptr;
    if (counters != nullptr) {
        counters->candidates_sum += heap->size();
    }
    const auto begin    = counters ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool       selected = false;
    block_tree_cache_detail::ScopeRollback record([&] {
        if (counters != nullptr) {
            counters->observed = true;
            ++counters->calls;
            counters->selected += selected;
            counters->latency_sum_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        }
    });
    EvictionBatchContext                   local_eviction_batch_context;
    if (eviction_batch_context == nullptr) {
        eviction_batch_context = &local_eviction_batch_context;
    }
    struct SelectionContext {
        EvictionVictimCounters* counters;
        EvictionBatchContext*   eviction_batch_context;

        bool skip(EvictionVictimSkipReason reason) const {
            if (counters != nullptr) {
                ++counters->skipped[static_cast<size_t>(reason)];
            }
            return false;
        }
    } context{counters, eviction_batch_context};
    const auto can_evict_uncached = [this, group_set_id, tier, force_drop, scope, &context](TreeNode* node) {
        const GroupSetPtr&      group_set = tree_->groupSets()[group_set_id];
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (tier == Tier::HOST || tier == Tier::DISK) {
            const auto& pool = tier == Tier::HOST ? group_set->pools().host : group_set->pools().disk;
            const auto block = tier == Tier::HOST ? resource.host_block : resource.disk_block;
            if (!isNullBlockIdx(block)) {
                if (node->opaque && !pool->unpinnedCacheReferences(block)) {
                    return context.skip(pool->isFrozen(block) ? EvictionVictimSkipReason::FROZEN
                                                             : EvictionVictimSkipReason::NOT_RECLAIMABLE);
                }
                if (!node->opaque && !force_drop && pool->isFrozen(block)) {
                    return context.skip(EvictionVictimSkipReason::FROZEN);
                }
            }
        }
        if (force_drop && tier == Tier::DEVICE && isEvictable(node, group_set_id, tier)) {
            const TransferDescriptor desc(node,
                                          group_set_id,
                                          /*path_index=*/0,
                                          tier,
                                          Tier::NONE,
                                          resource.getBlocks(tier),
                                          /*initial_owner=*/true);
            if (!canReclaimDeviceDrop(desc, context.eviction_batch_context, scope)) {
                return context.skip(EvictionVictimSkipReason::NOT_RECLAIMABLE);
            }
        }
        if (!force_drop && tier == Tier::DEVICE) {
            for (size_t i = 0; i < resource.device_blocks.size(); ++i) {
                const auto block = resource.device_blocks[i];
                if (isNullBlockIdx(block)) {
                    continue;
                }
                const auto& pool = group_set->pools().device[i];
                if (pool->hasRequestRef(block)) {
                    return context.skip(EvictionVictimSkipReason::REQUEST_REF);
                }
                if (pool->isFrozen(block)) {
                    return context.skip(EvictionVictimSkipReason::FROZEN);
                }
                if (node->opaque && !pool->unpinnedCacheReferences(block)) {
                    return context.skip(EvictionVictimSkipReason::NOT_RECLAIMABLE);
                }
            }
        }
        if (force_drop || tier != Tier::DEVICE
            || tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL) {
            return true;
        }
        const bool children_idle =
            std::none_of(node->children.begin(), node->children.end(), [group_set_id](const auto& child_entry) {
                const GroupSetTransferState state =
                    child_entry.second->group_set_resources[group_set_id].transfer_state;
                return state == GroupSetTransferState::LOAD_PENDING || state == GroupSetTransferState::LOADING;
            });
        return children_idle || context.skip(EvictionVictimSkipReason::CHILD_LOADING);
    };
    // Keep the std::function passed to best() at its original two-pointer capture size.
    const auto can_evict = [&context, &can_evict_uncached](TreeNode* node) {
        if (context.counters != nullptr) {
            ++context.counters->scanned_candidates;
        }
        if (context.eviction_batch_context->rejected_candidates.count(node) != 0) {
            return context.skip(EvictionVictimSkipReason::CACHED_REJECTION);
        }
        if (!can_evict_uncached(node)) {
            // Request releases may happen concurrently. Retry those candidates in
            // the next pass instead of repeatedly locking their pools in this one.
            context.eviction_batch_context->rejected_candidates.insert(node);
            return false;
        }
        return true;
    };
    std::optional<EvictionEntry> entry;
    while ((entry = heap->best(can_evict)).has_value()) {
        if (isEvictable(entry->node, group_set_id, tier)) {
            break;
        }
        context.skip(EvictionVictimSkipReason::INVALID_CANDIDATE);
        const GroupSetResource& resource = entry->node->group_set_resources[group_set_id];
        BTC_LOG_ERROR("invalid eviction candidate: node=%p key=%ld group_set_id=%zu source_tier=%s "
                          "top_tier=%s transfer_state=%d",
                          static_cast<void*>(entry->node),
                          entry->node->cache_key,
                          group_set_id,
                          tierName(tier),
                          tierName(resource.getTopTier()),
                          static_cast<int>(resource.transfer_state));
        heap->erase(entry->node);
    }
    if (!entry.has_value()) {
        return std::nullopt;
    }

    GroupSetResource& resource    = entry->node->group_set_resources[group_set_id];
    const Tier target_tier = force_drop ? Tier::NONE : watermarkTargetTier(tier);
    TransferDescriptor victim(entry->node,
                              group_set_id,
                              /*path_index=*/0,
                              tier,
                              target_tier,
                              resource.getBlocks(tier),
                              /*initial_owner=*/true);
    selected = true;
    return victim;
}

Tier BlockTreeEvictor::watermarkTargetTier(Tier source_tier) const {
    if (source_tier == Tier::DEVICE && is_tier_enabled_(Tier::HOST)) {
        return Tier::HOST;
    }
    if (source_tier == Tier::HOST && is_tier_enabled_(Tier::DISK)) {
        return Tier::DISK;
    }
    if (source_tier == Tier::HOST && host_writeback_backend_ && host_writeback_backend_->supportsWriteTier(Tier::HOST)) {
        return Tier::REMOTE;
    }
    return Tier::NONE;
}

size_t
BlockTreeEvictor::watermarkLogicalBatchLimit(const GroupSet& group_set, Tier source_tier, Tier target_tier) const {
    if (source_tier != Tier::HOST || target_tier != Tier::DISK || !group_set.pools().disk) {
        return max_descriptors_per_batch_;
    }
    size_t available_bytes = disk_cache_evict_max_pending_bytes_;
    bool   has_pending     = false;
    for (const auto& pending_group : tree_->groupSets()) {
        const size_t count =
            heaps_[pending_group->groupSetId()].watermark_rounds[static_cast<size_t>(Tier::HOST)].pending_count;
        if (count == 0) {
            continue;
        }
        has_pending        = true;
        const size_t bytes = pending_group->storageStrideBytes();
        if (count > available_bytes / bytes) {
            return 0;
        }
        available_bytes -= count * bytes;
    }
    size_t count = available_bytes / group_set.storageStrideBytes();
    if (count == 0 && !has_pending) {
        count = 1;
    }
    return std::min(count, max_descriptors_per_batch_);
}

void BlockTreeEvictor::scheduleWatermarkEvictionsLocked(Tier tier, std::optional<TierWatermark> tier_watermark) {
    const Tier   target_tier        = watermarkTargetTier(tier);
    const bool   direct_drop        = target_tier == Tier::NONE;
    const bool   is_host_disk       = tier == Tier::HOST && target_tier == Tier::DISK;
    const auto&  group_sets         = tree_->groupSets();
    const size_t first_group        = is_host_disk ? next_disk_evict_group_ : 0;
    bool         waiting_for_budget = false;
    for (size_t i = 0; i < group_sets.size(); ++i) {
        const size_t group_index = (first_group + i) % group_sets.size();
        const auto&  group_set   = group_sets[group_index];
        auto&        round       = heaps_[group_set->groupSetId()].watermark_rounds[static_cast<size_t>(tier)];
        if (direct_drop && round.pending_count != 0) {
            continue;
        }
        if (tier_watermark.has_value()
            && (!is_host_disk
                || (!round.watermark.has_value() && round.required_count == 0 && round.pending_count == 0))) {
            round.watermark = tier_watermark;
        }
        if (!round.watermark.has_value()) {
            continue;
        }
        const auto& watermark      = *round.watermark;
        size_t      required_count = computeWatermarkEvictCount(*group_set, tier, watermark);
        if (round.required_count == 0) {
            round.required_count = required_count;
        }
        bool eviction_reported = (is_host_disk || direct_drop) && round.scheduled_count != 0;
        while (required_count > 0 && !waiting_for_budget) {
            const size_t batch_count =
                std::min(required_count, watermarkLogicalBatchLimit(*group_set, tier, target_tier));
            if (batch_count == 0) {
                waiting_for_budget = true;
                // Let pending writes drain so smaller groups cannot starve this block.
                next_disk_evict_group_ = group_index;
                break;
            }
            const BatchEvictResult result = batchEvictLocked(group_set->groupSetId(), tier, batch_count);
            round.scheduled_count += result.scheduled_count;
            if (result.madeProgress() && !eviction_reported) {
                metrics_reporter_->reportEvictionTriggered(tier, group_set->groupType(), /*force_drop=*/false);
                eviction_reported = true;
            }
            if (!result.madeProgress()) {
                if (!is_host_disk || round.pending_count == 0) {
                    round.watermark.reset();
                }
                break;
            }
            if (is_host_disk && result.async_submitted) {
                next_disk_evict_group_ = (group_index + 1) % group_sets.size();
            }
            // Even rejected drop settlement must not turn into an inline drain.
            if (result.async_submitted || direct_drop) {
                break;
            }

            const size_t remaining_count = computeWatermarkEvictCount(*group_set, tier, watermark);
            if (remaining_count >= required_count) {
                break;
            }
            required_count = remaining_count;
        }
        if (!is_host_disk || !round.watermark.has_value() || required_count == 0) {
            finishWatermarkRoundLocked(group_set->groupSetId(), tier);
        }
    }
}

void BlockTreeEvictor::finishWatermarkRoundLocked(size_t group_set_id, Tier source_tier) {
    auto& round = heaps_[group_set_id].watermark_rounds[static_cast<size_t>(source_tier)];
    if (round.pending_count != 0) {
        return;
    }
    if (round.required_count != 0) {
        metrics_reporter_->reportEvictionBlocks(source_tier,
                                                tree_->groupSets()[group_set_id]->groupType(),
                                                /*force_drop=*/false,
                                                round.required_count,
                                                round.scheduled_count);
    }
    round = {};
}

void BlockTreeEvictor::updatePendingRelease(DescriptorView descs, bool reserve) {
    updatePendingReleaseBlocks(descs, reserve, /*revoked=*/false);
    for (const auto& desc : descs) {
        auto& pending_count =
            heaps_[desc.group_set_id].watermark_rounds[static_cast<size_t>(desc.source_tier)].pending_count;
        if (reserve) {
            ++pending_count;
        } else {
            BTC_CHECK(pending_count > 0);
            --pending_count;
        }
    }
}

void BlockTreeEvictor::updatePendingReleaseBlocks(DescriptorView descs, bool reserve, bool revoked) {
    if (descs.empty()) {
        return;
    }
    std::unordered_map<IBlockPool*, size_t> deltas;
    deltas.reserve(descs.size());
    const auto add_delta = [&deltas](IBlockPool* pool, size_t delta) {
        BTC_CHECK_WITH_INFO(pool != nullptr, "pending eviction release references a null pool");
        deltas[pool] += delta;
    };
    for (const TransferDescriptor& desc : descs) {
        if (!reserve) {
            if (revoked) {
                const auto& pending = heaps_[desc.group_set_id].pending_timings;
                const auto pos = pending.find({desc.node, desc.generation});
                if (pos == pending.end() || pos->second.retain_source) {
                    continue;
                }
            } else if (!desc.isCurrentOwner()) {
                continue;
            }
        }
        const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
        if (desc.source_tier == Tier::DEVICE) {
            const auto& pools = group_set->pools().device;
            BTC_CHECK_WITH_INFO(desc.source_blocks.size() == pools.size(),
                                    "pending eviction release device block count mismatch: group_set=%zu blocks=%zu "
                                    "pools=%zu",
                                    desc.group_set_id,
                                    desc.source_blocks.size(),
                                    pools.size());
            for (const auto& pool : pools) {
                add_delta(pool.get(), 1);
            }
        } else if (desc.source_tier == Tier::HOST) {
            add_delta(group_set->pools().host.get(), desc.source_blocks.size());
        }
    }

    std::lock_guard<std::mutex> lock(pending_release_mutex_);
    if (reserve) {
        for (const auto& [pool, delta] : deltas) {
            pending_release_counts_[pool] += delta;
        }
        return;
    }
    for (const auto& [pool, delta] : deltas) {
        const auto it = pending_release_counts_.find(pool);
        BTC_CHECK_WITH_INFO(it != pending_release_counts_.end() && it->second >= delta,
                                "invalid pending eviction release settlement: "
                                "pool=%s address=%p required=%zu pending=%zu",
                                pool->poolName().c_str(),
                                static_cast<const void*>(pool),
                                delta,
                                it == pending_release_counts_.end() ? 0 : it->second);
    }
    for (const auto& [pool, delta] : deltas) {
        auto it = pending_release_counts_.find(pool);
        it->second -= delta;
    }
}

// ---- Migration pipeline (begin -> copy -> finish) ----
void BlockTreeEvictor::collectFullPrune(const TransferDescriptor&                  eviction_desc,
                                        EvictionDropTask&                          task,
                                        std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const {
    if (task.hasFullPrune() || tree_->groupSets()[eviction_desc.group_set_id]->groupType() != CacheGroupType::FULL) {
        return;
    }

    // Reuse the output vector as the worklist. Reversing a parent-before-child
    // traversal gives the bottom-up order required by topology cleanup.
    task.full_prune_nodes_bottom_up.push_back(eviction_desc.node);
    for (size_t i = 0; i < task.full_prune_nodes_bottom_up.size(); ++i) {
        TreeNode* node = task.full_prune_nodes_bottom_up[i];
        for (const auto& [_, child] : node->children) {
            task.full_prune_nodes_bottom_up.push_back(child);
        }
        if (node == eviction_desc.node) {
            continue;
        }

        // Once a FULL prefix is removed, every descendant resource is unreachable
        // through matching. Prune every idle resource and detach in-flight transfers.
        for (const GroupSetPtr& group_set : tree_->groupSets()) {
            const size_t            group_set_id = group_set->groupSetId();
            const GroupSetResource& resource     = node->group_set_resources[group_set_id];
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                detached_resources.emplace_back(node, group_set_id);
                continue;
            }
            const Tier source_tier = resource.getTopTier();
            if (source_tier == Tier::NONE) {
                continue;
            }
            task.dependent_prune_descs.emplace_back(node,
                                                    group_set_id,
                                                    /*path_index=*/0,
                                                    source_tier,
                                                    Tier::NONE,
                                                    resource.getBlocks(source_tier),
                                                    /*initial_owner=*/true);
        }
    }
    std::reverse(task.full_prune_nodes_bottom_up.begin(), task.full_prune_nodes_bottom_up.end());
}

void BlockTreeEvictor::runDropTask(TransferDescriptor    eviction_desc,
                                   bool                  notify_settled,
                                   const char*           trigger_type,
                                   EvictionBatchContext* eviction_batch_context,
                                   DropScope             scope) {
    const EvictionDropTask task = createDropTask(std::move(eviction_desc), trigger_type, eviction_batch_context, scope);
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        completeDrop(dependent_desc);
    }
    if (!task.primary_desc.node->group_set_resources[task.primary_desc.group_set_id].transfer_detached) {
        completeDrop(task.primary_desc);
    }
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        completeDrop(cascade_desc);
    }
    if (task.hasFullPrune()) {
        TreeNode* const boundary_node = task.primary_desc.node->parent;
        for (TreeNode* node : task.full_prune_nodes_bottom_up) {
            eraseNodeFromAllHeaps(node);
            tree_->removeNode(node);
        }
        TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(boundary_node);
        updateFullCandidate(survivor);
    } else {
        settleSingleEviction(task.primary_desc.node);
    }
    if (notify_settled) {
        settled_(false);
    }
    metrics_reporter_->reportEvictionFinished(task, tree_->groupSets(), trigger_type);
}

void BlockTreeEvictor::completeDrop(const TransferDescriptor& desc) {
    const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
    auto&              resource  = desc.node->group_set_resources[desc.group_set_id];
    BTC_CHECK_WITH_INFO(resource.transfer_state == GroupSetTransferState::IDLE,
                            "drop completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s",
                            desc.group_set_id,
                            desc.node->cache_key,
                            static_cast<int>(resource.transfer_state),
                            tierName(desc.source_tier));

    suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
    const MultiNodeResource source_holder{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    resource.evictFromTier(desc.source_tier);
    tree_->refreshPublishedState(desc.node);
    admitCandidate(desc.node, desc.group_set_id, resource.getTopTier());
    BTC_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "drop settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            desc.group_set_id,
                            desc.node->cache_key);
}

void BlockTreeEvictor::completeEvict(DescriptorView descs, bool retain_source) {
    const TransferDescriptor&       first     = descs.front();
    const GroupSetPtr&              group_set = tree_->groupSets()[first.group_set_id];
    if (first.target_tier == Tier::REMOTE) {
        std::vector<TransferDescriptor> restore;
        for (const auto& desc : descs) {
            if (!desc.isCurrentOwner()) {
                continue;
            }
            auto& resource = desc.node->group_set_resources[desc.group_set_id];
            if (resource.transfer_detached || resource.candidate_meta.last_access_seq != resource.eviction_access_seq) {
                restore.push_back(desc);
                continue;
            }
            resource.transfer_state = GroupSetTransferState::IDLE;
            resource.evictFromTier(Tier::HOST);
            group_set->releaseBlocks(Tier::HOST, desc.source_blocks, BlockTreeRefType::CACHE);
            tree_->refreshPublishedState(desc.node);
            admitCandidate(desc.node, desc.group_set_id, resource.getTopTier());
        }
        restoreSource(restore);
        return;
    }
    MultiNodeResource               source_holder{first.group_set_id, first.source_tier, {}};
    MultiNodeResource               target_holder{first.group_set_id, first.target_tier, {}};
    MultiNodeResource               active_target_holder{first.group_set_id, first.target_tier, {}};
    std::vector<TransferDescriptor> restore_descs;
    source_holder.node_blocks.reserve(descs.size());
    target_holder.node_blocks.reserve(descs.size());
    active_target_holder.node_blocks.reserve(descs.size());

    const int64_t tier_enter_time_us = cacheNowUs();
    for (const TransferDescriptor& desc : descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        if (!desc.isCurrentOwner()) {
            continue;
        }
        if (resource.transfer_detached) {
            restore_descs.push_back(desc);
            continue;
        }
        BTC_CHECK_WITH_INFO(
            resource.transfer_state == GroupSetTransferState::EVICTING,
            "eviction completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s target=%s",
            desc.group_set_id,
            desc.node->cache_key,
            static_cast<int>(resource.transfer_state),
            tierName(desc.source_tier),
            tierName(desc.target_tier));
        if (!retain_source && resource.candidate_meta.last_access_seq != resource.eviction_access_seq) {
            restore_descs.push_back(desc);
            continue;
        }
        if (retain_source) {
            resource.transfer_state = GroupSetTransferState::IDLE;
            if (tree_->retainHostReplica(desc.node, desc.group_set_id, desc.target_blocks.front())) {
                admitCandidate(desc.node, desc.group_set_id, Tier::HOST);
            }
            admitCandidate(desc.node, desc.group_set_id, Tier::DEVICE);
            continue;
        }
        active_target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        source_holder.node_blocks.emplace_back(desc.node, desc.source_blocks);

        resource.evictFromTier(desc.source_tier);
        resource.transfer_state = GroupSetTransferState::IDLE;
        resource.setBlocks(desc.target_tier, desc.target_blocks);
        if (desc.target_tier == Tier::DISK && resource.hasCompleteDeviceValue()) {
            resource.retained_disk_replica = true;
        }
        tree_->refreshPublishedState(desc.node);
        BTC_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                                desc.group_set_id,
                                desc.node->cache_key);

        resource.candidate_meta.admission_seq         = ++admission_seq_;
        resource.candidate_meta.tier_enter_time_us    = tier_enter_time_us;
        resource.candidate_meta.last_access_time_us   = tier_enter_time_us;
        resource.candidate_meta.tier_last_hit_time_us = 0;
        admitCandidate(desc.node, desc.group_set_id, desc.target_tier);
    }

    if (!active_target_holder.node_blocks.empty()) {
        group_set->referenceBlocks(active_target_holder, BlockTreeRefType::CACHE);
    }
    group_set->unreferenceBlocks(target_holder, BlockTreeRefType::EVICT);
    if (!source_holder.node_blocks.empty()) {
        group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    }

    restoreSource(restore_descs);
}

void BlockTreeEvictor::rollbackTransferLocked(DescriptorView descs) {
    releaseTargetBlocks(descs);
    restoreSource(descs);
    settleEviction(descs);
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
void BlockTreeEvictor::restoreSource(DescriptorView eviction_descs) {
    std::vector<TransferDescriptor> detached_descs;
    for (const TransferDescriptor& desc : eviction_descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (!desc.isCurrentOwner()) {
            continue;
        }
        if (resource.transfer_detached) {
            detached_descs.push_back(desc);
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::EVICTING) {
            BTC_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", desc.group_set_id, desc.node->cache_key);
            continue;
        }
        resource.transfer_state = GroupSetTransferState::IDLE;
        tree_->refreshPublishedState(desc.node);
        BTC_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                                desc.group_set_id,
                                desc.node->cache_key);

        admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
    }
    discardDetachedTransfer(detached_descs);
}

void BlockTreeEvictor::discardDetachedTransfer(DescriptorView transfer_descs) {
    std::map<std::pair<size_t, Tier>, MultiNodeResource> source_holders;
    for (const TransferDescriptor& desc : transfer_descs) {
        GroupSetResource& resource     = desc.node->group_set_resources[desc.group_set_id];
        const Tier        serving_tier = resource.getTopTier();
        auto&             holder       = source_holders
                           .try_emplace(std::make_pair(desc.group_set_id, serving_tier),
                                        MultiNodeResource{desc.group_set_id, serving_tier, {}})
                           .first->second;
        holder.node_blocks.emplace_back(desc.node, resource.getBlocks(serving_tier));
        resource.evictFromTier(serving_tier);
        resource.transfer_state    = GroupSetTransferState::IDLE;
        resource.transfer_detached = false;
        tree_->refreshPublishedState(desc.node);
    }
    for (const auto& [key, holder] : source_holders) {
        tree_->groupSets()[holder.group_set_id]->unreferenceBlocks(holder, BlockTreeRefType::CACHE);
    }
    for (const auto& desc : transfer_descs) {
        updateFullCandidate(desc.node->parent, desc.group_set_id);
    }
}

void BlockTreeEvictor::invalidateSources(const std::vector<InvalidationSource>& sources) {
    for (const auto& source : sources) {
        const auto path = tree_->findNode(source.cache_keys);
        // findNode returns the surviving prefix when an earlier drop removed this source.
        if (!path.empty() && path.size() == source.cache_keys.size()) {
            invalidateSource(path.back(), source.group_set_id, source.source_tier, DropScope::CASCADE);
        }
    }
}

void BlockTreeEvictor::invalidateSource(TreeNode* node, size_t group_id, Tier source_tier, DropScope scope) {
    auto& resource = node->group_set_resources[group_id];
    if (!resource.hasTier(source_tier)) {
        return;
    }
    if (resource.transfer_state != GroupSetTransferState::IDLE
        && (scope == DropScope::PRIMARY_ONLY || tree_->groupSets()[group_id]->groupType() != CacheGroupType::FULL)) {
        resource.transfer_detached = true;
        tree_->refreshPublishedState(node);
        return;
    }
    if (resource.retained_host_replica || resource.retained_disk_replica) {
        scope = DropScope::PRIMARY_ONLY;
    }
    runDropTask(TransferDescriptor(node, group_id, 0, source_tier, Tier::NONE, resource.getBlocks(source_tier), true),
                false, "invalidate", nullptr, scope);
}

void BlockTreeEvictor::releaseTargetBlocks(DescriptorView descs) {
    if (descs.empty()) {
        return;
    }

    BlockIdList target_blocks;
    target_blocks.reserve(descs.size());
    for (const TransferDescriptor& desc : descs) {
        target_blocks.insert(target_blocks.end(), desc.target_blocks.begin(), desc.target_blocks.end());
    }

    if (!target_blocks.empty()) {
        const TransferDescriptor& first = descs.front();
        tree_->groupSets()[first.group_set_id]->releaseBlocks(
            first.target_tier, target_blocks, BlockTreeRefType::EVICT);
    }
}

void BlockTreeEvictor::settleEviction(DescriptorView descs) {
    std::map<std::pair<size_t, Tier>, MultiNodeResource> source_holders;
    std::unordered_map<TreeNode*, size_t>                pending_nodes;
    std::unordered_set<TreeNode*>                        refresh_nodes;
    pending_nodes.reserve(descs.size());
    refresh_nodes.reserve(descs.size());
    for (const TransferDescriptor& desc : descs) {
        if (!desc.isCurrentOwner()) {
            auto& holder = source_holders
                               .try_emplace(std::make_pair(desc.group_set_id, desc.source_tier),
                                            MultiNodeResource{desc.group_set_id, desc.source_tier, {}})
                               .first->second;
            holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
            ++pending_nodes[desc.node];
        }
        if (desc.node->parent == nullptr) {
            pending_nodes.try_emplace(desc.node, 0);
        } else {
            refresh_nodes.insert(desc.node->parent);
        }
    }

    for (const auto& [key, holder] : source_holders) {
        tree_->groupSets()[holder.group_set_id]->unreferenceBlocks(holder, BlockTreeRefType::EVICT);
    }
    tree_->removePendingNodes(pending_nodes);
    for (TreeNode* node : refresh_nodes) {
        updateFullCandidate(node);
    }
}

void BlockTreeEvictor::settleSingleEviction(TreeNode* node) {
    if (tree_->isRemovable(node)) {
        BTC_LOG_DEBUG("deleting empty node key=%ld", node->cache_key);
        eraseNodeFromAllHeaps(node);
        TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(node);
        updateFullCandidate(survivor);
    } else {
        updateFullCandidate(node->parent);
    }
}

void BlockTreeEvictor::eraseNodeFromAllHeaps(TreeNode* node) {
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            heapFor(group_set->groupSetId(), tier)->erase(node);
        }
    }
}

bool BlockTreeEvictor::canReclaimDeviceDrop(const TransferDescriptor& desc,
                                            EvictionBatchContext*     eviction_batch_context,
                                            DropScope                 scope) const {
    const auto can_reclaim = [this](const TransferDescriptor& candidate) {
        if (candidate.source_tier != Tier::DEVICE) {
            return true;
        }
        const auto& pools = tree_->groupSets()[candidate.group_set_id]->pools().device;
        for (size_t i = 0; i < candidate.source_blocks.size(); ++i) {
            if (!isNullBlockIdx(candidate.source_blocks[i])
                && !pools[i]->canReclaimCacheBlock(candidate.source_blocks[i])) {
                return false;
            }
        }
        return true;
    };
    if (!can_reclaim(desc)) {
        return false;
    }
    if (tree_->groupSets()[desc.group_set_id]->groupType() == CacheGroupType::FULL) {
        // FULL usually shares request ownership with its related resources. Rarely,
        // an upward cascade can reach a request-held ancestor whose SWA was dropped.
        // Check only the primary resource to avoid planning the closure twice; any
        // held related block stays physically alive through its request reference.
        return scope == DropScope::CASCADE;
    }
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    const EvictionDropTask task = planDropTask(desc, detached_resources, eviction_batch_context, DropScope::CASCADE);
    if (scope == DropScope::PRIMARY_ONLY) {
        // A request-held FULL sibling must actually be in the original cascade.
        for (const auto& sibling : task.cascade_descs) {
            const auto& group_set = tree_->groupSets()[sibling.group_set_id];
            if (sibling.node != desc.node || group_set->groupType() != CacheGroupType::FULL) {
                continue;
            }
            for (size_t i = 0; i < sibling.source_blocks.size(); ++i) {
                const auto block = sibling.source_blocks[i];
                if (!isNullBlockIdx(block) && group_set->pools().device[i]->hasRequestRef(block)) {
                    return true;
                }
            }
        }
        return false;
    }
    // FULL pruning and cascades must not discard cache entries backed by held
    // device blocks either. Planning does not reserve or detach any resource.
    return std::all_of(task.cascade_descs.begin(), task.cascade_descs.end(), can_reclaim)
           && std::all_of(task.dependent_prune_descs.begin(), task.dependent_prune_descs.end(), can_reclaim)
           && std::none_of(detached_resources.begin(), detached_resources.end(), [](const auto& entry) {
                  return entry.first->group_set_resources[entry.second].hasTier(Tier::DEVICE);
              });
}

EvictionDropTask BlockTreeEvictor::planDropTask(TransferDescriptor                         eviction_desc,
                                                std::vector<std::pair<TreeNode*, size_t>>& detached_resources,
                                                EvictionBatchContext*                      eviction_batch_context,
                                                DropScope                                  scope) const {
    EvictionDropTask task;
    task.primary_timing =
        EvictionTimingMetrics(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    task.primary_desc = std::move(eviction_desc);
    if (scope == DropScope::PRIMARY_ONLY) {
        return task;
    }
    const TransferDescriptor& primary_desc = task.primary_desc;
    const Tier                target_tier  = primary_desc.target_tier;
    collectFullPrune(primary_desc, task, detached_resources);
    TreeNode* const                           primary_node     = primary_desc.node;
    const size_t                              primary_group_id = primary_desc.group_set_id;
    const Tier                                source_tier      = primary_desc.source_tier;
    const auto&                               group_sets       = tree_->groupSets();
    if (primary_node->group_set_resources[primary_group_id].transfer_state != GroupSetTransferState::IDLE) {
        detached_resources.emplace_back(primary_node, primary_group_id);
    }
    std::vector<std::pair<TreeNode*, size_t>> busy_root_cascades;

    const auto append_desc = [&](TreeNode* node, size_t group_set_id) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(node,
                                        group_set_id,
                                        /*path_index=*/0,
                                        source_tier,
                                        target_tier,
                                        resource.getBlocks(source_tier),
                                        /*initial_owner=*/true);
    };

    const bool           primary_is_leaf = tree_->isLeafAtTier(primary_node, primary_group_id, source_tier);
    const CacheGroupType primary_type    = group_sets[primary_group_id]->groupType();
    for (const GroupSetPtr& group_set : group_sets) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set_id == primary_group_id) {
            continue;
        }

        bool should_cascade = primary_is_leaf;
        if (!should_cascade) {
            switch (primary_type) {
                case CacheGroupType::FULL:
                    should_cascade = group_set->groupType() == CacheGroupType::SWA
                                     || group_set->groupType() == CacheGroupType::LINEAR;
                    break;
                case CacheGroupType::SWA:
                    should_cascade = group_set->groupType() == CacheGroupType::LINEAR;
                    break;
                case CacheGroupType::LINEAR:
                    break;
            }
        }
        if (!should_cascade) {
            continue;
        }
        const GroupSetResource& resource = primary_node->group_set_resources[group_set_id];
        if (resource.transfer_state != GroupSetTransferState::IDLE) {
            busy_root_cascades.emplace_back(primary_node, group_set_id);
            continue;
        }
        if (!isEvictable(primary_node, group_set_id, source_tier)) {
            continue;
        }

        append_desc(primary_node, group_set_id);
        collectFullPrune(task.cascade_descs.back(), task, detached_resources);
    }

    if (task.hasFullPrune()) {
        detached_resources.insert(detached_resources.end(), busy_root_cascades.begin(), busy_root_cascades.end());
        // Keep the later closure-root cleanup, using one serving tier per resource.
        // Primary and cascade descriptors already own their selected root resources.
        TreeNode* const closure_root = task.full_prune_nodes_bottom_up.back();
        for (const GroupSetPtr& group_set : group_sets) {
            const size_t group_set_id = group_set->groupSetId();
            if ((task.primary_desc.node == closure_root && task.primary_desc.group_set_id == group_set_id)
                || std::any_of(task.cascade_descs.begin(), task.cascade_descs.end(), [&](const auto& desc) {
                       return desc.node == closure_root && desc.group_set_id == group_set_id;
                   })) {
                continue;
            }
            const GroupSetResource& resource = closure_root->group_set_resources[group_set_id];
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                const auto entry = std::make_pair(closure_root, group_set_id);
                if (std::find(detached_resources.begin(), detached_resources.end(), entry)
                    == detached_resources.end()) {
                    detached_resources.push_back(entry);
                }
            } else if (!resource.is_empty()) {
                const Tier tier = resource.getTopTier();
                task.dependent_prune_descs.emplace_back(closure_root,
                                                        group_set_id,
                                                        /*path_index=*/0,
                                                        tier,
                                                        Tier::NONE,
                                                        resource.getBlocks(tier),
                                                        /*initial_owner=*/true);
            }
        }
    }

    if (detached_resources.empty()) {
        selectUpwardCascades(task, eviction_batch_context);
    }

    return task;
}

EvictionDropTask BlockTreeEvictor::createDropTask(TransferDescriptor    eviction_desc,
                                                  const char*           trigger_type,
                                                  EvictionBatchContext* eviction_batch_context,
                                                  DropScope             scope) {
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    EvictionDropTask task = planDropTask(std::move(eviction_desc), detached_resources, eviction_batch_context, scope);
    const auto&      group_sets = tree_->groupSets();

    task.dependent_prune_timings.resize(task.dependent_prune_descs.size());
    for (size_t i = 0; i < task.dependent_prune_descs.size(); ++i) {
        const TransferDescriptor& desc = task.dependent_prune_descs[i];
        task.dependent_prune_timings[i] =
            EvictionTimingMetrics(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }
    task.cascade_timings.reserve(task.cascade_descs.size());
    for (const TransferDescriptor& desc : task.cascade_descs) {
        task.cascade_timings.emplace_back(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }

    if (eviction_batch_context != nullptr) {
        auto& endpoint_matchable = eviction_batch_context->endpoint_matchable;
        // Removing a subtree preserves every surviving prefix. Interior resource
        // changes can invalidate descendant windows, so discard the whole snapshot.
        if (task.hasFullPrune() || task.primary_desc.node->children.empty()) {
            endpoint_matchable.erase(task.primary_desc.node);
            for (TreeNode* node : task.full_prune_nodes_bottom_up) {
                endpoint_matchable.erase(node);
            }
            for (const auto& desc : task.cascade_descs) {
                endpoint_matchable.erase(desc.node);
            }
        } else if (!endpoint_matchable.empty()) {
            endpoint_matchable.clear();
        }
    }

    for (const auto& [node, group_set_id] : detached_resources) {
        GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_detached) {
            continue;
        }
        const Tier               tier = resource.getTopTier();
        const TransferDescriptor desc{
            node, group_set_id, 0, tier, Tier::NONE, resource.getBlocks(tier), /*initial_owner=*/true};
        const auto timing = resource.transfer_state == GroupSetTransferState::EVICTING ?
                                heaps_[group_set_id].pending_timings.at({node, resource.generation}).timing :
                                EvictionTimingMetrics(resource.candidate_meta);
        if (node == task.primary_desc.node && group_set_id == task.primary_desc.group_set_id) {
            task.primary_timing = timing;
        } else {
            metrics_reporter_->reportEvictionDescriptor(desc, timing, group_sets, trigger_type, true);
        }
        resource.transfer_detached = true;
        tree_->refreshPublishedState(node);
    }
    return task;
}

void BlockTreeEvictor::selectUpwardCascades(EvictionDropTask&     task,
                                            EvictionBatchContext* eviction_batch_context) const {
    const TransferDescriptor& primary_desc = task.primary_desc;
    TreeNode* const           primary_node = primary_desc.node;
    const Tier                target_tier  = primary_desc.target_tier;
    const auto&               group_sets   = tree_->groupSets();

    std::vector<bool> selected(group_sets.size(), false);
    selected[primary_desc.group_set_id] = true;
    for (const TransferDescriptor& desc : task.cascade_descs) {
        if (desc.node == primary_node) {
            selected[desc.group_set_id] = true;
        }
    }
    for (const TransferDescriptor& desc : task.dependent_prune_descs) {
        if (desc.node == primary_node) {
            selected[desc.group_set_id] = true;
        }
    }
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        const GroupSetResource& resource = primary_node->group_set_resources[group_set_id];
        if (!resource.is_empty() && (resource.getTopTier() != primary_desc.source_tier || !selected[group_set_id])) {
            return;
        }
    }
    if (!task.hasFullPrune() && !primary_node->children.empty()) {
        return;
    }

    const auto append_desc = [&](TreeNode* node, size_t group_set_id, Tier source_tier) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(node,
                                        group_set_id,
                                        /*path_index=*/0,
                                        source_tier,
                                        target_tier,
                                        resource.getBlocks(source_tier),
                                        /*initial_owner=*/true);
    };

    std::vector<TreeNode*> ancestors;
    std::vector<bool>      endpoint_matchable;
    if (eviction_batch_context == nullptr
        || eviction_batch_context->endpoint_matchable.count(primary_node->parent) == 0) {
        for (TreeNode* node = primary_node->parent; node != tree_->root(); node = node->parent) {
            ancestors.push_back(node);
        }
        std::reverse(ancestors.begin(), ancestors.end());

        std::vector<std::unique_ptr<MatchValidator>> match_validators;
        match_validators.reserve(group_sets.size());
        for (const GroupSetPtr& group_set : group_sets) {
            match_validators.push_back(group_set->createMatchValidator());
        }
        endpoint_matchable.reserve(ancestors.size());
        for (TreeNode* node : ancestors) {
            bool all_groups_valid = true;
            for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
                if (!match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                    all_groups_valid = false;
                }
            }
            endpoint_matchable.push_back(all_groups_valid);
            if (eviction_batch_context != nullptr) {
                eviction_batch_context->endpoint_matchable[node] = all_groups_valid;
            }
        }
    }

    size_t path_index = ancestors.size();
    for (TreeNode* parent = primary_node->parent; parent != tree_->root(); parent = parent->parent) {
        const bool matchable = eviction_batch_context ? eviction_batch_context->endpoint_matchable.at(parent) :
                                                        endpoint_matchable[--path_index];
        if (parent->is_resident || matchable) {
            break;
        }

        if (parent->children.size() != 1) {
            break;
        }

        const size_t desc_begin = task.cascade_descs.size();
        bool         releasable = true;
        for (const GroupSetPtr& group_set : group_sets) {
            const size_t            group_set_id = group_set->groupSetId();
            const GroupSetResource& resource     = parent->group_set_resources[group_set_id];
            if (resource.is_empty()) {
                continue;
            }
            const Tier source_tier = resource.getTopTier();
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                releasable = false;
                break;
            }
            append_desc(parent, group_set_id, source_tier);
        }
        if (!releasable) {
            task.cascade_descs.resize(desc_begin);
            break;
        }
    }
}

void BlockTreeEvictor::balanceDevicePagesLocked(const IBlockPool& pool, const std::vector<BlockIdList>& slots,
                                               size_t eligible_count, size_t capacity, size_t free_slots,
                                               size_t returning, TierWatermark watermark, bool& triggered) {
    // Several native holds (and several tree sites) can refer to one allocator
    // slot. Credit that slot only if every remaining cache reference is moving.
    std::unordered_map<BlockIdxType, size_t> physical;
    BlockIdList eligible;
    for (size_t i = 0; i < slots.size(); ++i) {
        for (const auto block : slots[i]) {
            physical.emplace(block, i);
            if (i < eligible_count)
                eligible.push_back(block);
        }
    }
    std::unordered_map<BlockIdxType, size_t> moving;
    for (const auto& group : tree_->groupSets()) {
        const auto id = group->groupSetId();
        for (const auto& [site, pending] : heaps_[id].pending_timings) {
            if (pending.retain_source || pending.source_tier != Tier::DEVICE)
                continue;
            auto& value = site.first->group_set_resources[id];
            if (value.generation != site.second || value.transfer_detached
                || value.eviction_access_seq != value.candidate_meta.last_access_seq)
                continue;
            bool revoked = false;
            for (auto block : value.device_blocks) {
                const auto it = physical.find(block);
                revoked |= it != physical.end() && it->second >= eligible_count;
            }
            if (revoked) {
                // A new request or reuse watcher protects this source. Use the
                // ordinary access cancellation at settlement; DMA is still live.
                value.candidate_meta.last_access_seq = nextAccessSeq();
            } else {
                for (auto block : value.device_blocks)
                    ++moving[block];
            }
        }
    }
    size_t pending = returning;
    for (size_t i = 0; i < eligible_count; ++i) {
        pending += std::all_of(slots[i].begin(), slots[i].end(), [&](auto block) {
            const auto it = moving.find(block);
            return it != moving.end() && it->second == pool.unpinnedCacheReferences(block);
        });
    }
    const auto need = watermarkNeed(capacity, capacity - free_slots, pending, watermark);
    const auto demand = evictionDemand(need, pending != 0, triggered);
    if (demand && eligible_count)
        releaseDevicePagesLocked(pool, slots, eligible_count, eligible, demand, false, true, false);
}

void BlockTreeEvictor::balanceHostPoolsLocked(TierWatermark watermark) {
    std::unordered_set<const HostBlockPool*> seen;
    for (const auto& group : tree_->groupSets()) {
        const auto& pool = group->pools().host;
        if (!pool || !seen.insert(pool.get()).second)
            continue;
        const auto pending = pendingHostReleases(*pool);
        const auto need = watermarkNeed(pool->totalBlocksNum(), pool->usedBlocksNum(), pending, watermark);
        auto& triggered = heaps_[group->groupSetId()].watermark_triggered[static_cast<size_t>(Tier::HOST)];
        const auto demand = evictionDemand(need, pending != 0, triggered);
        if (demand && host_writeback_backend_) {
            scheduleHostReleaseLocked(*pool, pending + demand);
        } else {
            // A Host-only configuration may discard its coldest unpinned
            // local value. Search all groups sharing this physical pool.
            for (size_t i = 0; i < demand; ++i) {
                if (!dropColdHostPageLocked(group->groupSetId(), "watermark"))
                    break;
            }
        }
    }
}

size_t
BlockTreeEvictor::computeWatermarkEvictCount(const GroupSet& group_set, Tier tier, const TierWatermark& watermark) {
    if (tier != Tier::DEVICE && tier != Tier::HOST && tier != Tier::DISK) {
        return 0;
    }

    std::vector<IBlockPool*> pools;
    if (tier == Tier::DEVICE) {
        for (const auto& pool : group_set.pools().device) {
            pools.push_back(pool.get());
        }
    } else if (tier == Tier::HOST && group_set.pools().host != nullptr) {
        pools.push_back(group_set.pools().host.get());
    } else if (tier == Tier::DISK && group_set.pools().disk != nullptr) {
        pools.push_back(group_set.pools().disk.get());
    }

    bool   has_pending    = false;
    bool   high_reached   = false;
    size_t required_count = 0;
    for (IBlockPool* pool : pools) {
        size_t pending_count = 0;
        {
            std::lock_guard<std::mutex> lock(pending_release_mutex_);
            const auto                  pending = pending_release_counts_.find(pool);
            if (pending != pending_release_counts_.end()) {
                pending_count = pending->second;
            }
        }
        const size_t used  = pool->usedBlocksNum();
        const size_t total = pool->totalBlocksNum();
        if (pending_count > 0) {
            BTC_CHECK_WITH_INFO(pending_count <= used,
                                    "pending eviction releases exceed used blocks: "
                                    "pool=%s address=%p pending=%zu used=%zu total=%zu low_ratio=%f high_ratio=%f",
                                    pool->poolName().c_str(),
                                    static_cast<const void*>(pool),
                                    pending_count,
                                    used,
                                    total,
                                    watermark.low_ratio,
                                    watermark.high_ratio);
        }
        const auto need = watermarkNeed(total, used, pending_count, watermark);
        has_pending  = has_pending || pending_count != 0;
        high_reached = high_reached || need.high;
        required_count = std::max(required_count, need.to_low);
    }

    bool& triggered = heaps_[group_set.groupSetId()].watermark_triggered[static_cast<size_t>(tier)];
    return evictionDemand({high_reached, required_count}, has_pending, triggered);
}

}  // namespace rtp_llm
