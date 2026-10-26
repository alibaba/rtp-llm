#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                               BlockTreeCacheConfig                           config,
                               std::shared_ptr<StorageBackend>                storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>             task_pool,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    metrics_reporter_(std::move(metrics_reporter)),
    evictor_(
        tree_.get(),
        config_.device_eviction_policy,
        config_.host_eviction_policy,
        config_.disk_eviction_policy,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        *metrics_reporter_,
        mutex_,
        config_.host_cache_sync_timeout_ms,
        config_.disk_cache_sync_timeout_ms,
        config_.max_descriptors_per_transfer_batch,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }),
    loader_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.disk_cache_sync_timeout_ms,
            config_.host_cache_sync_timeout_ms,
            config_.enable_device_cache,
            storage_backend_,
            [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }),
    storer_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.host_cache_sync_timeout_ms,
            config_.disk_cache_sync_timeout_ms,
            storage_backend_,
            [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.task_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     tree_->groupSets().size(),
                     tree_->reusableGroupCount(),
                     config_.task_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_host_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    if (config_.full_prefix_scan_interval_ms > 0) {
        auto scanner =
            std::make_unique<FullPrefixInvariantScanner>(*tree_, mutex_, config_.full_prefix_scan_interval_ms);
        if (scanner->start()) {
            full_prefix_scanner_ = std::move(scanner);
            RTP_LLM_LOG_INFO("FULL prefix invariant scanner enabled, interval_ms=%ld",
                             config_.full_prefix_scan_interval_ms);
        }
    }
    initialized_ = true;
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    full_prefix_scanner_.reset();
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    shutdownLoads();
    if (storage_backend_) {
        storage_backend_->shutdown();
    }
    RTP_LLM_LOG_INFO("load tickets closed, stopping store admission and draining cache tasks...");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storer_.stopAdmissionLocked();
    }
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();
    task_pool_->waitForIdle();
    // Task-body idleness does not imply asynchronous RPC/transfer completion.
    // Keep dispatcher and cache state alive until transfer callbacks return,
    // then drain the settlement tasks those callbacks submitted.
    transfer_dispatcher_->drainTransfers();
    task_pool_->waitForIdle();
    transfer_dispatcher_.reset();
    task_pool_.reset();
    RTP_LLM_LOG_INFO("destroyed");
}

void BlockTreeCache::shutdownLoads() {
    loader_.shutdown();
}

TransferResult BlockTreeCache::executeTransfer(TransferTask task) {
    auto context = transfer_dispatcher_->executePerRank(std::move(task));
    context->waitDone();
    auto result = context->result();
    if (!result.ok()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", result.status.ToString().c_str());
    }
    return result;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType&               cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix,
                                           const PartialTailMatchRequest*     partial_tail) {
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result = loader_.matchLocked(cache_keys, valid_prefix, partial_tail);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    (void)insert(cache_keys, resources, target_tier, false, nullptr);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident,
                              const std::vector<std::vector<int>>*              token_segments,
                              const std::vector<size_t>*                         token_ends,
                              bool internal_fork,
                              bool* ready_after_insert) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    if (ready_after_insert != nullptr) {
        *ready_after_insert = false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (internal_fork) {
            // Recheck the proposed branch under the same mutation lock as the
            // existing storer. A short endpoint cannot become visible with
            // only a subset of the required group histories.
            if (target_tier != Tier::DEVICE || token_ends == nullptr || token_segments == nullptr
                || cache_keys.empty() || cache_keys.size() != resources.size()
                || cache_keys.size() != token_ends->size() || cache_keys.size() != token_segments->size()) {
                return 0;
            }
            const auto existing = tree_->findNode(cache_keys);
            const size_t ancestor_count = cache_keys.size() - 1;
            if (existing.size() < ancestor_count || (ancestor_count > 0 && tree_->isShortLeaf(existing[ancestor_count - 1]))) {
                return 0;
            }
            for (size_t index = 0; index < ancestor_count; ++index) {
                if (existing[index]->token_end != (*token_ends)[index]
                    || existing[index]->token_ids != (*token_segments)[index]) {
                    return 0;
                }
            }
            TreeNode projected_root;
            projected_root.token_end = 0;
            TreeNode projected;
            projected.parent = ancestor_count > 0 ? existing[ancestor_count - 1] : &projected_root;
            projected.token_end = token_ends->back();
            projected.group_set_resources.resize(tree_->groupSets().size());
            const TreeNode* present = existing.size() == cache_keys.size() ? existing.back() : nullptr;
            if (present != nullptr && (present->token_end != projected.token_end
                                       || present->token_ids != token_segments->back())) {
                return 0;
            }
            if (resources.back().size() != tree_->groupSets().size()) {
                return 0;
            }
            for (size_t group_id = 0; group_id < tree_->groupSets().size(); ++group_id) {
                const auto& incoming = resources.back()[group_id];
                const auto& retained = present != nullptr ? present->group_set_resources[group_id] : incoming;
                projected.group_set_resources[group_id] =
                    present != nullptr && !retained.is_empty() ? retained : incoming;
            }
            std::vector<TreeNode*> projected_path(existing.begin(), existing.begin() + ancestor_count);
            projected_path.push_back(&projected);
            for (const auto& group : tree_->groupSets()) {
                if (!group->hasDeviceCoverage(projected_path, projected.token_end)) {
                    return 0;
                }
            }
        }
        storage_write =
            storer_.storeLocked(cache_keys,
                                resources,
                               target_tier,
                               is_resident,
                               resident_prefix_length,
                               token_segments,
                               token_ends,
                                internal_fork,
                                ready_after_insert);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

CachedPrefixResources BlockTreeCache::pinRegisteredGroups(const std::vector<int>& tokens,
                                                           const CacheKeysType& base_keys,
                                                           size_t base_tokens,
                                                           const std::vector<size_t>& endpoints,
                                                           const std::vector<PrefixCachePlanPoint>* exact_points,
                                                           const std::function<bool(size_t, const std::vector<int>&,
                                                                                    CacheKeyType)>& source_identity) {
    CachedPrefixResources result;
    result.groups.resize(endpoints.size(), std::vector<CachedPrefixResources::Group>(tree_->groupSets().size()));
    if (endpoints.empty() || base_tokens == 0
        || base_keys.size() < (tokens.size() + base_tokens - 1) / base_tokens) {
        return result;
    }
    std::unordered_map<size_t, CacheKeyType> point_keys;
    if (exact_points != nullptr) {
        point_keys.reserve(exact_points->size());
        for (const auto& point : *exact_points) {
            point_keys.emplace(point.pos, point.key);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto endpoint_key = [&](size_t pos) {
        if (const auto point = point_keys.find(pos); point != point_keys.end()) {
            return point->second;
        }
        return base_keys[(pos - 1) / base_tokens];
    };
    std::vector<BlockTreeContentRange> ranges;
    if (source_identity) {
        ranges = tree_->contentMatchRanges(tokens, endpoint_key, source_identity);
    } else {
        for (auto* node : tree_->contentMatchCandidates(tokens, endpoint_key)) {
            ranges.push_back({node, node->token_end});
        }
        std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.matched_end > rhs.matched_end;
        });
    }
    std::map<std::pair<size_t, const TreeNode*>, size_t> path_indices;
    std::set<std::pair<std::string, BlockIdxType>> unique_pins;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIndicesType>> acquisitions;
    size_t acquired = 0;
    try {
        // Finish descriptor allocation before acquiring references. Incomplete
        // groups contribute no partial pins; complete groups may be retained.
        for (size_t row = 0; row < endpoints.size(); ++row) {
            const size_t endpoint = endpoints[row];
            for (size_t id = 0; id < tree_->groupSets().size(); ++id) {
                auto& output = result.groups[row][id];
                const auto& group = tree_->groupSets()[id];
                const auto sources = tree_->coveredDeviceGroupPath(ranges, endpoint, id);
                if (!sources) {
                    continue;
                }
                output.ready = true;
                for (auto* source : *sources) {
                    const auto& value = source->group_set_resources[id];
                    const auto key = std::make_pair(id, static_cast<const TreeNode*>(source));
                    auto existing = path_indices.find(key);
                    if (existing == path_indices.end()) {
                        const size_t entry = result.path_resources.size();
                        GroupSetResource device_only;
                        device_only.device_blocks = value.device_blocks;
                        result.path_resources.push_back({id, source->token_end, std::move(device_only)});
                        existing = path_indices.emplace(key, entry).first;
                    }
                    output.path_indices.push_back(existing->second);
                    for (size_t member = 0; member < group->groupTags().size(); ++member) {
                        const auto& tag = group->groupTags()[member];
                        const auto block = value.device_blocks[member];
                        if (unique_pins.emplace(tag, block).second) {
                            result.pins.push_back({tag, block});
                            acquisitions.push_back({group->devicePools()[member], BlockIndicesType{block}});
                        }
                    }
                }
                // A later immutable dense source can alias the same physical
                // prefix at F. Exact-state groups only resolve a node at F.
                if (!sources->empty() && (!group->requiresExactStateEndpoint()
                                          || sources->back()->token_end == endpoint)) {
                    output.endpoint_resource = result.path_resources[output.path_indices.back()].resource;
                }
            }
        }
        for (; acquired < acquisitions.size(); ++acquired) {
            acquisitions[acquired].first->incRef(acquisitions[acquired].second);
        }
    } catch (...) {
        while (acquired > 0) {
            --acquired;
            acquisitions[acquired].first->decRef(acquisitions[acquired].second);
        }
        throw;
    }
    return result;
}

int BlockTreeCache::evictForGroup(std::string_view group_tag, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return 0;
    }
    const GroupSetPtr& group_set   = tree_->groupSets()[location->group_set_id];
    const auto&        device_pool = group_set->devicePools()[location->member_group_id];

    const size_t initial_free       = device_pool->freeBlocksNum();
    size_t       reclaimed          = 0;
    bool         eviction_triggered = false;
    while (reclaimed < num_blocks) {
        if (!evictor_.dropLocked(location->group_set_id, Tier::DEVICE, true)) {
            break;
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
    RTP_LLM_LOG_DEBUG("tag=%.*s group_set[%zu] reclaimed %zu/%zu device blocks",
                      static_cast<int>(group_tag.size()),
                      group_tag.data(),
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(std::string_view                      group_tag,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_tag, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<BlockTreePoolMetricsSnapshot> BlockTreeCache::poolMetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_reporter_->collectPoolMetricsSnapshots(tree_->groupSets());
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_->collectEvictableMetricsSnapshots(tree_->groupSets(), evictor_);
    }
    metrics_reporter_->reportEvictableCandidateCount(snapshots);
    metrics_reporter_->reportQueueBacklog(task_pool_->queueSizes(), "business");
    metrics_reporter_->reportQueueBacklog(transfer_dispatcher_->queueSizes(), "transfer");
}

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                       const std::vector<std::string>& required_group_tags) {
    std::lock_guard<std::mutex> lock(mutex_);
    tree_->setEventPublisher(std::move(publisher), required_group_tags);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = tree_->keyVersion();
    if (latest_version != snapshot.version) {
        snapshot.keys = tree_->cacheKeys();
    }
    return snapshot;
}

bool BlockTreeCache::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return loader_.abortPendingLoad(context);
}

void BlockTreeCache::onWorkflowSettledLocked(bool check_watermark) {
    if (check_watermark) {
        checkWatermark();
    }
}

void BlockTreeCache::checkWatermark() {
    for (Tier tier : {Tier::DISK, Tier::HOST, Tier::DEVICE}) {
        const auto watermark = config_.watermarkForTier(tier);
        if (!config_.isTierEnabled(tier) || !watermark.enabled()) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark);
    }
}

}  // namespace rtp_llm
