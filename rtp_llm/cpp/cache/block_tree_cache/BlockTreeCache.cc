#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
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
#include "rtp_llm/cpp/utils/TimeUtil.h"

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
        [this](bool tree_data_mutated, bool check_watermark) {
            onWorkflowSettledLocked(tree_data_mutated, check_watermark);
        }),
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
            [this](bool tree_data_mutated, bool check_watermark) {
                onWorkflowSettledLocked(tree_data_mutated, check_watermark);
            }),
    storer_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.host_cache_sync_timeout_ms,
            config_.disk_cache_sync_timeout_ms,
            storage_backend_,
            [this](bool tree_data_mutated, bool check_watermark) {
                onWorkflowSettledLocked(tree_data_mutated, check_watermark);
            }) {}

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
    loader_.shutdown();
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

bool BlockTreeCache::executeTransfer(TransferTask task) {
    auto context = transfer_dispatcher_->executePerRank(std::move(task));
    context->waitDone();
    if (!context->success()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", context->errorInfo().ToString().c_str());
        return false;
    }
    return true;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType& cache_keys) {
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result = loader_.matchLocked(cache_keys);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

ErrorInfo BlockTreeCache::matchForExternalInsert(const CacheKeysType&  cache_keys,
                                                 size_t                required_prefix_blocks,
                                                 BlockTreeMatchResult& match_result) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  error = validateDevicePrefixLocked(cache_keys, required_prefix_blocks);
        if (error.hasError()) {
            return error;
        }
        // Take the prefix references before eviction can invalidate the admission result.
        match_result = loader_.matchLocked(cache_keys, /*allow_storage_backend_lookup=*/false);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(match_result.reuse_time_metrics_snapshots);
    return ErrorInfo::OkStatus();
}

ErrorInfo BlockTreeCache::validateDevicePrefixLocked(const CacheKeysType& cache_keys,
                                                     size_t               required_prefix_blocks) const {
    // Check that the request targets a valid DEVICE cache range.
    if (!config_.enable_device_cache || cache_keys.empty() || required_prefix_blocks > cache_keys.size()
        || tree_->groupSets().empty()) {
        return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                         "external insert requires a device cache and a valid key range");
    }

    // TODO(rym): need support all group types for external insert, currently only FULL group is supported.
    for (const auto& group : tree_->groupSets()) {
        // Check that every group uses the FULL layout supported by writeback.
        if (!group || group->groupType() != CacheGroupType::FULL) {
            return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "external insert currently requires FULL groups");
        }
    }
    const auto path = tree_->findNode(cache_keys);
    // Check that the complete required prefix still exists in the tree.
    if (path.size() < required_prefix_blocks) {
        return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted");
    }

    // Check that every prefix block of every group set is still resident on DEVICE and has a complete value.
    for (size_t i = 0; i < required_prefix_blocks; ++i) {
        for (size_t gid = 0; gid < tree_->groupSets().size(); ++gid) {
            const auto& resource = path[i]->group_set_resources[gid];
            if (!resource.isValidSteadyState() || resource.transfer_detached || !resource.hasCompleteDeviceValue()
                || resource.device_blocks.size() != tree_->groupSets()[gid]->devicePools().size()) {
                return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted");
            }
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo BlockTreeCache::insertExternalBlocks(const CacheKeysType&                              cache_keys,
                                               size_t                                            start_block,
                                               const std::vector<std::vector<GroupSetResource>>& resources,
                                               int64_t                                           deadline_ms) {
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Do not publish a writeback after its request deadline.
        if (currentTimeMs() >= deadline_ms) {
            return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "external insert expired before publication");
        }

        // Confirm that matched and locally loaded blocks are still published on DEVICE.
        const auto prefix_error = validateDevicePrefixLocked(cache_keys, start_block);
        if (prefix_error.hasError()) {
            return prefix_error;
        }

        // Let the normal BlockTree insert reuse concurrent hits and adopt only missing blocks.
        // Assume that any returned KV blocks already inserted into the BlockTree remain on DEVICE during this short
        // settle window; immediate demotion to HOST is unlikely, so writeback does not handle it specially.
        size_t resident_prefix_length = 0;
        storage_write = storer_.storeLocked(cache_keys, resources, Tier::DEVICE, false, resident_prefix_length);
    }
    // Persist the published blocks when a storage backend is configured.
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return ErrorInfo::OkStatus();
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    (void)insert(cache_keys, resources, target_tier, false);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storage_write = storer_.storeLocked(cache_keys, resources, target_tier, is_resident, resident_prefix_length);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

int BlockTreeCache::evictForGroup(size_t group_id, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_id);
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
    RTP_LLM_LOG_DEBUG("group_id=%zu group_set[%zu] reclaimed %zu/%zu device blocks",
                      group_id,
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(size_t                                group_id,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_id, matched_resources);
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

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<int>& required_group_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    tree_->setEventPublisher(std::move(publisher), required_group_ids);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->size());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty()) {
        const TreeNode* node = pending.back();
        pending.pop_back();
        const bool reusable = std::any_of(node->group_set_resources.begin(),
                                          node->group_set_resources.end(),
                                          [](const GroupSetResource& resource) { return !resource.is_empty(); });
        if (reusable) {
            snapshot.keys.push_back(node->cache_key);
        }
        for (const auto& [cache_key, child] : node->children) {
            (void)cache_key;
            if (child) {
                pending.push_back(child);
            }
        }
    }
    return snapshot;
}

bool BlockTreeCache::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return loader_.abortPendingLoad(context);
}

void BlockTreeCache::onWorkflowSettledLocked(bool tree_data_mutated, bool check_watermark) {
    if (tree_data_mutated) {
        ++mutation_version_;
    }
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
