#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

struct BlockTreeCache::CapacityCallbackState {
    explicit CapacityCallbackState(BlockTreeCache* cache): owner(cache) {}

    std::mutex      mutex;
    BlockTreeCache* owner;
};

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                               BlockTreeCacheConfig                           config,
                               std::shared_ptr<StorageBackend>                storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>             task_pool,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                               std::shared_ptr<BlockTreeReclaimer>            reclaimer):
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
        config_.disk_cache_evict_timeout_ms,
        config_.max_descriptors_per_transfer_batch,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); },
        config_.disk_cache_evict_max_pending_bytes),
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
    storer_(tree_.get(), evictor_, *metrics_reporter_, storage_backend_, [this](bool check_watermark) {
        onWorkflowSettledLocked(check_watermark);
    }),
    reclaimer_(std::move(reclaimer)),
    capacity_callback_state_(std::make_shared<CapacityCallbackState>(this)) {
    metrics_reporter_->initEvictionMetricGroupTypes(tree_->groupSets());
    if (storage_backend_) {
        storage_backend_->setStageMetricsReporter(metrics_reporter_);
    }
}

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
        auto scanner = std::make_unique<FullPrefixInvariantScanner>(
            *tree_, mutex_, config_.full_prefix_scan_interval_ms, metrics_reporter_.get());
        if (scanner->start()) {
            full_prefix_scanner_ = std::move(scanner);
            RTP_LLM_LOG_INFO("FULL prefix invariant scanner enabled, interval_ms=%ld",
                             config_.full_prefix_scan_interval_ms);
        }
    }
    if (!reclaimer_->start())
        return false;
    initialized_ = true;
    return true;
}

void BlockTreeCache::stopBackgroundTasks() {
    reclaimer_->stop();
}

BlockTreeCache::~BlockTreeCache() {
    stopping_.store(true);
    {
        std::lock_guard<std::mutex> lock(capacity_callback_state_->mutex);
        capacity_callback_state_->owner = nullptr;
    }
    stopBackgroundTasks();
    full_prefix_scanner_.reset();
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    shutdownLoads();
    if (storage_backend_) {
        storage_backend_->shutdown();
    }
    RTP_LLM_LOG_INFO("load tickets closed, draining cache tasks...");
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();
    task_pool_->waitForIdle();
    // Task-body idleness does not imply asynchronous RPC/transfer completion.
    // Keep dispatcher and cache state alive until transfer callbacks return,
    // then drain the settlement tasks those callbacks submitted.
    transfer_dispatcher_->shutdown();
    task_pool_->waitForIdle();
    transfer_dispatcher_.reset();
    task_pool_.reset();
    RTP_LLM_LOG_INFO("destroyed");
}

void BlockTreeCache::shutdownLoads() {
    loader_.shutdown();
}

std::function<void()> BlockTreeCache::capacityChangeCallback() const {
    return [state = capacity_callback_state_] {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->owner) {
            state->owner->onCapacityChanged();
        }
    };
}

void BlockTreeCache::onCapacityChanged() {
    if (stopping_.load() || watermark_check_pending_.exchange(true)) {
        return;
    }
    // Capacity callbacks can run under mutex_ while tree references are released.
    // Queue the check and coalesce wakeups instead of reacquiring that mutex inline.
    if (!task_pool_->submitCompletion([this] {
            watermark_check_pending_.store(false);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stopping_.load()) {
                checkWatermark();
            }
        })) {
        watermark_check_pending_.store(false);
    }
}

bool BlockTreeCache::executeFunction(BlockTreeCache*          cache,
                                     int                      rank,
                                     const FunctionRequestPB& request,
                                     FunctionResponsePB&      response,
                                     std::chrono::steady_clock::time_point deadline,
                                     bool is_pp) {
    switch (request.request_case()) {
        case FunctionRequestPB::kCompletionQuery:
        case FunctionRequestPB::kCompletionForget:
            return BlockTreeReclaimer::executeFunction(
                cache ? cache->reclaimer_.get() : nullptr, rank, request, response, deadline);
        case FunctionRequestPB::kRemoteRequest:
            if (!cache || !cache->storage_backend_) {
                RTP_LLM_LOG_WARNING("remote storage backend is not initialized");
                return false;
            }
            return cache->storage_backend_->executeRemoteRequest(request.remote_request(), response);
        case FunctionRequestPB::kMemRequest: {
            if (!cache) {
                RTP_LLM_LOG_WARNING("block tree cache is not initialized");
                return false;
            }
            RTP_LLM_LOG_INFO("transfer_lifecycle event=rpc_execute_enter trace=%lu request=%ld rank=%ld timeout_ms=%ld",
                             request.mem_request().transfer_trace_id(),
                             request.mem_request().transfer_request_id(),
                             static_cast<int64_t>(rank),
                             request.mem_request().timeout_ms());
            std::vector<size_t> wire_item_indices;
            auto& reply = *response.mutable_mem_response();
            auto task = BlockTransferRequestConverter::decodeTransferTask(
                request.mem_request(), cache->groupSets(), reply, is_pp ? &wire_item_indices : nullptr);
            if (!task) {
                return true;
            }
            auto status = cache->executeTransfer(std::move(*task));
            const size_t wire_item_count = static_cast<size_t>(request.mem_request().copy_items_size());
            if (is_pp && wire_item_indices.size() != wire_item_count) {
                status.expandBlockErrors(wire_item_indices.size());
                if (status.block_errors.size() != wire_item_indices.size()) {
                    status = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "invalid local block transfer result"),
                              CacheCopyError::RPC_FAILED};
                    status.expandBlockErrors(wire_item_indices.size());
                }
                std::vector<CacheCopyError> wire_errors(wire_item_count, CacheCopyError::NONE);
                for (size_t index = 0; index < wire_item_indices.size(); ++index) {
                    wire_errors[wire_item_indices[index]] = status.blockError(index);
                }
                status.block_errors = std::move(wire_errors);
            }
            BlockTransferRequestConverter::encodeTransferResult(reply, status);
            RTP_LLM_LOG_INFO("transfer_lifecycle event=rpc_execute_exit trace=%lu rank=%ld code=%ld buffer_safe=%d",
                             request.mem_request().transfer_trace_id(),
                             static_cast<int64_t>(rank),
                             static_cast<int64_t>(status.status.code()),
                             status.buffer_safe);
            return true;
        }
        default:
            RTP_LLM_LOG_WARNING("unsupported cache RPC request type");
            return false;
    }
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
                                           bool                               enable_storage_backend,
                                           std::optional<int64_t>             request_id,
                                           bool                               skip_linear) {
    BlockTreeMatchResult result;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "match");
        result = loader_.matchLocked(cache_keys, valid_prefix, enable_storage_backend, request_id, skip_linear);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics);
    return result;
}

ErrorInfo BlockTreeCache::matchForExternalInsert(const CacheKeysType&  cache_keys,
                                                 size_t                required_prefix_blocks,
                                                 BlockTreeMatchResult& match_result,
                                                 bool                  require_device_prefix) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  error = validateWritebackPrefixLocked(
            cache_keys, tree_->findNode(cache_keys), required_prefix_blocks, require_device_prefix);
        if (error.hasError()) {
            return error;
        }
        // Take the prefix references before eviction can invalidate the admission result.
        match_result = loader_.matchLocked(cache_keys, {}, /*enable_storage_backend=*/false);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(match_result.reuse_time_metrics);
    return ErrorInfo::OkStatus();
}

ErrorInfo BlockTreeCache::validateWritebackPrefixLocked(const CacheKeysType&          cache_keys,
                                                        const std::vector<TreeNode*>& path,
                                                        size_t                        required_prefix_blocks,
                                                        bool                          require_device_prefix) const {
    // Check that the request targets a valid DEVICE cache range.
    if (!config_.enable_device_cache || cache_keys.empty() || required_prefix_blocks > cache_keys.size()
        || tree_->groupSets().empty()) {
        return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                         "external insert requires a device cache and a valid key range");
    }

    // Check that the complete required prefix still exists in the tree.
    if (path.size() < required_prefix_blocks) {
        return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted");
    }

    // Use each group's normal reuse range: FULL prefix, SWA window, or LINEAR endpoint state.
    for (size_t gid = 0; gid < tree_->groupSets().size(); ++gid) {
        const auto reuse_blocks =
            std::min(tree_->groupSets()[gid]->computeReuseBlockCount(required_prefix_blocks), required_prefix_blocks);
        for (size_t i = required_prefix_blocks - reuse_blocks; i < required_prefix_blocks; ++i) {
            const auto& resource     = path[i]->group_set_resources[gid];
            const bool  device_ready = resource.hasCompleteDeviceValue()
                                      && resource.device_blocks.size() == tree_->groupSets()[gid]->devicePools().size();
            const bool loadable =
                !require_device_prefix && (resource.hasTier(Tier::HOST) || resource.hasTier(Tier::DISK));
            if (!resource.isValidSteadyState() || resource.transfer_detached || (!device_ready && !loadable)) {
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
        auto       path         = tree_->findNode(cache_keys);
        const auto prefix_error = validateWritebackPrefixLocked(cache_keys, path, start_block);
        if (prefix_error.hasError()) {
            return prefix_error;
        }

        const auto& groups = tree_->groupSets();
        if (resources.size() != cache_keys.size()
            || std::any_of(
                resources.begin(), resources.end(), [&](const auto& row) { return row.size() != groups.size(); })) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "external insert resource shape does not match target");
        }

        // A concurrent Prefill or eviction can claim suffix slots after admission.
        // Validate the final reuse window of each group before publishing any blocks.
        for (size_t gid = 0; gid < groups.size(); ++gid) {
            const auto&  group        = groups[gid];
            const size_t reuse_blocks = std::min(group->computeReuseBlockCount(cache_keys.size()), cache_keys.size());
            const size_t reuse_begin  = cache_keys.size() - reuse_blocks;
            for (size_t i = start_block; i < cache_keys.size(); ++i) {
                const GroupSetResource* existing = i < path.size() ? &path[i]->group_set_resources[gid] : nullptr;
                if (existing && (!existing->isValidSteadyState() || existing->transfer_detached)) {
                    return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                                     "writeback publication incomplete: existing suffix unavailable");
                }
                if (i >= reuse_begin) {
                    // Writeback can adopt DEVICE data or fill an empty slot.
                    const auto& value = existing && !existing->is_removable() ? *existing : resources[i][gid];
                    if (!value.hasCompleteDeviceValue() || value.device_blocks.size() != group->devicePools().size()) {
                        return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                                         "writeback publication incomplete: final reuse endpoint unavailable");
                    }
                }
            }
        }

        storer_.publishDeviceLocked(cache_keys, resources, /*is_resident=*/false, /*is_writeback=*/true);
        path = tree_->findNode(cache_keys);
        if (validateWritebackPrefixLocked(cache_keys, path, cache_keys.size()).hasError()) {
            return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                             "writeback publication incomplete: final reuse endpoint unavailable");
        }
        // The transfer only supplies the suffix. Pin the actual published blocks for every remote key.
        storage_write = storer_.preparePublishedWriteLocked(cache_keys, path);
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
                              bool                                              is_resident,
                              std::optional<int64_t>                            request_id,
                              size_t                                            remote_reuse_begin,
                              size_t                                            remote_reuse_end) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "insert");
        storage_write = storer_.storeLocked(cache_keys,
                                            resources,
                                            target_tier,
                                            is_resident,
                                            resident_prefix_length,
                                            request_id,
                                            remote_reuse_begin,
                                            remote_reuse_end);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

int BlockTreeCache::evictForGroup(std::string_view group_tag, size_t num_blocks) {
    CacheLockGuard lock(mutex_,
                        "force_drop",
                        CacheStageContext::forLock(metrics_reporter_.get(), "force_drop", Tier::DEVICE, Tier::NONE));
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return 0;
    }
    const size_t reclaimed =
        evictor_.reclaimDeviceBlocksLocked(location->group_set_id, location->member_group_id, num_blocks);
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
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    CacheStats     stats;
    stats.tree_node_count           = tree_->size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<CachePoolMetrics> BlockTreeCache::poolMetrics() const {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    return metrics_reporter_->collectPoolMetrics(tree_->groupSets());
}

void BlockTreeCache::reportMetrics() const {
    EvictionVictimMetrics metrics_data;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
        metrics_data = evictor_.getAndResetEvictionVictimMetrics();
    }
    metrics_reporter_->reportEvictionMetrics(metrics_data);
    metrics_reporter_->reportQueueBacklog(task_pool_->queueSizes(), "business");
    metrics_reporter_->reportQueueBacklog(transfer_dispatcher_->queueSizes(), "transfer");
    if (storage_backend_)
        metrics_reporter_->reportQueueBacklog(storage_backend_->queueSizes(), "remote");
}

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                       const std::vector<std::string>& required_group_tags) {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    tree_->setEventPublisher(std::move(publisher), required_group_tags);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "snapshot");
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(int64_t latest_version) const {
    CacheLockGuard       lock(mutex_, *metrics_reporter_, "snapshot");
    BlockTreeKeySnapshot snapshot;
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
        if (!config_.isTierEnabled(tier)) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark);
    }
}

}  // namespace rtp_llm
