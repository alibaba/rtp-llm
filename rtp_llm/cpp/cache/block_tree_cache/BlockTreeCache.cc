#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>               tree,
                               BlockTreeCacheConfig                     config,
                               std::shared_ptr<StorageBackend>          storage_backend,
                               std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>       task_pool):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    evictor_(
        tree_.get(),
        config_.device_eviction_policy,
        config_.host_eviction_policy,
        config_.disk_eviction_policy,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        metrics_reporter_,
        mutex_,
        config_.host_cache_sync_timeout_ms,
        config_.disk_cache_sync_timeout_ms,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool tree_data_mutated, bool check_watermark) {
            onWorkflowSettledLocked(tree_data_mutated, check_watermark);
        }),
    loader_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            metrics_reporter_,
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
            metrics_reporter_,
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
    initialized_ = true;
    return true;
}

BlockTreeCache::~BlockTreeCache() {
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
    task_pool_->waitForIdle();
    task_pool_->shutdown();
    RTP_LLM_LOG_INFO("destroyed");
}

bool BlockTreeCache::executeTransfer(const std::vector<TransferDescriptor>& descriptors) {
    auto context = transfer_dispatcher_->executePerRank(descriptors);
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
    metrics_reporter_.reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storage_write = storer_.storeLocked(cache_keys, resources, target_tier);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
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

    const size_t initial_free = device_pool->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        if (!evictor_.evictLocked(location->group_set_id, Tier::DEVICE, /*force_drop=*/true)) {
            break;
        }
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
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
    return metrics_reporter_.collectPoolMetricsSnapshots(tree_->groupSets(), evictor_);
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_.collectEvictableMetricsSnapshots(tree_->groupSets(), evictor_);
    }
    metrics_reporter_.reportEvictableCandidateCount(snapshots);
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;
    if (limit == 0) {
        return snapshot;
    }

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->size());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty() && snapshot.keys.size() < limit) {
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
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        const auto watermark = config_.watermarkForTier(tier);
        if (!config_.isTierEnabled(tier) || watermark.ratio <= 0.0) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark.ratio);
    }
}

}  // namespace rtp_llm
