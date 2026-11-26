#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>
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
                                           const std::function<bool(size_t)>& valid_prefix, RoutedExpertPages* route_pages) {
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (commit_fenced_) {
            return result;
        }
        result = loader_.matchLocked(cache_keys, valid_prefix, route_pages);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    (void)insert(cache_keys, resources, target_tier, false);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident, const RoutedExpertPages& route_pages) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (commit_fenced_) {
            return 0;
        }
        storage_write = storer_.storeLocked(cache_keys, resources, target_tier, is_resident, resident_prefix_length, route_pages);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

bool BlockTreeCache::clearAll(std::string* error_message) {
    auto fail = [&](const std::string& msg) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::clearAll failed: %s", msg.c_str());
        if (error_message != nullptr) {
            *error_message = msg;
        }
        return false;
    };
    if (!initialized_) {
        return fail("BlockTreeCache is not initialized");
    }
    if (storage_backend_) {
        return fail("external StorageBackend invalidation is not supported in live RL commit");
    }

    // 1. Install a reversible commit fence so no new match/insert/watermark work enters.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        commit_fenced_ = true;
        storer_.stopAdmissionLocked();
    }

    // 2. Abort any pending loads and drain business tasks + async transfers to stable idleness.
    loader_.suspendAndDrain();
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();

    bool drained = false;
    for (int round = 0; round < 8; ++round) {
        task_pool_->waitForIdle();
        transfer_dispatcher_->drainTransfers();
        task_pool_->waitForIdle();
        const auto biz_q  = task_pool_->queueSizes();
        const auto xfer_q = transfer_dispatcher_->queueSizes();
        if (biz_q.load == 0 && biz_q.background == 0 && biz_q.completion == 0 && xfer_q.load == 0
            && xfer_q.background == 0 && xfer_q.completion == 0) {
            drained = true;
            break;
        }
    }
    if (!drained) {
        return fail("cache task/transfer queues did not reach stable idle state");
    }

    // 3. Under mutex_, validate pools, walk populated nodes child-first, release CACHE refs,
    //    remove nodes via publication-aware BlockTree::removeNode, and assert empty baselines.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tree_->pendingNodeCount() != 0) {
            return fail("BlockTree still has pending detached nodes after drain: "
                        + std::to_string(tree_->pendingNodeCount()));
        }

        std::unordered_set<IBlockPool*>       all_pools;
        std::unordered_set<DeviceBlockPool*>  device_pools;
        for (const GroupSetPtr& group_set : tree_->groupSets()) {
            for (const auto& device_pool : group_set->devicePools()) {
                if (device_pool != nullptr) {
                    all_pools.insert(device_pool.get());
                    device_pools.insert(device_pool.get());
                }
            }
            if (group_set->hostPool() != nullptr) {
                all_pools.insert(group_set->hostPool().get());
            }
            if (group_set->diskPool() != nullptr) {
                all_pools.insert(group_set->diskPool().get());
            }
        }

        // Pre-clear safety: zero active requests, zero in-flight LOAD/EVICTION/STORE, zero FROZEN.
        for (DeviceBlockPool* pool : device_pools) {
            if (pool->referencedBlocksNum() != 0) {
                return fail("device pool " + pool->poolName() + " still has active request refs="
                            + std::to_string(pool->referencedBlocksNum()));
            }
        }
        for (IBlockPool* pool : all_pools) {
            for (BlockTreeRefType ref_type :
                 {BlockTreeRefType::LOAD, BlockTreeRefType::EVICTION, BlockTreeRefType::STORE, BlockTreeRefType::FROZEN}) {
                const size_t count = pool->referencedBlocksNum(ref_type);
                if (count != 0) {
                    return fail("pool " + pool->poolName() + " has non-zero ref type "
                                + std::to_string(static_cast<int>(ref_type)) + " count=" + std::to_string(count));
                }
            }
        }

        // Collect all non-root nodes in post-order (children before parents).
        std::vector<TreeNode*> post_order;
        post_order.reserve(tree_->size());
        std::vector<std::pair<TreeNode*, bool>> stack;
        for (const auto& [_, child] : tree_->root()->children) {
            stack.emplace_back(child, false);
        }
        while (!stack.empty()) {
            auto [node, visited] = stack.back();
            stack.pop_back();
            if (visited) {
                post_order.push_back(node);
            } else {
                stack.emplace_back(node, true);
                for (const auto& [_, child] : node->children) {
                    stack.emplace_back(child, false);
                }
            }
        }

        if (post_order.size() != tree_->size()) {
            return fail("BlockTree reachable node count (" + std::to_string(post_order.size())
                        + ") != node_pool size (" + std::to_string(tree_->size()) + ")");
        }

        for (TreeNode* node : post_order) {
            if (node->is_resident) {
                return fail("cannot invalidate resident system-prompt node key=" + std::to_string(node->cache_key));
            }
            for (const GroupSetResource& resource : node->group_set_resources) {
                if (resource.transfer_state != GroupSetTransferState::IDLE || resource.transfer_detached) {
                    return fail("BlockTree node key=" + std::to_string(node->cache_key)
                                + " is not IDLE after drain");
                }
            }
        }

        for (TreeNode* node : post_order) {
            evictor_.eraseNodeFromAllHeaps(node);
            tree_->releaseNode(node);
            if (!tree_->isRemovable(node)) {
                return fail("BlockTree node key=" + std::to_string(node->cache_key)
                            + " remained non-removable after releasing CACHE refs");
            }
            tree_->removeNode(node);
        }

        // Postconditions: tree empty, heaps empty, publication snapshot empty, every pool at empty baseline.
        if (tree_->size() != 0 || tree_->pendingNodeCount() != 0 || !tree_->root()->children.empty()) {
            return fail("BlockTree not empty after clearAll: size=" + std::to_string(tree_->size())
                        + " pending=" + std::to_string(tree_->pendingNodeCount())
                        + " root_children=" + std::to_string(tree_->root()->children.size()));
        }
        const KVCacheSnapshot snapshot = tree_->logicalCacheSnapshot();
        if (!snapshot.block_keys.empty()) {
            return fail("published logicalCacheSnapshot is non-empty after clearAll: "
                        + std::to_string(snapshot.block_keys.size()));
        }
        const CandidateStats candidates = evictor_.candidateStats();
        if (candidates.device_candidates != 0 || candidates.host_candidates != 0 || candidates.disk_candidates != 0) {
            return fail("evictor heaps non-empty after clearAll");
        }
        for (IBlockPool* pool : all_pools) {
            if (pool->freeBlocksNum() != pool->totalBlocksNum() || pool->activeBlocksNum() != 0
                || pool->referencedBlocksNum(BlockTreeRefType::CACHE) != 0
                || pool->referencedBlocksNum(BlockTreeRefType::LOAD) != 0
                || pool->referencedBlocksNum(BlockTreeRefType::EVICTION) != 0
                || pool->referencedBlocksNum(BlockTreeRefType::STORE) != 0
                || pool->referencedBlocksNum(BlockTreeRefType::FROZEN) != 0) {
                std::ostringstream oss;
                oss << "pool " << pool->poolName() << " did not return to empty baseline after clearAll: free="
                    << pool->freeBlocksNum() << " total=" << pool->totalBlocksNum()
                    << " active=" << pool->activeBlocksNum()
                    << " cache_refs=" << pool->referencedBlocksNum(BlockTreeRefType::CACHE);
                return fail(oss.str());
            }
        }
        for (DeviceBlockPool* pool : device_pools) {
            if (pool->referencedBlocksNum() != 0) {
                return fail("device pool " + pool->poolName() + " has non-zero request refs after clearAll");
            }
        }

        RTP_LLM_LOG_INFO("BlockTreeCache::clearAll cleared %zu nodes across %zu distinct pools; key_version=%ld",
                         post_order.size(),
                         all_pools.size(),
                         tree_->keyVersion());

        // 4. All postconditions verified: reopen the reversible fence.
        storer_.resumeAdmissionLocked();
        commit_fenced_ = false;
    }

    loader_.resumeAdmission();
    task_pool_->resumeAdmission();
    return true;
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
    if (commit_fenced_) {
        return;
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
