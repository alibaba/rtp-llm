#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <cassert>
#include <chrono>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeStorer::BlockTreeStorer(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             host_timeout_ms,
                                 int                             disk_timeout_ms,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    store_task_runner_(tree_->groupSets(), tree),
    mutex_(mutex),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)) {}

void BlockTreeStorer::stopAdmissionLocked() {
    stopping_.store(true);
}

StorageWriteTask BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier                                              target_tier,
                                              bool                                              is_resident,
                                              size_t&                                           resident_prefix_length,
                                              std::optional<int64_t>                            request_id,
                                              size_t                                            remote_reuse_begin,
                                              size_t                                            remote_reuse_end,
                                              bool                                              mock_memory_cache,
                                              bool                                              mock_disk_cache) {
    resident_prefix_length = 0;
    assert(!is_resident || target_tier == Tier::DEVICE);
    RTP_LLM_CHECK_WITH_INFO(target_tier == Tier::DEVICE || target_tier == Tier::HOST || target_tier == Tier::DISK,
                            "unsupported store target tier: %s",
                            tierName(target_tier));
    if (target_tier == Tier::DEVICE) {
        publishDeviceLocked(cache_keys, resources, is_resident, resident_prefix_length);
    } else {
        submitLowerTierLocked(cache_keys, resources, target_tier, mock_memory_cache, mock_disk_cache);
        return {};
    }
    return storage_backend_ ? storage_backend_->prepareWrite(makeStorageRequest(
                                  cache_keys, resources, request_id, remote_reuse_begin, remote_reuse_end)) :
                              StorageWriteTask{};
}

void BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources,
                                          bool                                              is_resident,
                                          size_t&                                           resident_prefix_length) {
    const BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources, false, is_resident);
    if (insert_result.hard_stop) {
        metrics_reporter_.reportInsertHardStop();
    }
    const bool resources_inserted = !insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty();
    if (resources_inserted || !insert_result.newly_resident_nodes.empty()) {
        evictor_.onInserted(insert_result);
    }
    if (resources_inserted) {
        settled_(true);
    }
    resident_prefix_length = insert_result.resident_prefix_length;
}

StorageRequest BlockTreeStorer::makeStorageRequest(const CacheKeysType&                              cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources,
                                                   const std::optional<int64_t>&                     request_id,
                                                   size_t                                            remote_reuse_begin,
                                                   size_t remote_reuse_end) const {
    RTP_LLM_CHECK(remote_reuse_begin <= remote_reuse_end && remote_reuse_end <= cache_keys.size());
    const size_t key_count = cache_keys.size() - (remote_reuse_end - remote_reuse_begin);
    auto         keys      = std::make_shared<CacheKeysType>();
    keys->reserve(key_count);
    keys->insert(keys->end(), cache_keys.begin(), cache_keys.begin() + remote_reuse_begin);
    keys->insert(keys->end(), cache_keys.begin() + remote_reuse_end, cache_keys.end());
    StorageRequest request{std::move(keys), std::vector<std::vector<StorageBlockHandle>>(key_count)};
    request.request_id = request_id;
    size_t output      = 0;
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        if (key_index >= remote_reuse_begin && key_index < remote_reuse_end) {
            continue;
        }
        auto& key_handles = request.handles[output++];
        for (size_t group_set = 0; group_set < tree_->groupSets().size(); ++group_set) {
            const auto& resource = resources[key_index][group_set];
            if (!resource.hasCompleteDeviceValue()) {
                continue;
            }
            const auto& group = *tree_->groupSets()[group_set];
            for (size_t member = 0; member < group.groupTags().size(); ++member) {
                key_handles.push_back({group.groupTags()[member], resource.device_blocks[member]});
            }
        }
    }
    return request;
}

void BlockTreeStorer::submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            Tier                                              target_tier,
                                            bool                                              mock_memory_cache,
                                            bool                                              mock_disk_cache) {
    const auto rejected = [&] {
        if (mock_memory_cache || mock_disk_cache) {
            metrics_reporter_.recordLocalStoreRejection(target_tier);
        }
    };
    if (stopping_.load()) {
        rejected();
        return;
    }

    StoreTaskPtr task = std::make_shared<StoreTask>(
        target_tier,
        cache_keys,
        std::chrono::milliseconds(target_tier == Tier::DISK ? disk_timeout_ms_ : host_timeout_ms_));
    task->mock_memory_cache = mock_memory_cache;
    task->mock_disk_cache   = mock_disk_cache;

    block_tree_cache_detail::ScopeRollback prepare_guard([this, &task]() { settleLocked(*task, /*publish=*/false); });

    if (!store_task_runner_.prepareTask(*task, resources)) {
        rejected();
        return;
    }
    auto on_timeout = [this, task]() {
        RTP_LLM_LOG_WARNING("store expired in business queue, target=%s blocks=%zu",
                            tierName(task->target_tier),
                            task->descriptors().size());
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "store business queue timeout"));
    };
    const auto deadline = task->transfer_task.deadline();
    if (!task_pool_->submit(
            BlockTreeTaskClass::BACKGROUND,
            [this, task]() { runStoreTask(task); },
            deadline,
            std::move(on_timeout))) {
        RTP_LLM_LOG_WARNING("store aborted: business task submission rejected, target=%s blocks=%zu",
                            tierName(target_tier),
                            task->descriptors().size());
        rejected();
        return;
    }
    // A synchronous timeout or immediate physical callback may already have
    // recorded admission. The task makes both entry points idempotent.
    task->recordMockSubmission(metrics_reporter_);
    prepare_guard.dismiss();
}

void BlockTreeStorer::runStoreTask(const StoreTaskPtr& task) {
    if (stopping_.load()) {
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store stopped before transfer"));
        return;
    }

    try {
        store_task_runner_.runTransfer(
            task, *transfer_dispatcher_, metrics_reporter_, [this, task](TransferResult result) {
                scheduleStoreSettlement(task, std::move(result));
            });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("store copy threw: %s", error.what());
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        RTP_LLM_LOG_ERROR("store copy threw an unknown exception");
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store copy exception"));
    }
}

void BlockTreeStorer::scheduleStoreSettlement(const StoreTaskPtr& task, TransferResult result) {
    // Queue timeout/shutdown/submission exceptions have no physical callback.
    // The task guard prevents a physical outcome being counted again here.
    task->recordMockTransferOutcome(metrics_reporter_, result.ok());
    auto settle = [this, task, success = result.ok()] { settleTask(*task, success); };
    if (!task_pool_->submitCompletion(settle)) {
        RTP_LLM_LOG_WARNING("store completion queue is closed; settling inline");
        settle();
    }
}

void BlockTreeStorer::settleTask(const StoreTask& task, bool copy_success) {
    bool   stopping = false;
    size_t accepted = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping = stopping_.load();
        accepted = settleLocked(task, copy_success && !stopping);
    }

    if (stopping) {
        RTP_LLM_LOG_INFO("store rolled back during shutdown, target=%s blocks=%zu",
                         tierName(task.target_tier),
                         task.descriptors().size());
    } else if (!copy_success) {
        RTP_LLM_LOG_WARNING("store copy failed, target=%s blocks=%zu; tree unchanged",
                            tierName(task.target_tier),
                            task.descriptors().size());
    } else {
        RTP_LLM_LOG_DEBUG("store published target=%s accepted=%zu duplicate=%zu",
                          tierName(task.target_tier),
                          accepted,
                          task.descriptors().size() - accepted);
    }
}

size_t BlockTreeStorer::settleLocked(const StoreTask& task, bool publish) {
    BlockTreeInsertResult insert_result;
    if (publish) {
        std::vector<std::vector<GroupSetResource>> resources(task.cache_keys.size(),
                                                             std::vector<GroupSetResource>(tree_->groupSets().size()));
        for (const TransferDescriptor& descriptor : task.descriptors()) {
            resources[descriptor.path_index][descriptor.group_set_id].setBlocks(
                task.target_tier, {descriptor.singleBlockAt(task.target_tier)});
        }
        insert_result = tree_->insertNode(task.cache_keys, resources, true, false);
        for (const auto& descriptor : task.descriptors()) {
            if (descriptor.path_index >= insert_result.path.size()) {
                continue;
            }
            const auto* node     = insert_result.path[descriptor.path_index];
            const auto& resource = node->group_set_resources[descriptor.group_set_id];
            if (!resource.transfer_detached && resource.hasTier(task.target_tier)
                && resource.getBlocks(task.target_tier) == descriptor.target_blocks) {
                tree_->recordPressurePublication(CacheTransferOperation::STORE, descriptor);
            }
        }
    }

    store_task_runner_.releaseTaskResources(task);

    if (insert_result.accepted_resource_count > 0) {
        evictor_.onInserted(insert_result);
    }
    if (!task.descriptors().empty()) {
        settled_(true);
    }
    return insert_result.accepted_resource_count;
}

}  // namespace rtp_llm
