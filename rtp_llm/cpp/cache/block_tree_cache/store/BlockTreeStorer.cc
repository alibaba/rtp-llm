#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <algorithm>
#include <chrono>
#include <cassert>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

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
    store_task_runner_(tree_->groupSets()),
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
                                              size_t& resident_prefix_length,
                                              const std::vector<std::vector<int>>* token_segments,
                                              const std::vector<size_t>* token_ends,
                                              bool internal_fork,
                                              bool* ready_after_insert) {
    resident_prefix_length = 0;
    assert(!is_resident || target_tier == Tier::DEVICE);
    RTP_LLM_CHECK_WITH_INFO(target_tier == Tier::DEVICE || target_tier == Tier::HOST || target_tier == Tier::DISK,
                            "unsupported store target tier: %s",
                            tierName(target_tier));
    if (target_tier == Tier::DEVICE) {
        publishDeviceLocked(cache_keys,
                            resources,
                            is_resident,
                            resident_prefix_length,
                            token_segments,
                            token_ends,
                            internal_fork,
                            ready_after_insert);
    } else {
        submitLowerTierLocked(cache_keys, resources, target_tier);
        return {};
    }
    return storage_backend_ && !internal_fork ?
               storage_backend_->prepareWrite(makeStorageRequest(cache_keys, resources, token_ends)) :
                              StorageWriteTask{};
}

void BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources,
                                          bool                                              is_resident,
                                          size_t&                                           resident_prefix_length,
                                          const std::vector<std::vector<int>>*              token_segments,
                                          const std::vector<size_t>*                         token_ends,
                                          bool internal_fork,
                                          bool* ready_after_insert) {
    const BlockTreeInsertResult insert_result =
        tree_->insertNode(cache_keys, resources, false, is_resident, token_segments, token_ends);
    std::optional<bool> fork_ready;
    if (internal_fork) {
        const auto path = tree_->findNode(cache_keys);
        fork_ready = token_ends != nullptr && path.size() == cache_keys.size()
                     && path.back()->token_end == token_ends->back()
                     && std::all_of(tree_->groupSets().begin(), tree_->groupSets().end(), [&](const auto& group) {
                            return group->hasDeviceCoverage(path, token_ends->back());
                        });
        if (ready_after_insert != nullptr) {
            *ready_after_insert = *fork_ready;
        }
    }
    const bool tree_data_mutated = !insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty();
    if (tree_data_mutated || !insert_result.newly_resident_nodes.empty()) {
        evictor_.onInserted(insert_result);
    }
    if (tree_data_mutated) {
        settled_(true);
    }
    resident_prefix_length = insert_result.resident_prefix_length;
}

StorageRequest BlockTreeStorer::makeStorageRequest(const CacheKeysType&                              cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources,
                                                   const std::vector<size_t>* token_ends) const {
    auto keys = std::make_shared<CacheKeysType>();
    StorageRequest request{keys, {}};
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        // The existing backend protocol only names complete cumulative keys.
        if (token_ends != nullptr && (*token_ends)[key_index] % tree_->reuseUnitTokens() != 0) {
            continue;
        }
        keys->push_back(cache_keys[key_index]);
        auto& key_handles = request.handles.emplace_back();
        for (size_t group_set = 0; group_set < tree_->groupSets().size(); ++group_set) {
            if (tree_->groupSets()[group_set]->localOnly()) {
                continue;
            }
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
                                            Tier                                              target_tier) {
    if (stopping_.load()) {
        return;
    }

    StoreTaskPtr task = std::make_shared<StoreTask>(
        target_tier,
        cache_keys,
        std::chrono::milliseconds(target_tier == Tier::DISK ? disk_timeout_ms_ : host_timeout_ms_));

    block_tree_cache_detail::ScopeRollback prepare_guard(
        [this, &task]() { settleLocked(*task, /*publish=*/false); });

    if (!store_task_runner_.prepareTask(*task, resources)) {
        return;
    }
    const int64_t queue_begin = currentTimeUs();
    auto          on_timeout  = [this, task, queue_begin]() {
        metrics_reporter_.reportQueueWaitMetric(false,
                                                "business",
                                                cacheTransferOperationName(CacheTransferOperation::STORE),
                                                Tier::NONE,
                                                Tier::NONE,
                                                currentTimeUs() - queue_begin);
        RTP_LLM_LOG_WARNING("store expired in business queue, target=%s blocks=%zu",
                            tierName(task->target_tier),
                            task->descriptors().size());
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "store business queue timeout"));
    };
    const auto deadline = task->transfer_task.deadline();
    if (!task_pool_->submit(
            BlockTreeTaskClass::BACKGROUND,
            [this, task, queue_begin]() {
                metrics_reporter_.reportQueueWaitMetric(false,
                                                        "business",
                                                        cacheTransferOperationName(CacheTransferOperation::STORE),
                                                        Tier::NONE,
                                                        Tier::NONE,
                                                        currentTimeUs() - queue_begin);
                runStoreTask(task);
            },
            deadline,
            std::move(on_timeout))) {
        RTP_LLM_LOG_WARNING("store aborted: business task submission rejected, target=%s blocks=%zu",
                            tierName(target_tier),
                            task->descriptors().size());
        return;
    }
    prepare_guard.dismiss();
}

void BlockTreeStorer::runStoreTask(const StoreTaskPtr& task) {
    if (stopping_.load()) {
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store stopped before transfer"));
        return;
    }

    try {
        store_task_runner_.runTransfer(task, *transfer_dispatcher_, metrics_reporter_, [this, task](ErrorInfo error) {
            scheduleStoreSettlement(task, std::move(error));
        });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("store copy threw: %s", error.what());
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        RTP_LLM_LOG_ERROR("store copy threw an unknown exception");
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store copy exception"));
    }
}

void BlockTreeStorer::scheduleStoreSettlement(const StoreTaskPtr& task, ErrorInfo error) {
    if (!task->markFinished()) {
        return;
    }
    auto          settle      = [this, task, error = std::move(error)]() mutable { settleTask(*task, error.ok()); };
    const int64_t queue_begin = currentTimeUs();
    if (!task_pool_->submitCompletion([this, settle, queue_begin]() mutable {
            metrics_reporter_.reportQueueWaitMetric(true,
                                                    "business",
                                                    cacheTransferOperationName(CacheTransferOperation::STORE),
                                                    Tier::NONE,
                                                    Tier::NONE,
                                                    currentTimeUs() - queue_begin);
            settle();
        })) {
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
        metrics_reporter_.reportStorePublish(task.target_tier, accepted, task.descriptors().size() - accepted);
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
