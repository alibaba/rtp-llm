#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

struct BlockTransferDispatcher::CompletionTicket {
    explicit CompletionTicket(const BlockTransferDispatcher* owner): owner_(owner) {
        owner_->registerTransfer();
    }

    ~CompletionTicket() noexcept {
        owner_->retireTransfer();
    }

    CompletionTicket(const CompletionTicket&)            = delete;
    CompletionTicket& operator=(const CompletionTicket&) = delete;

private:
    const BlockTransferDispatcher* owner_;
};

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t max_descriptors_per_batch,
                                                 std::shared_ptr<StorageBackend> local_kvcm_backend):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
    if (local_kvcm_backend && local_kvcm_backend->supportsLocalCompletionQueries()) {
        local_kvcm_backend_ = std::move(local_kvcm_backend);
    }
    if (multi_rank_engine_ || local_kvcm_backend_) {
        frozen_thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lock(frozen_stop_mutex_);
            while (!frozen_stop_cv_.wait_for(lock, std::chrono::seconds(1), [this] { return frozen_stopping_; })) {
                lock.unlock();
                try {
                    pollFrozenTasks();
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_WARNING("frozen task polling failed: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_WARNING("frozen task polling failed with unknown exception");
                }
                lock.lock();
            }
        });
    }
}

BlockTransferDispatcher::~BlockTransferDispatcher() {
    stopFrozenPolling();
    shutdown();
}

void BlockTransferDispatcher::stopFrozenPolling() {
    {
        std::lock_guard<std::mutex> lock(frozen_stop_mutex_);
        frozen_stopping_ = true;
    }
    frozen_stop_cv_.notify_all();
    if (frozen_thread_.joinable()) frozen_thread_.join();
}

void BlockTransferDispatcher::pollFrozenTasks() const {
    if (multi_rank_engine_) {
        multi_rank_engine_->forgetFinishedTasks();
        multi_rank_engine_->forgetFinishedKVCMOperations();
        pollFrozenTasksForKind(FrozenTask::QueryKind::TRANSFER,
                               [this](const auto& ids) { return multi_rank_engine_->queryTasks(ids); });
        pollFrozenTasksForKind(FrozenTask::QueryKind::KVCM,
                               [this](const auto& ids) { return multi_rank_engine_->queryKVCMOperations(ids); });
    } else if (local_kvcm_backend_) {
        forgetFinishedLocalKVCMOperations();
        pollFrozenTasksForKind(FrozenTask::QueryKind::KVCM,
                               [this](const auto& ids) { return local_kvcm_backend_->queryLocalKVCMOperations(ids); });
    }
}

void BlockTransferDispatcher::pollFrozenTasksWithQuery(const QueryFrozenTasks& query) const {
    pollFrozenTasksForKind(FrozenTask::QueryKind::TRANSFER, query);
}

void BlockTransferDispatcher::pollFrozenTasksForKind(FrozenTask::QueryKind query_kind,
                                                     const QueryFrozenTasks& query) const {
    std::unique_lock<std::mutex> polling(frozen_poll_mutex_, std::try_to_lock);
    if (!polling.owns_lock()) return;
    std::unordered_set<std::string> unique_ids;
    {
        std::lock_guard<std::mutex> lock(frozen_mutex_);
        for (const auto& task : frozen_tasks_)
            if (!task.finalFrozen && task.query_kind == query_kind)
                unique_ids.insert(task.recovery_task_ids.begin(), task.recovery_task_ids.end());
    }
    if (unique_ids.empty()) return;
    const std::vector<std::string> ids(unique_ids.begin(), unique_ids.end());
    const auto states = query(ids);  // No frozen-list lock or borrowed task references across RPC.
    if (states.size() != ids.size()) return;
    std::unordered_set<std::string> completed_ids;
    for (size_t i = 0; i < ids.size(); ++i)
        if (states[i] == TaskCompletion::FINISHED) completed_ids.insert(ids[i]);
    releaseCompletedFrozenTasks(completed_ids, query_kind);
}

void BlockTransferDispatcher::releaseCompletedFrozenTasks(
    const std::unordered_set<std::string>& completed_ids, FrozenTask::QueryKind query_kind) const {
    std::vector<FrozenTask> completed;
    {
        std::lock_guard<std::mutex> lock(frozen_mutex_);
        for (auto it = frozen_tasks_.begin(); it != frozen_tasks_.end();) {
            if (it->query_kind == query_kind && it->readyToRelease(completed_ids)) {
                completed.push_back(std::move(*it));
                it = frozen_tasks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Pool capacity callbacks must not run under frozen_mutex_.
    for (auto& task : completed) {
        if (!task.unfreeze(completed_ids)) continue;
        if (query_kind == FrozenTask::QueryKind::TRANSFER) {
            if (multi_rank_engine_) multi_rank_engine_->enqueueFinishedTasks(task.recovery_task_ids);
        } else {
            for (const auto& id : task.recovery_task_ids) enqueueFinishedKVCMOperation(id);
        }
    }
}

void BlockTransferDispatcher::enqueueFinishedKVCMOperation(const std::string& operation_id) const {
    if (multi_rank_engine_) {
        multi_rank_engine_->enqueueFinishedKVCMOperations({operation_id});
    } else if (local_kvcm_backend_) {
        std::lock_guard<std::mutex> lock(local_kvcm_cleanup_mutex_);
        local_kvcm_finished_ids_.insert(operation_id);
    }
}

void BlockTransferDispatcher::forgetFinishedLocalKVCMOperations() const {
    std::unique_lock<std::mutex> polling(local_kvcm_cleanup_poll_mutex_, std::try_to_lock);
    if (!polling.owns_lock()) return;
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(local_kvcm_cleanup_mutex_);
        constexpr size_t kMaxCleanupBatch = 1024;
        for (const auto& id : local_kvcm_finished_ids_) {
            ids.push_back(id);
            if (ids.size() == kMaxCleanupBatch) break;
        }
    }
    if (ids.empty() || !local_kvcm_backend_->forgetLocalKVCMOperations(ids)) return;
    std::lock_guard<std::mutex> lock(local_kvcm_cleanup_mutex_);
    for (const auto& id : ids) local_kvcm_finished_ids_.erase(id);
}

void BlockTransferDispatcher::shutdown() const {
    if (per_rank_engine_)
        per_rank_engine_->shutdown();
    waitForTransfers();
}

void BlockTransferDispatcher::waitForTransfers() const {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    if (inflight_transfers_ != 0 && drain_observer_for_test_) {
        lock.unlock();
        drain_observer_for_test_();
        lock.lock();
    }
    completion_cv_.wait(lock, [this] { return inflight_transfers_ == 0; });
}

void BlockTransferDispatcher::registerTransfer() const {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    RTP_LLM_CHECK(inflight_transfers_ < std::numeric_limits<size_t>::max());
    ++inflight_transfers_;
}

void BlockTransferDispatcher::retireTransfer() const noexcept {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    RTP_LLM_CHECK(inflight_transfers_ != 0);
    if (--inflight_transfers_ == 0) {
        // Notify before unlocking: a drain waiter can destroy the dispatcher as
        // soon as it acquires this mutex and observes zero.
        completion_cv_.notify_all();
    }
}

std::shared_ptr<TransferAsyncContext> BlockTransferDispatcher::executePerRank(TransferTask task) const {
    if (task.expired()) {
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
                           CacheCopyError::NOT_EXECUTED),
            task.descriptors().size());
    }
    // The final callback releases this ticket after notifying all local consumers;
    // the submission stack retains another owner through finishSubmitting().
    const auto ticket  = std::make_shared<CompletionTicket>(this);
    auto       context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    dispatchGroups(
        std::move(task), [ticket, context](TransferResult result) { context->complete(std::move(result)); }, true);
    return context;
}

std::shared_ptr<TransferAsyncContext>
BlockTransferDispatcher::executeMultiRank(TransferTask task, const char* completion_operation) const {
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(std::move(task), completion_operation);
    }
    return executePerRank(std::move(task));
}

void BlockTransferDispatcher::runTransfer(TransferTask         task,
                                          TransferDoneCallback callback,
                                          const char*          completion_operation) const {
    // Keep a local owner even if the final callback completes synchronously.
    const auto ticket = std::make_shared<CompletionTicket>(this);
    callback = [ticket, block_count = task.descriptors().size(), done = std::move(callback)](TransferResult result) {
        result.expandBlockErrors(block_count);
        if (!result.ok()) {
            RTP_LLM_LOG_ERROR("block transfer failed: descriptor_count=%zu, error_type=%s, error_code=%d, details=%s",
                              block_count,
                              cacheCopyErrorName(result.copy_error),
                              static_cast<int>(result.status.code()),
                              result.status.ToString().c_str());
        }
        done(std::move(result));
    };
    dispatchGroups(std::move(task), std::move(callback), false, completion_operation);
}

void BlockTransferDispatcher::dispatchGroups(TransferTask         task,
                                             TransferDoneCallback callback,
                                             bool                 local,
                                             const char*          completion_operation) const {
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        callback(ErrorInfo(ErrorCode::INVALID_PARAMS, "transfer task contains no descriptors"));
        return;
    }
    if (task.expired()) {
        callback(TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before dispatch"),
                                CacheCopyError::NOT_EXECUTED));
        return;
    }

    struct DescriptorGroup {
        Tier                            source{Tier::NONE};
        Tier                            target{Tier::NONE};
        size_t                          group_set_id{0};
        std::vector<TransferDescriptor> descriptors;
        std::vector<size_t>             indices;
    };

    std::vector<DescriptorGroup> groups;
    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        const auto  group =
            std::find_if(groups.begin(), groups.end(), [&descriptor, local](const DescriptorGroup& item) {
                return item.source == descriptor.source_tier && item.target == descriptor.target_tier
                       && (!local || item.group_set_id == descriptor.group_set_id);
            });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
            group->indices.push_back(index);
        } else {
            groups.push_back(DescriptorGroup{
                descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id, {descriptor}, {index}});
        }
    }

    auto stage_state          = std::make_shared<TransferStageState>(std::move(callback), descriptors.size());
    bool any_batch_registered = false;
    for (const auto& group : groups) {
        const size_t batch_limit =
            local ? std::min(max_descriptors_per_batch_, per_rank_engine_->batchLimit()) : group.descriptors.size();
        for (size_t begin = 0; begin < group.descriptors.size(); begin += batch_limit) {
            std::vector<size_t> indices;
            bool                batch_registered = false;
            TransferResult      batch_error;
            try {
                const size_t                    end = std::min(begin + batch_limit, group.descriptors.size());
                std::vector<TransferDescriptor> batch(group.descriptors.begin() + begin,
                                                      group.descriptors.begin() + end);
                indices.assign(group.indices.begin() + begin, group.indices.begin() + end);
                stage_state->addBatch();
                batch_registered     = true;
                any_batch_registered = true;
                auto context         = local ? per_rank_engine_->execute(task.subtask(std::move(batch))) :
                                               executeMultiRank(task.rpcSubtask(std::move(batch)), completion_operation);
                if (context == nullptr) {
                    batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context");
                } else {
                    context->onDone([stage_state, indices](TransferResult result) {
                        stage_state->completeBatch(std::move(result), indices);
                    });
                }
            } catch (const std::exception& error) {
                if (!any_batch_registered) {
                    throw;
                }
                batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what());
            } catch (...) {
                if (!any_batch_registered) {
                    throw;
                }
                batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                        batch_registered ? "unknown transfer submission exception" :
                                                           "unknown transfer preparation exception");
            }

            if (!batch_error.ok()) {
                if (!batch_registered) {
                    // Stop new submissions and join the batches already registered.
                    stage_state->finishSubmitting(std::move(batch_error));
                    return;
                }
                stage_state->completeBatch(std::move(batch_error), indices);
            }
        }
    }
    // Final callback exceptions must propagate without re-entering submission handling.
    stage_state->finishSubmitting();
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
