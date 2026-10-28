#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t max_descriptors_per_batch):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
    if (multi_rank_engine_) {
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
    drainTransfers();
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
    if (!multi_rank_engine_) return;
    multi_rank_engine_->forgetFinishedTasks();
    pollFrozenTasksForKind(FrozenTask::QueryKind::TRANSFER,
                           [this](const auto& ids) { return multi_rank_engine_->queryTasks(ids); });
    pollFrozenTasksForKind(FrozenTask::QueryKind::KVCM,
                           [this](const auto& ids) { return multi_rank_engine_->queryKVCMOperations(ids); });
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
        if (task.unfreeze(completed_ids) && multi_rank_engine_ && query_kind == FrozenTask::QueryKind::TRANSFER) {
            multi_rank_engine_->enqueueFinishedTasks(task.recovery_task_ids);
        }
    }
}

void BlockTransferDispatcher::drainTransfers() const {
    bool observed = false;
    for (;;) {
        std::vector<std::shared_future<void>> completions;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            completions.swap(transfer_completions_);
        }
        if (completions.empty()) {
            return;
        }
        if (!observed && drain_observer_for_test_) {
            observed = true;
            drain_observer_for_test_();
        }
        for (const auto& completion : completions) {
            completion.wait();
        }
        // A callback may have submitted another transfer before returning.
    }
}

std::shared_ptr<TransferAsyncContext> BlockTransferDispatcher::executePerRank(TransferTask task) const {
    if (task.expired()) {
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
                           CacheCopyError::NOT_EXECUTED),
            task.descriptors().size());
    }
    return per_rank_engine_->execute(std::move(task));
}

std::shared_ptr<TransferAsyncContext> BlockTransferDispatcher::executeMultiRank(TransferTask task) const {
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(std::move(task));
    }
    return executePerRank(std::move(task));
}

void BlockTransferDispatcher::runTransfer(TransferTask task, TransferDoneCallback callback) const {
    auto completion = std::make_shared<std::promise<void>>();
    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        transfer_completions_.erase(
            std::remove_if(transfer_completions_.begin(), transfer_completions_.end(), [](const auto& future) {
                return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }),
            transfer_completions_.end());
        transfer_completions_.push_back(completion->get_future().share());
    }
    callback =
        [completion, block_count = task.descriptors().size(), done = std::move(callback)](TransferResult result) {
            result.expandBlockErrors(block_count);
            // Context readiness can precede callback execution. Drain only after
            // the callback has submitted settlement (also on an exception).
            block_tree_cache_detail::ScopeRollback completed([&] { completion->set_value(); });
            if (!result.ok()) {
                RTP_LLM_LOG_ERROR("block transfer failed: descriptor_count=%zu, error_type=%s, error_code=%d, details=%s",
                                  block_count,
                                  cacheCopyErrorName(result.copy_error),
                                  static_cast<int>(result.status.code()),
                                  result.status.ToString().c_str());
            }
            done(std::move(result));
        };
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
        const auto  group      = std::find_if(groups.begin(), groups.end(), [&descriptor](const DescriptorGroup& item) {
            return item.source == descriptor.source_tier && item.target == descriptor.target_tier
                   && item.group_set_id == descriptor.group_set_id;
        });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
            group->indices.push_back(index);
        } else {
            groups.push_back(DescriptorGroup{
                descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id, {descriptor}, {index}});
        }
    }

    auto stage_state = std::make_shared<TransferStageState>(std::move(callback), descriptors.size());
    for (const auto& group : groups) {
        const size_t batch_limit = max_descriptors_per_batch_;
        for (size_t begin = 0; begin < group.descriptors.size(); begin += batch_limit) {
            const size_t                    end = std::min(begin + batch_limit, group.descriptors.size());
            std::vector<TransferDescriptor> batch(group.descriptors.begin() + begin, group.descriptors.begin() + end);
            std::vector<size_t>             indices(group.indices.begin() + begin, group.indices.begin() + end);
            stage_state->addBatch();
            try {
                auto context = executeMultiRank(task.subtask(std::move(batch)));
                if (context == nullptr) {
                    stage_state->completeBatch(
                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"), indices);
                    continue;
                }
                context->onDone([stage_state, indices](TransferResult result) {
                    stage_state->completeBatch(std::move(result), indices);
                });
            } catch (const std::exception& error) {
                stage_state->completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()), indices);
            } catch (...) {
                stage_state->completeBatch(
                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer submission exception"), indices);
            }
        }
    }
    stage_state->finishSubmitting();
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
