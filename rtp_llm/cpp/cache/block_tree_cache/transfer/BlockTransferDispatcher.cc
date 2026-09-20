#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t max_descriptors_per_batch):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
}

BlockTransferDispatcher::~BlockTransferDispatcher() {
    drainTransfers();
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

std::shared_ptr<AsyncContext> BlockTransferDispatcher::executePerRank(TransferTask task) const {
    if (task.expired()) {
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"));
    }
    return per_rank_engine_->execute(std::move(task));
}

std::shared_ptr<AsyncContext> BlockTransferDispatcher::executeMultiRank(TransferTask task) const {
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
    callback = [completion, done = std::move(callback)](ErrorInfo error) {
        // Context readiness can precede callback execution. Drain only after
        // the callback has submitted settlement (also on an exception).
        block_tree_cache_detail::ScopeRollback completed([&] { completion->set_value(); });
        done(std::move(error));
    };
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        callback(ErrorInfo(ErrorCode::INVALID_PARAMS, "transfer task contains no descriptors"));
        return;
    }
    if (task.expired()) {
        callback(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before dispatch"));
        return;
    }

    struct DescriptorGroup {
        Tier                            source{Tier::NONE};
        Tier                            target{Tier::NONE};
        size_t                          group_set_id{0};
        std::vector<TransferDescriptor> descriptors;
    };

    std::vector<DescriptorGroup> groups;
    for (const auto& descriptor : descriptors) {
        const auto group = std::find_if(groups.begin(), groups.end(), [&descriptor](const DescriptorGroup& item) {
            return item.source == descriptor.source_tier && item.target == descriptor.target_tier
                   && item.group_set_id == descriptor.group_set_id;
        });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
        } else {
            groups.push_back(
                DescriptorGroup{descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id, {descriptor}});
        }
    }

    auto stage_state = std::make_shared<TransferStageState>(std::move(callback));
    for (const auto& group : groups) {
        const size_t batch_limit = max_descriptors_per_batch_;
        for (size_t begin = 0; begin < group.descriptors.size(); begin += batch_limit) {
            const size_t                    end = std::min(begin + batch_limit, group.descriptors.size());
            std::vector<TransferDescriptor> batch(group.descriptors.begin() + begin, group.descriptors.begin() + end);
            stage_state->addBatch();
            try {
                auto context = executeMultiRank(task.subtask(std::move(batch)));
                if (context == nullptr) {
                    stage_state->completeBatch(
                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
                    continue;
                }
                context->onDone([stage_state](ErrorInfo error) { stage_state->completeBatch(std::move(error)); });
            } catch (const std::exception& error) {
                stage_state->completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
            } catch (...) {
                stage_state->completeBatch(
                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer submission exception"));
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
