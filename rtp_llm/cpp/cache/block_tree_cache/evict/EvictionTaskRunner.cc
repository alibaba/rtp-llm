#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

namespace {

class EvictionTransferState: public std::enable_shared_from_this<EvictionTransferState> {
public:
    EvictionTransferState(std::shared_ptr<const EvictionTask>      task,
                          const std::vector<GroupSetPtr>&          group_sets,
                          const BlockTransferDispatcher*           transfer_dispatcher,
                          BlockTreeCacheMetricsReporter&           metrics_reporter,
                          int                                      transfer_timeout_ms,
                          int64_t                                  transfer_begin_time_us,
                          EvictionTaskRunner::EvictionDoneCallback on_done):
        task_(std::move(task)),
        group_sets_(group_sets),
        transfer_dispatcher_(transfer_dispatcher),
        metrics_reporter_(metrics_reporter),
        transfer_timeout_ms_(transfer_timeout_ms),
        transfer_begin_time_us_(transfer_begin_time_us),
        on_done_(std::move(on_done)) {}

    void start() {
        auto self = shared_from_this();
        transfer_dispatcher_->runTransfer(
            {task_->primary_desc},
            transfer_timeout_ms_,
            [self](ErrorInfo error) { self->onPrimaryDone(std::move(error)); });
    }

private:
    void onPrimaryDone(ErrorInfo error) {
        task_result_.primary_success = error.ok();
        if (!task_result_.primary_success) {
            task_result_.cascade_success.assign(task_->cascade_descs.size(), false);
            finish();
            return;
        }
        overall_success_ = true;
        task_result_.cascade_success.reserve(task_->cascade_descs.size());
        submitNextCascade();
    }

    void submitNextCascade() {
        if (next_cascade_index_ >= task_->cascade_descs.size()) {
            finish();
            return;
        }
        const size_t index = next_cascade_index_++;
        auto         self  = shared_from_this();
        transfer_dispatcher_->runTransfer(
            {task_->cascade_descs[index]},
            transfer_timeout_ms_,
            [self](ErrorInfo error) { self->onCascadeDone(std::move(error)); });
    }

    void onCascadeDone(ErrorInfo error) {
        const bool success = error.ok();
        task_result_.cascade_success.push_back(success);
        overall_success_ = overall_success_ && success;
        submitNextCascade();
    }

    void finish() {
        if (finished_) {
            return;
        }
        finished_ = true;

        std::vector<TransferDescriptor> successful_descriptors;
        successful_descriptors.reserve(task_->cascade_descs.size() + 1);
        if (task_result_.primary_success) {
            successful_descriptors.push_back(task_->primary_desc);
        }
        for (size_t i = 0; i < task_result_.cascade_success.size(); ++i) {
            if (task_result_.cascade_success[i]) {
                successful_descriptors.push_back(task_->cascade_descs[i]);
            }
        }
        metrics_reporter_.reportTransferFinished(CacheTransferOperation::EVICT,
                                                 task_->primary_desc.source_tier,
                                                 task_->primary_desc.target_tier,
                                                 task_->cascade_descs.size() + 1,
                                                 transfer_begin_time_us_,
                                                 overall_success_,
                                                 successful_descriptors,
                                                 group_sets_);

        auto on_done = std::move(on_done_);
        if (on_done) {
            on_done(std::move(task_result_));
        }
    }

    std::shared_ptr<const EvictionTask>      task_;
    const std::vector<GroupSetPtr>&          group_sets_;
    const BlockTransferDispatcher*           transfer_dispatcher_{nullptr};
    BlockTreeCacheMetricsReporter&           metrics_reporter_;
    int                                      transfer_timeout_ms_{0};
    int64_t                                  transfer_begin_time_us_{0};
    EvictionTaskRunner::EvictionDoneCallback on_done_;
    EvictionTaskResult                       task_result_;
    size_t                                   next_cascade_index_{0};
    bool                                     overall_success_{false};
    bool                                     finished_{false};
};

}  // namespace

EvictionTaskRunner::EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher,
                                       int                             memory_timeout_ms,
                                       int                             disk_timeout_ms):
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    memory_timeout_ms_(memory_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms) {}

void EvictionTaskRunner::runTransfer(std::shared_ptr<const EvictionTask> task,
                                     BlockTreeCacheMetricsReporter&     metrics_reporter,
                                     EvictionDoneCallback               on_done) const {
    if (task == nullptr) {
        if (on_done) {
            on_done(EvictionTaskResult{});
        }
        return;
    }

    const int64_t transfer_begin_time_us = metrics_reporter.reportTransferStarted(
        CacheTransferOperation::EVICT, task->primary_desc.source_tier, task->primary_desc.target_tier);
    if (!validateTransferDescriptors(*task)) {
        EvictionTaskResult task_result;
        task_result.cascade_success.assign(task->cascade_descs.size(), false);
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                task->primary_desc.source_tier,
                                                task->primary_desc.target_tier,
                                                task->cascade_descs.size() + 1,
                                                transfer_begin_time_us,
                                                false,
                                                {},
                                                group_sets_);
        if (on_done) {
            on_done(std::move(task_result));
        }
        return;
    }

    const int transfer_timeout_ms = selectTransferTimeoutMs(*task, memory_timeout_ms_, disk_timeout_ms_);
    auto state = std::make_shared<EvictionTransferState>(std::move(task),
                                                        group_sets_,
                                                        transfer_dispatcher_,
                                                        metrics_reporter,
                                                        transfer_timeout_ms,
                                                        transfer_begin_time_us,
                                                        std::move(on_done));
    state->start();
}

bool EvictionTaskRunner::validateTransferDescriptors(const EvictionTask& task) {
    if (!task.primary_desc.isExecutable()) {
        return false;
    }
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (!cascade_desc.isExecutable()) {
            return false;
        }
    }
    return true;
}

int EvictionTaskRunner::selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms) {
    bool uses_disk = task.primary_desc.source_tier == Tier::DISK || task.primary_desc.target_tier == Tier::DISK;
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (cascade_desc.source_tier == Tier::DISK || cascade_desc.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    return uses_disk ? disk_timeout_ms : memory_timeout_ms;
}

}  // namespace rtp_llm
