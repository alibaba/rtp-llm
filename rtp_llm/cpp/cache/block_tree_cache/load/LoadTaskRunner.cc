#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

LoadTaskRunner::LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets, int host_timeout_ms, int disk_timeout_ms):
    group_sets_(group_sets), host_timeout_ms_(host_timeout_ms), disk_timeout_ms_(disk_timeout_ms) {}

LoadTaskRunner::TaskPtr LoadTaskRunner::createTask(const std::shared_ptr<LoadAsyncContext>& context) {
    const std::vector<TransferDescriptor>& load_descs = context->loadDescs();
    std::vector<TransferDescriptor>        task_load_descs;
    std::vector<TransferDescriptor>        host_to_device_descriptors;
    std::vector<TransferDescriptor>        disk_to_device_descriptors;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (context->joinedLoads()[desc_index] || desc.source_tier == Tier::DEVICE) {
            continue;
        }
        task_load_descs.push_back(desc);
        if (desc.source_tier == Tier::HOST) {
            host_to_device_descriptors.push_back(desc);
        } else {
            disk_to_device_descriptors.push_back(desc);
        }
    }
    if (task_load_descs.empty()) {
        return nullptr;
    }

    return std::make_shared<Task>(
        std::move(task_load_descs),
        TransferTask(std::move(host_to_device_descriptors), std::chrono::milliseconds(host_timeout_ms_)),
        TransferTask(std::move(disk_to_device_descriptors), std::chrono::milliseconds(disk_timeout_ms_)),
        context);
}

void LoadTaskRunner::runTransfer(TaskPtr                        task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 TransferDoneCallback           callback) {
    try {
        if (task->host_to_device_task.descriptors().empty()) {
            startDiskTransfer(task, transfer_dispatcher, metrics_reporter, callback);
            return;
        }

        task->phase = Task::Phase::HOST_TO_DEVICE;
        const int64_t transfer_begin =
            metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::HOST, Tier::DEVICE);
        transfer_dispatcher.runTransfer(
            task->host_to_device_task,
            [this, task, &transfer_dispatcher, &metrics_reporter, transfer_begin, callback = std::move(callback)](
                ErrorInfo error) mutable {
                try {
                    reportStageFinished(*task,
                                        metrics_reporter,
                                        Tier::HOST,
                                        task->host_to_device_task.descriptors(),
                                        transfer_begin,
                                        error.ok());
                    if (!error.ok()) {
                        task->phase = Task::Phase::FINISHED;
                        callback(std::move(error));
                        return;
                    }
                    startDiskTransfer(task, transfer_dispatcher, metrics_reporter, callback);
                } catch (const std::exception& exception) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()));
                } catch (...) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown host load completion exception"));
                }
            });
    } catch (const std::exception& error) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown host load submission exception"));
    }
}

void LoadTaskRunner::startDiskTransfer(TaskPtr                        task,
                                       const BlockTransferDispatcher& transfer_dispatcher,
                                       BlockTreeCacheMetricsReporter& metrics_reporter,
                                       TransferDoneCallback           callback) {
    if (task->disk_to_device_task.descriptors().empty()) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo::OkStatus());
        return;
    }

    task->phase = Task::Phase::DISK_TO_DEVICE;
    const int64_t transfer_begin =
        metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::DISK, Tier::DEVICE);
    transfer_dispatcher.runTransfer(
        task->disk_to_device_task,
        [this, task, &metrics_reporter, transfer_begin, callback = std::move(callback)](ErrorInfo error) mutable {
            try {
                reportStageFinished(*task,
                                    metrics_reporter,
                                    Tier::DISK,
                                    task->disk_to_device_task.descriptors(),
                                    transfer_begin,
                                    error.ok());
                task->phase = Task::Phase::FINISHED;
                callback(std::move(error));
            } catch (const std::exception& exception) {
                task->phase = Task::Phase::FINISHED;
                callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()));
            } catch (...) {
                task->phase = Task::Phase::FINISHED;
                callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown disk load completion exception"));
            }
        });
}

void LoadTaskRunner::reportStageFinished(const Task&                            task,
                                         BlockTreeCacheMetricsReporter&         metrics_reporter,
                                         Tier                                   source_tier,
                                         const std::vector<TransferDescriptor>& descriptors,
                                         int64_t                                begin_time_us,
                                         bool                                   success) const {
    static const std::vector<TransferDescriptor> empty_descriptors;
    metrics_reporter.reportTransferFinished(CacheTransferOperation::LOAD,
                                            source_tier,
                                            Tier::DEVICE,
                                            descriptors.size(),
                                            begin_time_us,
                                            success,
                                            success ? descriptors : empty_descriptors,
                                            group_sets_);
}

void LoadTaskRunner::releaseTaskResources(const Task& task) {
    const auto& descriptors = task.load_descs;
    for (size_t desc_index = 0; desc_index < descriptors.size(); ++desc_index) {
        const TransferDescriptor& desc      = descriptors[desc_index];
        const GroupSetPtr&        group_set = group_sets_[desc.group_set_id];
        if (task.target_installed[desc_index]) {
            continue;
        }
        group_set->unreferenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockTreeRefType::LOAD);
    }
}

}  // namespace rtp_llm
