#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

LoadTaskRunner::LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets): group_sets_(group_sets) {}

LoadTaskRunner::TaskPtr LoadTaskRunner::createTask(const std::shared_ptr<LoadAsyncContext>& context) {
    const std::vector<TransferDescriptor>& load_descs   = context->loadDescs();
    std::vector<TransferDescriptor> task_load_descs;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (context->joinedLoads()[desc_index] || desc.source_tier == Tier::DEVICE) {
            continue;
        }
        task_load_descs.push_back(desc);
    }
    if (task_load_descs.empty()) {
        return nullptr;
    }

    TaskPtr task     = std::make_shared<Task>();
    task->load_descs = std::move(task_load_descs);
    task->target_installed.assign(task->load_descs.size(), false);
    task->context = context;
    return task;
}

void LoadTaskRunner::runTransfer(TaskPtr                         task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms,
                                 TransferDoneCallback           callback) {
    task->host_to_device_descriptors.clear();
    task->disk_to_device_descriptors.clear();
    for (const TransferDescriptor& desc : task->load_descs) {
        if (desc.source_tier == Tier::HOST) {
            task->host_to_device_descriptors.push_back(desc);
        } else {
            task->disk_to_device_descriptors.push_back(desc);
        }
    }

    try {
        if (task->host_to_device_descriptors.empty()) {
            startDiskTransfer(task, transfer_dispatcher, metrics_reporter, disk_timeout_ms, callback);
            return;
        }

        task->phase = Task::Phase::HOST_TO_DEVICE;
        task->host_transfer_begin_time_us =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::HOST, Tier::DEVICE);
        transfer_dispatcher.runTransfer(
            task->host_to_device_descriptors,
            host_timeout_ms,
            [this, task, &transfer_dispatcher, &metrics_reporter, disk_timeout_ms, callback = std::move(callback)](
                ErrorInfo error) mutable {
                try {
                    reportStageFinished(*task,
                                        metrics_reporter,
                                        Tier::HOST,
                                        task->host_to_device_descriptors,
                                        task->host_transfer_begin_time_us,
                                        error.ok());
                    if (!error.ok()) {
                        task->phase = Task::Phase::FINISHED;
                        callback(std::move(error));
                        return;
                    }
                    startDiskTransfer(task, transfer_dispatcher, metrics_reporter, disk_timeout_ms, callback);
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

void LoadTaskRunner::startDiskTransfer(TaskPtr                         task,
                                       const BlockTransferDispatcher& transfer_dispatcher,
                                       BlockTreeCacheMetricsReporter& metrics_reporter,
                                       int                            disk_timeout_ms,
                                       TransferDoneCallback           callback) {
    if (task->disk_to_device_descriptors.empty()) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo::OkStatus());
        return;
    }

    task->phase = Task::Phase::DISK_TO_DEVICE;
    task->disk_transfer_begin_time_us =
        metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::DISK, Tier::DEVICE);
    transfer_dispatcher.runTransfer(
        task->disk_to_device_descriptors,
        disk_timeout_ms,
        [this, task, &metrics_reporter, callback = std::move(callback)](ErrorInfo error) mutable {
            try {
                reportStageFinished(*task,
                                    metrics_reporter,
                                    Tier::DISK,
                                    task->disk_to_device_descriptors,
                                    task->disk_transfer_begin_time_us,
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

void LoadTaskRunner::reportStageFinished(const Task&                              task,
                                         BlockTreeCacheMetricsReporter&           metrics_reporter,
                                         Tier                                     source_tier,
                                         const std::vector<TransferDescriptor>& descriptors,
                                         int64_t                                  begin_time_us,
                                         bool                                     success) const {
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
    for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc      = task.load_descs[desc_index];
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
