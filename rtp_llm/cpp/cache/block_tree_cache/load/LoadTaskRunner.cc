#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

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

    auto task = std::make_shared<Task>(
        std::move(task_load_descs),
        TransferTask(
            std::move(host_to_device_descriptors), std::chrono::milliseconds(host_timeout_ms_), context->requestId()),
        TransferTask(
            std::move(disk_to_device_descriptors), std::chrono::milliseconds(disk_timeout_ms_), context->requestId()),
        context);
    task->host_to_device_task.setWorkflow(context->workflowId(), context->workflowId());
    task->disk_to_device_task.setWorkflow(context->workflowId(), context->workflowId());
    return task;
}

void LoadTaskRunner::runTransfer(TaskPtr                        task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 TransferDoneCallback           callback) {
    task->phase = Task::Phase::TRANSFERRING;
    const auto& completion_transfer =
        task->disk_to_device_task.descriptors().empty() ? task->host_to_device_task : task->disk_to_device_task;
    const auto workflow  = CompletionStageContext::forWorkflow(metrics_reporter,
                                                              task->load_descs,
                                                              "load",
                                                              completion_transfer.requestId().value_or(-1),
                                                              completion_transfer.traceId());
    auto       transfers = std::make_shared<TransferStageState>(
        [task, workflow, callback = std::move(callback)](TransferResult result) mutable {
            task->wait.finish(result.ok());
            task->phase         = Task::Phase::FINISHED;
            auto observation    = workflow;
            observation.success = result.ok();
            CompletionStageScope notify(observation, "context_complete", "workflow_callback");
            callback(std::move(result.status));
        });

    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        const auto& transfer = source_tier == Tier::HOST ? task->host_to_device_task : task->disk_to_device_task;
        if (transfer.descriptors().empty()) {
            continue;
        }
        transfers->addBatch();
        int64_t transfer_begin   = 0;
        bool    transfer_started = false;
        auto    complete = [this, task, transfers, &metrics_reporter, source_tier, observe_wait = workflow.enabled](
                            TransferResult result, int64_t transfer_begin, bool transfer_started) {
            const auto& transfer = source_tier == Tier::HOST ? task->host_to_device_task : task->disk_to_device_task;
            auto observation     = CompletionStageContext::forTransfer(metrics_reporter, transfer, "load", result.ok());
            CompletionStageScope total(observation, "callback_total", "workflow_callback");
            if (observe_wait && !task->host_to_device_task.descriptors().empty()
                && !task->disk_to_device_task.descriptors().empty()) {
                std::lock_guard<std::mutex> lock(task->wait_mutex);
                bool&      completed       = source_tier == Tier::HOST ? task->host_completed : task->disk_completed;
                const bool other_completed = source_tier == Tier::HOST ? task->disk_completed : task->host_completed;
                completed                  = true;
                if (other_completed) {
                    task->direction_wait.finish(result.ok());
                } else if (result.ok()) {
                    task->direction_wait =
                        observation.start(source_tier == Tier::HOST ? "wait_disk2d" : "wait_h2d", "workflow_callback");
                }
            }
            try {
                CompletionStageScope metrics(observation, "metrics_report", "workflow_callback");
                if (transfer_started) {
                    reportStageFinished(
                        *task, metrics_reporter, source_tier, transfer.descriptors(), transfer_begin, result.ok());
                    metrics_reporter.maybeReportCopyError(source_tier, Tier::DEVICE, result);
                }
            } catch (const std::exception& exception) {
                total.setSuccess(false);
                result.merge(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()));
            } catch (...) {
                total.setSuccess(false);
                result.merge(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown load completion exception"));
            }
            observation.success = result.ok();
            {
                CompletionStageScope save(observation, "save_result", "workflow_callback");
                auto&                saved_result = source_tier == Tier::HOST ? task->host_result : task->disk_result;
                saved_result                      = result;
            }
            transfers->completeBatch(std::move(result));
        };
        try {
            auto                 observation = CompletionStageContext::forTransfer(metrics_reporter, transfer, "load");
            CompletionStageScope submit(observation, "next_stage_submit", "workflow_callback");
            transfer_begin =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, source_tier, Tier::DEVICE);
            transfer_started = true;
            transfer_dispatcher.runTransfer(
                transfer,
                [complete, transfer_begin](TransferResult result) {
                    complete(std::move(result), transfer_begin, true);
                },
                "load");
        } catch (const std::exception& exception) {
            complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()), transfer_begin, transfer_started);
        } catch (...) {
            complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown load submission exception"),
                     transfer_begin,
                     transfer_started);
        }
    }
    // Keep sources and targets protected until every submitted direction has drained,
    // including when an earlier direction fails or completes inline.
    task->wait = workflow.start("wait", "workflow_callback");
    transfers->finishSubmitting();
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
