#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <map>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
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
                                 CacheObserver& metrics_reporter,
                                 TransferDoneCallback           callback) {
    task->phase = Task::Phase::TRANSFERRING;
    const auto& completion_transfer =
        task->disk_to_device_task.descriptors().empty() ? task->host_to_device_task : task->disk_to_device_task;
    const auto workflow  = CacheStageContext::forWorkflow(metrics_reporter,
                                                         task->load_descs,
                                                         "load",
                                                         completion_transfer.requestId().value_or(-1),
                                                         completion_transfer.traceId());
    auto       transfers = std::make_shared<TransferStageState>(
        [task, workflow, callback = std::move(callback)](TransferResult result) mutable {
            task->wait.finish(result.ok());
            task->phase         = Task::Phase::FINISHED;

            auto notify = TransferStageTimer::scope(workflow, "context_complete", "workflow_callback", result.ok());
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
            auto        stage_context = CacheStageContext::forTransfer(metrics_reporter, transfer, "load");
            auto total = TransferStageTimer::scope(stage_context, "callback_total", "workflow_callback", result.ok());
            if (observe_wait && !task->host_to_device_task.descriptors().empty()
                && !task->disk_to_device_task.descriptors().empty()) {
                std::lock_guard<std::mutex> lock(task->wait_mutex);
                bool&      completed       = source_tier == Tier::HOST ? task->host_completed : task->disk_completed;
                const bool other_completed = source_tier == Tier::HOST ? task->disk_completed : task->host_completed;
                completed                  = true;
                if (other_completed) {
                    task->direction_wait.finish(result.ok());
                } else if (result.ok()) {
                    task->direction_wait = TransferStageTimer::create(
                        stage_context, source_tier == Tier::HOST ? "wait_disk2d" : "wait_h2d", "workflow_callback");
                }
            }
            try {
                auto metrics =
                    TransferStageTimer::scope(stage_context, "metrics_report", "workflow_callback", result.ok());
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

            {
                auto save = TransferStageTimer::scope(stage_context, "save_result", "workflow_callback", result.ok());
                auto&                saved_result = source_tier == Tier::HOST ? task->host_result : task->disk_result;
                saved_result                      = result;
            }
            transfers->completeBatch(std::move(result));
        };
        try {
            auto stage_context = CacheStageContext::forTransfer(metrics_reporter, transfer, "load");
            auto submit        = TransferStageTimer::scope(stage_context, "next_stage_submit", "workflow_callback");
            transfer_begin =
                metrics_reporter.reportTransferStarted(TransferOperation::LOAD, source_tier, Tier::DEVICE);
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
    task->wait = TransferStageTimer::create(workflow, "wait", "workflow_callback");
    transfers->finishSubmitting();
}

void LoadTaskRunner::reportStageFinished(const Task&                    task,
                                         CacheObserver& metrics_reporter,
                                         Tier                           source_tier,
                                         DescriptorView                 descriptors,
                                         int64_t                        begin_time_us,
                                         bool                           success) const {
    static const std::vector<TransferDescriptor> empty_descriptors;
    metrics_reporter.reportTransferFinished(TransferOperation::LOAD,
                                            source_tier,
                                            Tier::DEVICE,
                                            descriptors.size(),
                                            begin_time_us,
                                            success,
                                            success ? descriptors : empty_descriptors,
                                            group_sets_);
}

void LoadTaskRunner::releaseTaskResources(const Task& task) {
    std::map<size_t, MultiNodeResource> target_holders;
    const auto&                         descriptors = task.load_descs;
    for (size_t desc_index = 0; desc_index < descriptors.size(); ++desc_index) {
        if (task.target_installed[desc_index]) {
            continue;
        }
        const TransferDescriptor& desc = descriptors[desc_index];
        auto&                     holder =
            target_holders.try_emplace(desc.group_set_id, MultiNodeResource{desc.group_set_id, Tier::DEVICE, {}})
                .first->second;
        holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
    }
    for (const auto& [group_set_id, holder] : target_holders) {
        group_sets_[group_set_id]->unreferenceBlocks(holder, BlockTreeRefType::LOAD);
    }
}

}  // namespace rtp_llm
