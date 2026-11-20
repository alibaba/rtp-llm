#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

LoadTaskRunner::LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                               const BlockTransferDispatcher*  transfer_dispatcher,
                               BlockTreeTaskPool&              task_pool,
                               BlockTreeCacheMetricsReporter&  metrics_reporter):
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter) {}

bool LoadTaskRunner::submit(std::shared_ptr<LoadTask> task) const {
    auto queue_wait = TransferStageTimer::create(CacheStageContext::forWorkflow(metrics_reporter_,
                                                                                task->load_descs,
                                                                                "load",
                                                                                task->context->requestId().value_or(-1),
                                                                                task->context->workflowId()),
                                                 "business_queue_wait",
                                                 "business_task",
                                                 AsyncProfileKind::Wait);
    block_tree_cache_detail::ScopeRollback discard_wait([&] { queue_wait.discard(); });
    auto                                   on_timeout = [task, queue_wait]() {
        queue_wait.finish(false);
        const size_t descriptor_count = task->load_descs.size();
        RTP_LLM_LOG_WARNING("load expired in business queue, descriptor_count=%zu", descriptor_count);
        if (!task->context->completeTransfers(descriptor_count, false)) {
            RTP_LLM_LOG_WARNING("failed to record expired load, descriptor_count=%zu", descriptor_count);
        }
    };
    const TransferTask& queue_task =
        task->disk_to_device_task.descriptors().empty() ? task->host_to_device_task : task->disk_to_device_task;
    if (queue_task.expired()) {
        on_timeout();
    } else if (!task_pool_.submit(
                   BlockTreeTaskClass::LOAD,
                   [this, task, queue_wait]() {
                       queue_wait.finish(true);
                       const auto complete = [task]() {
                           for (const Tier tier : {Tier::HOST, Tier::DISK}) {
                               const auto& transfer =
                                   tier == Tier::HOST ? task->host_to_device_task : task->disk_to_device_task;
                               const auto&  result  = tier == Tier::HOST ? task->host_result : task->disk_result;
                               const size_t count   = transfer.descriptors().size();
                               const bool   success = result.ok() && !result.block_errors.empty();
                               if (count != 0 && !task->context->completeTransfers(count, success, tier)) {
                                   RTP_LLM_LOG_WARNING("failed to record load copy completion, descriptor_count=%zu",
                                                       count);
                               }
                           }
                       };
                       try {
                           run(task, complete);
                       } catch (const std::exception& error) {
                           RTP_LLM_LOG_ERROR("load task runner failed with exception: %s", error.what());
                           complete();
                       } catch (...) {
                           RTP_LLM_LOG_ERROR("load task runner failed with unknown exception");
                           complete();
                       }
                   },
                   queue_task.deadline(),
                   std::move(on_timeout))) {
        return false;
    }
    discard_wait.dismiss();
    return true;
}

void LoadTaskRunner::run(std::shared_ptr<LoadTask> task, std::function<void()> callback) const {
    const auto& completion_transfer =
        task->disk_to_device_task.descriptors().empty() ? task->host_to_device_task : task->disk_to_device_task;
    const auto workflow = CacheStageContext::forWorkflow(metrics_reporter_,
                                                         task->load_descs,
                                                         "load",
                                                         completion_transfer.requestId().value_or(-1),
                                                         completion_transfer.traceId());
    auto       transfers =
        std::make_shared<TransferStageState>([task, workflow, callback = std::move(callback)](TransferResult result) {
            task->wait.finish(result.ok());
            auto notify = TransferStageTimer::scope(workflow, "context_complete", "workflow_callback", result.ok());
            callback();
        });

    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        const auto& transfer = source_tier == Tier::HOST ? task->host_to_device_task : task->disk_to_device_task;
        if (transfer.descriptors().empty()) {
            continue;
        }
        transfers->addBatch();
        int64_t transfer_begin   = 0;
        bool    transfer_started = false;
        auto complete = [task,
                         transfers,
                         &metrics_reporter = metrics_reporter_,
                         &group_sets       = group_sets_,
                         source_tier,
                         observe_wait =
                             workflow.enabled](TransferResult result, int64_t transfer_begin, bool transfer_started) {
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
                    metrics_reporter.reportTransferFinished(TransferOperation::LOAD,
                                                            source_tier,
                                                            Tier::DEVICE,
                                                            transfer.descriptors().size(),
                                                            transfer_begin,
                                                            result.ok(),
                                                            result.ok() ? transfer.descriptors() : DescriptorView{},
                                                            group_sets);
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
                auto  save = TransferStageTimer::scope(stage_context, "save_result", "workflow_callback", result.ok());
                auto& saved_result = source_tier == Tier::HOST ? task->host_result : task->disk_result;
                saved_result       = result;
            }
            transfers->completeBatch(std::move(result));
        };
        try {
            auto stage_context = CacheStageContext::forTransfer(metrics_reporter_, transfer, "load");
            auto submit        = TransferStageTimer::scope(stage_context, "next_stage_submit", "workflow_callback");
            transfer_begin =
                metrics_reporter_.reportTransferStarted(TransferOperation::LOAD, source_tier, Tier::DEVICE);
            transfer_started = true;
            transfer_dispatcher_->runTransfer(
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

}  // namespace rtp_llm
