#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"

namespace rtp_llm {

EvictionTaskRunner::EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher,
                                       BlockTreeTaskPool*              task_pool,
                                       BlockTreeCacheMetricsReporter&  metrics_reporter):
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter) {}

bool EvictionTaskRunner::submit(std::shared_ptr<const EvictionTransferTask> task,
                                std::function<void(bool)>                   on_done) const {
    auto queue_wait =
        TransferStageTimer::create(CacheStageContext::forTransfer(metrics_reporter_, task->transfer_task, "evict"),
                                   "business_queue_wait",
                                   "business_task",
                                   AsyncProfileKind::Wait);
    block_tree_cache_detail::ScopeRollback discard_wait([&] { queue_wait.discard(); });
    auto                                   on_timeout = [task, queue_wait, on_done]() {
        queue_wait.finish(false);
        RTP_LLM_LOG_WARNING("eviction expired in business queue, source=%s target=%s descriptors=%zu",
                            tierName(task->descriptors().front().source_tier),
                            tierName(task->descriptors().front().target_tier),
                            task->descriptors().size());
        on_done(false);
    };
    if (!task_pool_->submit(
            BlockTreeTaskClass::BACKGROUND,
            [this, task, queue_wait, on_done]() {
                queue_wait.finish(true);
                try {
                    run(task, on_done);
                    return;
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
                }
                on_done(false);
            },
            task->transfer_task.deadline(),
            std::move(on_timeout))) {
        const auto& first  = task->descriptors().front();
        const auto  queues = task_pool_->queueSizes();
        RTP_LLM_INTERVAL_LOG(5,
                             WARN,
                             "eviction submission rejected: group_set=%zu source=%s target=%s blocks=%zu "
                             "queue_load=%zu queue_background=%zu queue_completion=%zu",
                             first.group_set_id,
                             tierName(first.source_tier),
                             tierName(first.target_tier),
                             task->descriptors().size(),
                             queues.load,
                             queues.background,
                             queues.completion);
        return false;
    }
    discard_wait.dismiss();
    return true;
}

void EvictionTaskRunner::run(std::shared_ptr<const EvictionTransferTask> task,
                             std::function<void(bool)>                   on_done) const {
    const auto&               descriptors = task->descriptors();
    const TransferDescriptor& first       = descriptors.front();
    const bool    valid = std::all_of(descriptors.begin(), descriptors.end(), [&first](const TransferDescriptor& desc) {
        return desc.isExecutable() && desc.group_set_id == first.group_set_id && desc.source_tier == first.source_tier
               && desc.target_tier == first.target_tier;
    });
    const int64_t transfer_begin =
        metrics_reporter_.reportTransferStarted(TransferOperation::EVICT, first.source_tier, first.target_tier);
    if (!valid) {
        metrics_reporter_.reportTransferFinished(TransferOperation::EVICT,
                                                 first.source_tier,
                                                 first.target_tier,
                                                 descriptors.size(),
                                                 transfer_begin,
                                                 false,
                                                 {},
                                                 group_sets_);
        metrics_reporter_.maybeReportCopyError(
            first.source_tier, first.target_tier, ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid eviction transfer"));
        on_done(false);
        return;
    }

    transfer_dispatcher_->runTransfer(
        task->transfer_task,
        [this, task, transfer_begin, on_done = std::move(on_done)](TransferResult result) {
            const bool success = result.ok();
            auto       stage_context = CacheStageContext::forTransfer(metrics_reporter_, task->transfer_task, "evict");
            auto       total = TransferStageTimer::scope(stage_context, "callback_total", "workflow_callback", success);
            {
                auto metrics = TransferStageTimer::scope(stage_context, "metrics_report", "workflow_callback", success);
                metrics_reporter_.reportTransferFinished(TransferOperation::EVICT,
                                                         task->descriptors().front().source_tier,
                                                         task->descriptors().front().target_tier,
                                                         task->descriptors().size(),
                                                         transfer_begin,
                                                         success,
                                                         success ? task->descriptors() :
                                                                   std::vector<TransferDescriptor>{},
                                                         group_sets_);
                metrics_reporter_.maybeReportCopyError(
                    task->descriptors().front().source_tier, task->descriptors().front().target_tier, result);
            }
            on_done(success);
        },
        "evict");
}

}  // namespace rtp_llm
