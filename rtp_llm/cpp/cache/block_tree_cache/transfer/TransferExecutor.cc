#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

TransferExecutor::TransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                                   size_t                                         max_descriptors_per_batch,
                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                   BlockTreeTaskPool*                             load_task_pool,
                                   int64_t                                        world_rank):
    transfer_task_pool_(transfer_task_pool),
    load_task_pool_(load_task_pool == nullptr ? &transfer_task_pool : load_task_pool),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    metrics_reporter_(std::move(metrics_reporter)),
    world_rank_(world_rank) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
}

std::shared_ptr<TransferAsyncContext> TransferExecutor::execute(TransferTask                 task,
                                                                std::vector<HostBufferView>  hosts,
                                                                std::vector<const GroupSet*> group_sets) {
    const auto&   descriptors = task.descriptors();
    const Tier    source      = descriptors.front().source_tier;
    const Tier    target      = descriptors.front().target_tier;
    const auto    deadline    = task.deadline();
    auto          context     = std::make_shared<TransferBatchAsyncContext>(descriptors.size());
    const int64_t queue_begin = currentTimeUs();
    const auto    identity    = CacheStageContext::local(source, target);
    const auto    request_id  = task.requestId().value_or(-1);
    const auto    transfer_id = task.traceId();
    auto          total_timer = TransferStageTimer::create(metrics_reporter_,
                                                           "worker_task_total",
                                                           request_id,
                                                           0,
                                                           descriptors.size(),
                                                           world_rank_,
                                                           "worker_task",
                                                           transfer_id,
                                                           identity);
    auto          queue_timer = TransferStageTimer::create(metrics_reporter_,
                                                           "transfer_queue_wait",
                                                           request_id,
                                                           0,
                                                           descriptors.size(),
                                                           world_rank_,
                                                           "worker_task",
                                                           transfer_id,
                                                           identity);
    auto          on_timeout  = [this, context, source, target, queue_begin, queue_timer, total_timer]() {
        queue_timer.finish(false);
        total_timer.finish(false);
        if (metrics_reporter_ != nullptr) {
            metrics_reporter_->reportQueueWaitMetric(
                false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
        }
        context->complete(TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired in TE worker queue"),
                                         CacheCopyError::NOT_EXECUTED));
    };
    const auto task_class = target == Tier::DEVICE ? BlockTreeTaskClass::LOAD : BlockTreeTaskClass::BACKGROUND;
    auto&      task_pool  = task_class == BlockTreeTaskClass::LOAD ? *load_task_pool_ : transfer_task_pool_;
    const bool accepted   = task_pool.submit(
        task_class,
        [this,
         task       = std::move(task),
         hosts      = std::move(hosts),
         group_sets = std::move(group_sets),
         context,
         source,
         target,
         identity,
         request_id,
         transfer_id,
         queue_timer,
         total_timer,
         queue_begin]() {
            queue_timer.finish(true);
            const auto& descriptors = task.descriptors();
            if (metrics_reporter_ != nullptr) {
                metrics_reporter_->reportQueueWaitMetric(
                    false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
            }
            TransferResult result(descriptors.size());
            size_t         active_begin = 0;
            size_t         active_end   = 0;
            const auto     record_batch = [&](TransferResult batch) {
                for (size_t i = active_begin; i < active_end; ++i) {
                    result.block_errors[i] = batch.blockError(i - active_begin);
                }
                result.merge(std::move(batch));
            };
            try {
                for (size_t begin = 0; begin < descriptors.size(); begin += max_descriptors_per_batch_) {
                    const size_t end = std::min(begin + max_descriptors_per_batch_, descriptors.size());
                    active_begin     = begin;
                    active_end       = end;
                    const std::vector<HostBufferView>     sub_hosts(hosts.begin() + begin, hosts.begin() + end);
                    const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                          descriptors.begin() + end);
                    const std::vector<const GroupSet*>    sub_group_sets(group_sets.begin() + begin,
                                                                      group_sets.begin() + end);
                    const char*                           stage = source == Tier::DISK   ? "disk_read" :
                                                                  target == Tier::DISK   ? "disk_write" :
                                                                  target == Tier::DEVICE ? "h2d" :
                                                                                           "d2h";
                    TransferStageTimer                    batch_timer(metrics_reporter_,
                                                                      stage,
                                                                      request_id,
                                                                      begin,
                                                                      end,
                                                                      world_rank_,
                                                                      "worker_batch",
                                                                      transfer_id,
                                                                      identity);
                    auto batch = executeBatch(sub_hosts, sub_descriptors, sub_group_sets);
                    batch_timer.finish(batch.ok());
                    if (!batch.ok()) {
                        for (size_t index = begin; index < end; ++index) {
                            RTP_LLM_LOG_WARNING("transfer batch item failed, index=%zu %s",
                                                index,
                                                descriptors[index].debugString().c_str());
                        }
                        batch.status = ErrorInfo(batch.status.code(),
                                                 batch.status.ToString() + ", descriptor_range=["
                                                     + std::to_string(begin) + "," + std::to_string(end) + ")");
                        record_batch(std::move(batch));
                        break;
                    }
                    record_batch(std::move(batch));
                }
            } catch (const std::exception& error) {
                record_batch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
            } catch (...) {
                record_batch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer executor exception"));
            }
            total_timer.finish(result.ok());
            context->complete(std::move(result));
        },
        deadline,
        std::move(on_timeout));
    if (!accepted) {
        queue_timer.finish(false);
        total_timer.finish(false);
        context->complete(TransferResult(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"),
            CacheCopyError::NOT_EXECUTED));
    }
    return context;
}

}  // namespace rtp_llm
