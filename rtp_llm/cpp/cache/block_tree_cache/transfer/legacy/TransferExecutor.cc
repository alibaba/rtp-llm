#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
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
    const auto proof = task.completionProof();
    block_tree_cache_detail::ScopeRollback unsubmitted([proof] {
        if (proof) proof->report(true);
    });
    const auto&   descriptors = task.descriptors();
    const Tier    source      = descriptors.front().source_tier;
    const Tier    target      = descriptors.front().target_tier;
    const auto    deadline    = task.deadline();
    auto          context     =
        std::make_shared<TransferBatchAsyncContext>(descriptors.size(), nullptr, task.completionProof());
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
                                                           identity,
                                                           AsyncProfileKind::Wait);
    auto          on_timeout  = [this, context, proof, source, target, queue_begin, queue_timer, total_timer]() {
        block_tree_cache_detail::ScopeRollback publish([proof] {
            if (proof) proof->report(true);
        });
        try {
            queue_timer.finish(false);
            total_timer.finish(false);
            if (metrics_reporter_ != nullptr) {
                metrics_reporter_->reportQueueWaitMetric(
                    false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
            }
            TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired in TE worker queue"),
                                  CacheCopyError::NOT_EXECUTED);
            result.buffer_safe = true;
            context->complete(std::move(result));
        } catch (const std::exception& error) {
            context->complete(TransferResult::executionException(true, error.what()));
        } catch (...) {
            context->complete(TransferResult::executionException(true));
        }
    };
    const auto task_class = isLoadTransfer(source, target) ? BlockTreeTaskClass::LOAD : BlockTreeTaskClass::BACKGROUND;
    auto&      task_pool  = task_class == BlockTreeTaskClass::LOAD ? *load_task_pool_ : transfer_task_pool_;
    const bool accepted   = task_pool.submit(
        task_class,
        [this,
         task       = std::move(task),
         hosts      = std::move(hosts),
         group_sets = std::move(group_sets),
         context,
         proof,
         source,
         target,
         identity,
         request_id,
         transfer_id,
         queue_timer,
         total_timer,
         queue_begin]() {
            bool buffer_safe = true;  // No operation has started yet.
            block_tree_cache_detail::ScopeRollback publish([&] {
                if (proof) proof->report(buffer_safe);
            });
            const auto& descriptors = task.descriptors();
            TransferResult result;
            size_t         active_begin = 0;
            size_t         active_end   = 0;
            const auto     record_batch = [&](TransferResult batch) {
                for (size_t i = active_begin; i < active_end; ++i) {
                    result.block_errors[i] = batch.blockError(i - active_begin);
                }
                result.merge(std::move(batch));
            };
            try {
                queue_timer.finish(true);
                if (metrics_reporter_ != nullptr) {
                    metrics_reporter_->reportQueueWaitMetric(
                        false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
                }
                result = TransferResult::emptyAggregate(descriptors.size());
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
                    const bool previous_safe = buffer_safe;
                    buffer_safe = false;  // An arbitrary producer exception is not proof.
                    auto batch = executeBatch(sub_hosts, sub_descriptors, sub_group_sets);
                    buffer_safe = previous_safe && batch.buffer_safe;
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
                result.buffer_safe = buffer_safe;
                record_batch(TransferResult::executionException(buffer_safe, error.what()));
            } catch (...) {
                result.buffer_safe = buffer_safe;
                record_batch(TransferResult::executionException(buffer_safe, "unknown transfer executor exception"));
            }
            total_timer.finish(result.ok());
            context->complete(std::move(result));
        },
        deadline,
        std::move(on_timeout));
    if (accepted)
        unsubmitted.dismiss();
    if (!accepted) {
        queue_timer.finish(false);
        total_timer.finish(false);
        TransferResult result(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"),
            CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;
        context->complete(std::move(result));
    }
    return context;
}

}  // namespace rtp_llm
