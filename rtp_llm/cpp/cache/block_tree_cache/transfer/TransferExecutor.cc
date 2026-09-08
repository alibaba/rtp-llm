#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

TransferExecutor::TransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                                   size_t                                         max_descriptors_per_batch,
                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    transfer_task_pool_(transfer_task_pool),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    metrics_reporter_(std::move(metrics_reporter)) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
}

std::shared_ptr<AsyncContext> TransferExecutor::execute(TransferTask                 task,
                                                        std::vector<HostBufferView>  hosts,
                                                        std::vector<const GroupSet*> group_sets) {
    const auto&   descriptors = task.descriptors();
    const Tier    source      = descriptors.front().source_tier;
    const Tier    target      = descriptors.front().target_tier;
    const auto    deadline    = task.deadline();
    auto          context     = std::make_shared<TransferBatchAsyncContext>();
    const int64_t queue_begin = currentTimeUs();
    auto          on_timeout  = [this, context, source, target, queue_begin]() {
        if (metrics_reporter_ != nullptr) {
            metrics_reporter_->reportQueueWaitMetric(
                false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
        }
        context->complete(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired in TE worker queue"));
    };
    const auto task_class = target == Tier::DEVICE ? BlockTreeTaskClass::LOAD : BlockTreeTaskClass::BACKGROUND;
    const bool accepted   = transfer_task_pool_.submit(
        task_class,
        [this,
         task       = std::move(task),
         hosts      = std::move(hosts),
         group_sets = std::move(group_sets),
         context,
         source,
         target,
         queue_begin]() {
            const auto& descriptors = task.descriptors();
            if (metrics_reporter_ != nullptr) {
                metrics_reporter_->reportQueueWaitMetric(
                    false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
            }
            try {
                for (size_t begin = 0; begin < descriptors.size(); begin += max_descriptors_per_batch_) {
                    const size_t end = std::min(begin + max_descriptors_per_batch_, descriptors.size());
                    const std::vector<HostBufferView>     sub_hosts(hosts.begin() + begin, hosts.begin() + end);
                    const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                          descriptors.begin() + end);
                    const std::vector<const GroupSet*>    sub_group_sets(group_sets.begin() + begin,
                                                                      group_sets.begin() + end);
                    const TransferStatus status = executeBatch(sub_hosts, sub_descriptors, sub_group_sets);
                    if (status != TransferStatus::OK) {
                        for (size_t index = begin; index < end; ++index) {
                            RTP_LLM_LOG_WARNING("transfer batch item failed, index=%zu %s",
                                                index,
                                                descriptors[index].debugString().c_str());
                        }
                        const auto error = transferStatusToErrorInfo(status);
                        context->complete(ErrorInfo(error.code(),
                                                    error.ToString() + ", descriptor_range=[" + std::to_string(begin)
                                                        + "," + std::to_string(end) + ")"));
                        return;
                    }
                }
                context->complete(ErrorInfo::OkStatus());
            } catch (const std::exception& error) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
            } catch (...) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer executor exception"));
            }
        },
        deadline,
        std::move(on_timeout));
    if (!accepted) {
        context->complete(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"));
    }
    return context;
}

}  // namespace rtp_llm
