#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

EvictionTaskRunner::EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher,
                                       int                             memory_timeout_ms,
                                       int                             disk_timeout_ms):
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    memory_timeout_ms_(memory_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms) {}

void EvictionTaskRunner::runTransfer(std::shared_ptr<const EvictionTransferTask> task,
                                     BlockTreeCacheMetricsReporter&              metrics_reporter,
                                     EvictionDoneCallback                        on_done) const {
    const int64_t transfer_begin_time_us = metrics_reporter.reportTransferStarted(
        CacheTransferOperation::EVICT, task->desc.source_tier, task->desc.target_tier);
    if (!task->desc.isExecutable()) {
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                task->desc.source_tier,
                                                task->desc.target_tier,
                                                1,
                                                transfer_begin_time_us,
                                                false,
                                                {},
                                                group_sets_);
        if (on_done) {
            on_done(false);
        }
        return;
    }

    const bool uses_disk           = task->desc.source_tier == Tier::DISK || task->desc.target_tier == Tier::DISK;
    const int  transfer_timeout_ms = uses_disk ? disk_timeout_ms_ : memory_timeout_ms_;
    transfer_dispatcher_->runTransfer(
        {task->desc},
        transfer_timeout_ms,
        [this, task, &metrics_reporter, transfer_begin_time_us, on_done = std::move(on_done)](ErrorInfo error) mutable {
            const bool                            success = error.ok();
            const std::vector<TransferDescriptor> successful_descriptors =
                success ? std::vector<TransferDescriptor>{task->desc} : std::vector<TransferDescriptor>{};
            metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                    task->desc.source_tier,
                                                    task->desc.target_tier,
                                                    1,
                                                    transfer_begin_time_us,
                                                    success,
                                                    successful_descriptors,
                                                    group_sets_);
            if (on_done) {
                on_done(success);
            }
        });
}

}  // namespace rtp_llm
