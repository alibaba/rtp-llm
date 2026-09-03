#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>
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
    const TransferDescriptor& first = task->descs.front();
    const bool    valid = std::all_of(task->descs.begin(), task->descs.end(), [&first](const TransferDescriptor& desc) {
        return desc.isExecutable() && desc.group_set_id == first.group_set_id && desc.source_tier == first.source_tier
               && desc.target_tier == first.target_tier;
    });
    const int64_t transfer_begin_time_us =
        metrics_reporter.reportTransferStarted(CacheTransferOperation::EVICT, first.source_tier, first.target_tier);
    if (!valid) {
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                first.source_tier,
                                                first.target_tier,
                                                task->descs.size(),
                                                transfer_begin_time_us,
                                                false,
                                                {},
                                                group_sets_);
        if (on_done) {
            on_done(false);
        }
        return;
    }

    const bool uses_disk           = first.source_tier == Tier::DISK || first.target_tier == Tier::DISK;
    const int  transfer_timeout_ms = uses_disk ? disk_timeout_ms_ : memory_timeout_ms_;
    transfer_dispatcher_->runTransfer(
        task->descs,
        transfer_timeout_ms,
        [this, task, &metrics_reporter, transfer_begin_time_us, on_done = std::move(on_done)](ErrorInfo error) mutable {
            const bool success = error.ok();
            metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                    task->descs.front().source_tier,
                                                    task->descs.front().target_tier,
                                                    task->descs.size(),
                                                    transfer_begin_time_us,
                                                    success,
                                                    success ? task->descs : std::vector<TransferDescriptor>{},
                                                    group_sets_);
            if (on_done) {
                on_done(success);
            }
        });
}

}  // namespace rtp_llm
