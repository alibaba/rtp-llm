#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

EvictionTaskRunner::EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher):
    group_sets_(group_sets), transfer_dispatcher_(transfer_dispatcher) {}

void EvictionTaskRunner::runTransfer(std::shared_ptr<const EvictionTransferTask> task,
                                     BlockTreeCacheMetricsReporter&              metrics_reporter,
                                     EvictionDoneCallback                        on_done) const {
    const auto&               descriptors = task->descriptors();
    const TransferDescriptor& first       = descriptors.front();
    const bool    valid = std::all_of(descriptors.begin(), descriptors.end(), [&first](const TransferDescriptor& desc) {
        return desc.isExecutable() && desc.group_set_id == first.group_set_id && desc.source_tier == first.source_tier
               && desc.target_tier == first.target_tier;
    });
    const int64_t transfer_begin =
        metrics_reporter.reportTransferStarted(CacheTransferOperation::EVICT, first.source_tier, first.target_tier);
    if (!valid) {
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                first.source_tier,
                                                first.target_tier,
                                                descriptors.size(),
                                                transfer_begin,
                                                false,
                                                {},
                                                group_sets_);
        if (on_done) {
            on_done(false);
        }
        return;
    }

    transfer_dispatcher_->runTransfer(
        task->transfer_task,
        [this, task, &metrics_reporter, transfer_begin, on_done = std::move(on_done)](ErrorInfo error) mutable {
            const bool success = error.ok();
            metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                    task->descriptors().front().source_tier,
                                                    task->descriptors().front().target_tier,
                                                    task->descriptors().size(),
                                                    transfer_begin,
                                                    success,
                                                    success ? task->descriptors() : std::vector<TransferDescriptor>{},
                                                    group_sets_);
            if (on_done) {
                on_done(success);
            }
        });
}

}  // namespace rtp_llm
