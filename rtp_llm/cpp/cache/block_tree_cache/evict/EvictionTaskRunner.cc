#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>

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

EvictionTaskResult EvictionTaskRunner::runTransfer(const EvictionTask&            task,
                                                   BlockTreeCacheMetricsReporter& metrics_reporter) const {
    EvictionTaskResult     task_result;
    BlockTreeTransferBytes transfer_bytes;
    int64_t                transfer_begin_time_us = 0;
    bool                   transfer_started       = false;
    const auto             finish_metrics         = [&]() {
        if (!transfer_started) {
            return;
        }
        transfer_started = false;
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                task.primary_desc.source_tier,
                                                task.primary_desc.target_tier,
                                                task.cascade_descs.size() + 1,
                                                transfer_begin_time_us,
                                                task_result.primary_success,
                                                transfer_bytes);
    };

    try {
        transfer_begin_time_us = metrics_reporter.reportTransferStarted(
            CacheTransferOperation::EVICT, task.primary_desc.source_tier, task.primary_desc.target_tier);
        transfer_started = true;

        std::vector<TransferDescriptor> descriptors;
        const bool                      batch_ready      = buildTransferDescriptors(task, descriptors);
        bool                            transfer_success = false;
        if (batch_ready) {
            const auto                                 batches = partitionTransferDescriptors(descriptors);
            std::vector<std::shared_ptr<AsyncContext>> contexts;
            contexts.reserve(batches.size());
            for (const auto& batch : batches) {
                contexts.push_back(transfer_dispatcher_->executeMultiRank(
                    batch, selectTransferTimeoutMs(task, memory_timeout_ms_, disk_timeout_ms_)));
            }
            FusedAsyncContext context(contexts);
            context.waitDone();
            transfer_success = context.success();
        }
        if (transfer_success) {
            metrics_reporter.accumulateTransferBytes(descriptors, group_sets_, transfer_bytes);
        }
        task_result.primary_success = transfer_success;
        task_result.cascade_success.assign(task.cascade_descs.size(), transfer_success);
        finish_metrics();
        return task_result;
    } catch (...) {
        finish_metrics();
        throw;
    }
}

std::vector<std::vector<TransferDescriptor>>
EvictionTaskRunner::partitionTransferDescriptors(const std::vector<TransferDescriptor>& descriptors) const {
    const auto disk_pool = [this](const TransferDescriptor& descriptor) {
        if (descriptor.source_tier != Tier::DISK && descriptor.target_tier != Tier::DISK) {
            return static_cast<BlockTreeDiskBlockPool*>(nullptr);
        }
        return group_sets_[descriptor.group_set_id]->diskPool().get();
    };
    std::vector<std::vector<TransferDescriptor>> batches;
    for (const auto& descriptor : descriptors) {
        auto batch = std::find_if(batches.begin(), batches.end(), [&](const auto& current) {
            return current.front().source_tier == descriptor.source_tier
                && current.front().target_tier == descriptor.target_tier
                && disk_pool(current.front()) == disk_pool(descriptor);
        });
        if (batch == batches.end()) {
            batches.push_back({descriptor});
        } else {
            batch->push_back(descriptor);
        }
    }
    return batches;
}

bool EvictionTaskRunner::buildTransferDescriptors(const EvictionTask&              task,
                                                  std::vector<TransferDescriptor>& descriptors) {
    descriptors.clear();
    descriptors.reserve(1 + task.cascade_descs.size());

    if (!task.primary_desc.isExecutable()) {
        return false;
    }
    descriptors.push_back(task.primary_desc);

    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (!cascade_desc.isExecutable()) {
            descriptors.clear();
            return false;
        }
        descriptors.push_back(cascade_desc);
    }
    return true;
}

int EvictionTaskRunner::selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms) {
    bool uses_disk = task.primary_desc.source_tier == Tier::DISK || task.primary_desc.target_tier == Tier::DISK;
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (cascade_desc.source_tier == Tier::DISK || cascade_desc.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    return uses_disk ? disk_timeout_ms : memory_timeout_ms;
}

}  // namespace rtp_llm
