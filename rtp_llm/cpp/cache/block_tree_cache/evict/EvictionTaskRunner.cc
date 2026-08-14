#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>

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

EvictionTaskResult EvictionTaskRunner::runPerRankTransfer(const EvictionTask& task) const {
    EvictionTaskResult              task_result;
    std::vector<TransferDescriptor> descriptors;
    bool                            transfer_success = buildTransferDescriptors(task, descriptors);
    const auto                      batches =
        transfer_success ? partitionTransferDescriptors(descriptors) : std::vector<std::vector<TransferDescriptor>>{};

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    contexts.reserve(batches.size());
    for (const auto& batch : batches) {
        contexts.push_back(transfer_dispatcher_->executePerRank(batch));
    }
    FusedAsyncContext context(contexts);
    context.waitDone();
    transfer_success = transfer_success && context.success();
    task_result.primary_success = transfer_success;
    task_result.cascade_success.assign(task.cascade_descs.size(), transfer_success);
    return task_result;
}

EvictionTaskResult EvictionTaskRunner::runTransfer(const EvictionTask& task) const {
    if (!transfer_dispatcher_->hasMultiRankEngine()) {
        return runPerRankTransfer(task);
    }

    EvictionTaskResult              task_result;
    std::vector<TransferDescriptor> descriptors;
    const bool                      batch_ready      = buildTransferDescriptors(task, descriptors);
    bool transfer_success = false;
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
    task_result.primary_success = transfer_success;
    task_result.cascade_success.assign(task.cascade_descs.size(), transfer_success);
    return task_result;
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
