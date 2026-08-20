#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

LoadTaskRunner::LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets): group_sets_(group_sets) {}

LoadTaskRunner::TaskPtr LoadTaskRunner::createTask(const std::shared_ptr<LoadAsyncContext>& context) {
    const std::vector<TransferDescriptor>& load_descs   = context->loadDescs();
    std::vector<TransferDescriptor> task_load_descs;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (context->joinedLoads()[desc_index] || desc.source_tier == Tier::DEVICE) {
            continue;
        }
        task_load_descs.push_back(desc);
    }
    if (task_load_descs.empty()) {
        return nullptr;
    }

    TaskPtr task     = std::make_shared<Task>();
    task->load_descs = std::move(task_load_descs);
    task->target_installed.assign(task->load_descs.size(), false);
    task->context = context;
    return task;
}

bool LoadTaskRunner::runTransfer(Task&                          task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms) {
    std::vector<std::vector<TransferDescriptor>> host_batches(group_sets_.size());
    std::vector<std::vector<TransferDescriptor>> disk_batches(group_sets_.size());
    for (const TransferDescriptor& desc : task.load_descs) {
        if (desc.source_tier == Tier::HOST) {
            task.host_to_device_descriptors.push_back(desc);
            host_batches[desc.group_set_id].push_back(desc);
        } else {
            task.disk_to_device_descriptors.push_back(desc);
            disk_batches[desc.group_set_id].push_back(desc);
        }
    }
    int64_t    host_transfer_begin_time_us = 0;
    int64_t    disk_transfer_begin_time_us = 0;
    bool       host_transfer_started       = false;
    bool       disk_transfer_started       = false;
    const std::vector<TransferDescriptor> empty_descriptors;
    bool                                  host_success = false;
    bool                                  disk_success = false;
    const auto finish_metrics = [&](bool host_success, bool disk_success) {
        if (host_transfer_started) {
            host_transfer_started = false;
            metrics_reporter.reportTransferFinished(CacheTransferOperation::LOAD,
                                                    Tier::HOST,
                                                    Tier::DEVICE,
                                                    task.host_to_device_descriptors.size(),
                                                    host_transfer_begin_time_us,
                                                    host_success,
                                                    host_success ? task.host_to_device_descriptors : empty_descriptors,
                                                    group_sets_);
        }
        if (disk_transfer_started) {
            disk_transfer_started = false;
            metrics_reporter.reportTransferFinished(CacheTransferOperation::LOAD,
                                                    Tier::DISK,
                                                    Tier::DEVICE,
                                                    task.disk_to_device_descriptors.size(),
                                                    disk_transfer_begin_time_us,
                                                    disk_success,
                                                    disk_success ? task.disk_to_device_descriptors : empty_descriptors,
                                                    group_sets_);
        }
    };
    const auto execute_batches = [&](const std::vector<std::vector<TransferDescriptor>>& batches, int timeout_ms) {
        std::vector<std::shared_ptr<AsyncContext>> contexts;
        for (const auto& batch : batches) {
            if (!batch.empty()) {
                contexts.push_back(transfer_dispatcher.executeMultiRank(batch, timeout_ms));
            }
        }
        if (contexts.empty()) {
            return true;
        }
        FusedAsyncContext context(contexts);
        context.waitDone();
        return context.success();
    };

    try {
        if (!task.host_to_device_descriptors.empty()) {
            host_transfer_begin_time_us =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::HOST, Tier::DEVICE);
            host_transfer_started = true;
        }
        host_success = execute_batches(host_batches, host_timeout_ms);
        finish_metrics(host_success, false);
        if (!host_success) {
            return false;
        }

        if (!task.disk_to_device_descriptors.empty()) {
            disk_transfer_begin_time_us =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::DISK, Tier::DEVICE);
            disk_transfer_started = true;
        }
        disk_success = execute_batches(disk_batches, disk_timeout_ms);
        finish_metrics(host_success, disk_success);
        return disk_success;
    } catch (...) {
        finish_metrics(host_success, disk_success);
        throw;
    }
}

void LoadTaskRunner::releaseTaskResources(const Task& task) {
    for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc      = task.load_descs[desc_index];
        const GroupSetPtr&        group_set = group_sets_[desc.group_set_id];
        if (task.target_installed[desc_index]) {
            continue;
        }
        group_set->unreferenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockTreeRefType::LOAD);
    }
}

}  // namespace rtp_llm
