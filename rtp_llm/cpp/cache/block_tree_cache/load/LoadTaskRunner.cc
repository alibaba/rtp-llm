#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

LoadTaskRunner::TaskPtr LoadTaskRunner::createTask(const std::vector<TransferDescriptor>&   load_descs,
                                                   const std::vector<bool>&                 joined_load,
                                                   const std::vector<GroupSetPtr>&          group_sets,
                                                   const std::shared_ptr<LoadAsyncContext>& context) {
    std::vector<TransferDescriptor> task_load_descs;
    std::vector<GroupSetPtr>        task_desc_group_sets;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc      = load_descs[desc_index];
        const GroupSetPtr&        group_set = group_sets[desc.group_set_id];
        if (joined_load[desc_index]) {
            continue;
        }
        task_load_descs.push_back(desc);
        task_desc_group_sets.push_back(group_set);
    }
    if (task_load_descs.empty()) {
        return nullptr;
    }

    TaskPtr task          = std::make_shared<Task>();
    task->load_descs      = std::move(task_load_descs);
    task->desc_group_sets = std::move(task_desc_group_sets);
    task->target_installed.assign(task->load_descs.size(), false);
    task->context = context;
    return task;
}

bool LoadTaskRunner::prepareTransferDescriptor(Task& task, size_t desc_index) {
    const TransferDescriptor& desc      = task.load_descs[desc_index];
    const GroupSetPtr&        group_set = task.desc_group_sets[desc_index];
    if (desc.source_tier == Tier::DEVICE) {
        return true;
    }

    if (desc.source_tier == Tier::HOST) {
        if (group_set->hostPool() == nullptr) {
            RTP_LLM_LOG_WARNING("host load without host pool, group_set=%zu", desc.group_set_id);
            return false;
        }
        task.host_to_device_descriptors.push_back(desc);
        return true;
    }

    if (group_set->diskPool() == nullptr) {
        RTP_LLM_LOG_WARNING("disk load without disk pool, group_set=%zu", desc.group_set_id);
        return false;
    }
    task.disk_to_device_descriptors.push_back(desc);
    return true;
}

bool LoadTaskRunner::runTransfer(Task&                          task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms,
                                 bool                           prepared) {
    size_t host_transfer_blocks = 0;
    size_t disk_transfer_blocks = 0;
    for (const TransferDescriptor& desc : task.load_descs) {
        if (desc.source_tier == Tier::HOST) {
            ++host_transfer_blocks;
        } else if (desc.source_tier == Tier::DISK) {
            ++disk_transfer_blocks;
        }
    }

    int64_t    host_transfer_begin_time_us = 0;
    int64_t    disk_transfer_begin_time_us = 0;
    bool       host_transfer_started       = false;
    bool       disk_transfer_started       = false;
    const auto finish_metrics              = [&](bool success) {
        if (host_transfer_started) {
            host_transfer_started = false;
            metrics_reporter.reportTransferFinished(CacheTransferOperation::LOAD,
                                                    Tier::HOST,
                                                    Tier::DEVICE,
                                                    host_transfer_blocks,
                                                    host_transfer_begin_time_us,
                                                    success);
        }
        if (disk_transfer_started) {
            disk_transfer_started = false;
            metrics_reporter.reportTransferFinished(CacheTransferOperation::LOAD,
                                                    Tier::DISK,
                                                    Tier::DEVICE,
                                                    disk_transfer_blocks,
                                                    disk_transfer_begin_time_us,
                                                    success);
        }
    };

    try {
        if (host_transfer_blocks > 0) {
            host_transfer_begin_time_us =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::HOST, Tier::DEVICE);
            host_transfer_started = true;
        }
        if (disk_transfer_blocks > 0) {
            disk_transfer_begin_time_us =
                metrics_reporter.reportTransferStarted(CacheTransferOperation::LOAD, Tier::DISK, Tier::DEVICE);
            disk_transfer_started = true;
        }

        bool copy_success = prepared;
        if (copy_success) {
            copy_success = transfer_dispatcher.executeMultiRank(task.host_to_device_descriptors, host_timeout_ms);
        }
        if (copy_success) {
            copy_success = transfer_dispatcher.executeMultiRank(task.disk_to_device_descriptors, disk_timeout_ms);
        }
        finish_metrics(copy_success);
        return copy_success;
    } catch (...) {
        finish_metrics(false);
        throw;
    }
}

void LoadTaskRunner::releaseTaskResources(const Task& task) {
    releaseUninstalledTargetHolders(task);
}

void LoadTaskRunner::releaseUninstalledTargetHolders(const Task& task) {
    for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc      = task.load_descs[desc_index];
        const GroupSetPtr&        group_set = task.desc_group_sets[desc_index];
        if (desc.source_tier == Tier::DEVICE || task.target_installed[desc_index]) {
            continue;
        }
        group_set->unreferenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockRefType::REQUEST);
    }
}

}  // namespace rtp_llm
