#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <functional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

LoadTaskRunner::TaskPtr LoadTaskRunner::createTask(const LoadAsyncContext::PendingLoadItems& items,
                                                   const std::vector<bool>&                  joined_load,
                                                   const std::vector<GroupSetPtr>&           group_sets,
                                                   const std::shared_ptr<LoadAsyncContext>&  context) {
    LoadAsyncContext::PendingLoadItems task_items;
    std::vector<GroupSetPtr>           task_item_group_sets;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item      = items[item_index];
        const GroupSetPtr&                       group_set = group_sets[item.group_set_id];
        if (joined_load[item_index]) {
            continue;
        }
        task_items.push_back(item);
        task_item_group_sets.push_back(group_set);
    }
    if (task_items.empty()) {
        return nullptr;
    }

    TaskPtr task          = std::make_shared<Task>();
    task->items           = std::move(task_items);
    task->item_group_sets = std::move(task_item_group_sets);
    task->target_installed.assign(task->items.size(), false);
    task->context = context;
    return task;
}

bool LoadTaskRunner::prepareTransferItem(Task& task, size_t item_index) {
    const LoadAsyncContext::PendingLoadItem& item      = task.items[item_index];
    const GroupSetPtr&                       group_set = task.item_group_sets[item_index];
    if (item.source_tier == Tier::DEVICE) {
        return true;
    }

    if (item.source_tier == Tier::HOST) {
        if (group_set->hostPool() == nullptr) {
            RTP_LLM_LOG_WARNING("host load without host pool, group_set=%zu", item.group_set_id);
            return false;
        }
        task.host_to_device_descriptors.push_back(
            TransferDescriptor::hostToDevice(item.group_set_id, item.source_blocks[0], item.target_device_blocks));
        return true;
    }

    if (group_set->diskPool() == nullptr) {
        RTP_LLM_LOG_WARNING("disk load without disk pool, group_set=%zu", item.group_set_id);
        return false;
    }
    task.disk_to_device_descriptors.push_back(
        TransferDescriptor::diskToDevice(item.group_set_id, item.source_blocks[0], item.target_device_blocks));
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
    for (const LoadAsyncContext::PendingLoadItem& item : task.items) {
        if (item.source_tier == Tier::HOST) {
            ++host_transfer_blocks;
        } else if (item.source_tier == Tier::DISK) {
            ++disk_transfer_blocks;
        }
    }

    int64_t                         host_transfer_begin_time_us = 0;
    int64_t                         disk_transfer_begin_time_us = 0;
    bool                            host_transfer_started       = false;
    bool                            disk_transfer_started       = false;
    const std::function<void(bool)> finish_metrics              = [&](bool success) {
        if (host_transfer_started) {
            host_transfer_started = false;
            metrics_reporter.reportTransferFinished(
                Tier::HOST, Tier::DEVICE, host_transfer_blocks, host_transfer_begin_time_us, success);
        }
        if (disk_transfer_started) {
            disk_transfer_started = false;
            metrics_reporter.reportTransferFinished(
                Tier::DISK, Tier::DEVICE, disk_transfer_blocks, disk_transfer_begin_time_us, success);
        }
    };

    try {
        if (host_transfer_blocks > 0) {
            host_transfer_begin_time_us = metrics_reporter.reportTransferStarted(Tier::HOST, Tier::DEVICE);
            host_transfer_started       = true;
        }
        if (disk_transfer_blocks > 0) {
            disk_transfer_begin_time_us = metrics_reporter.reportTransferStarted(Tier::DISK, Tier::DEVICE);
            disk_transfer_started       = true;
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
        try {
            finish_metrics(false);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("failed to finalize load metrics: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to finalize load metrics with unknown exception");
        }
        throw;
    }
}

void LoadTaskRunner::releaseTaskResources(const Task& task) {
    releaseUninstalledTargetHolders(task);
}

void LoadTaskRunner::releaseUninstalledTargetHolders(const Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item      = task.items[item_index];
        const GroupSetPtr&                       group_set = task.item_group_sets[item_index];
        if (item.source_tier == Tier::DEVICE || task.target_installed[item_index]) {
            continue;
        }
        group_set->unreferenceBlocks(MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}},
                                     BlockRefType::REQUEST);
    }
}

}  // namespace rtp_llm
