#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LoadTaskRunner::createTask(const LoadTicket::PendingLoadItems&      items,
                                const std::vector<GroupSetPtr>&          group_sets,
                                const std::shared_ptr<LoadAsyncContext>& context,
                                TaskPtr&                                 task) {
    task.reset();
    if (context == nullptr) {
        RTP_LLM_LOG_ERROR("invalid load task context");
        return false;
    }

    LoadTicket::PendingLoadItems task_items;
    std::vector<GroupSetPtr>     task_item_group_sets;
    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.group_set_id >= group_sets.size()) {
            RTP_LLM_LOG_ERROR("invalid load task group set, group_set_id=%zu", item.group_set_id);
            return false;
        }
        const GroupSetPtr& group_set = group_sets[item.group_set_id];
        if (group_set == nullptr || group_set->groupSetId() != item.group_set_id) {
            RTP_LLM_LOG_ERROR("mismatched load task group set, group_set_id=%zu", item.group_set_id);
            return false;
        }
        if (item.joined_load) {
            continue;
        }
        task_items.push_back(item);
        task_item_group_sets.push_back(group_set);
    }
    if (task_items.empty()) {
        return true;
    }

    task                  = std::make_shared<Task>();
    task->items           = std::move(task_items);
    task->item_group_sets = std::move(task_item_group_sets);
    task->staging_host_blocks.assign(task->items.size(), NULL_BLOCK_IDX);
    task->target_installed.assign(task->items.size(), false);
    task->context = context;
    return true;
}

LoadTaskRunner::PrepareStatus LoadTaskRunner::prepareTransferItem(Task& task, size_t item_index) {
    if (item_index >= task.items.size() || item_index >= task.item_group_sets.size()) {
        RTP_LLM_LOG_WARNING("invalid load item index, index=%zu count=%zu", item_index, task.items.size());
        return PrepareStatus::FAILED;
    }

    const LoadTicket::PendingLoadItem& item      = task.items[item_index];
    const GroupSetPtr&                 group_set = task.item_group_sets[item_index];
    if (group_set == nullptr || group_set->groupSetId() != item.group_set_id) {
        RTP_LLM_LOG_WARNING("invalid group set id, group_set=%zu", item.group_set_id);
        return PrepareStatus::FAILED;
    }
    if (item.target_device_blocks.size() != group_set->devicePoolCount()) {
        RTP_LLM_LOG_WARNING("target block count mismatch, group_set=%zu expected=%zu actual=%zu",
                            item.group_set_id,
                            group_set->devicePoolCount(),
                            item.target_device_blocks.size());
        return PrepareStatus::FAILED;
    }
    if (item.source_tier == Tier::DEVICE) {
        if (item.source_blocks.empty() || item.source_blocks != item.target_device_blocks) {
            RTP_LLM_LOG_WARNING("resident identity changed, group_set=%zu", item.group_set_id);
            return PrepareStatus::FAILED;
        }
        return PrepareStatus::READY;
    }
    if (item.node == nullptr) {
        RTP_LLM_LOG_WARNING("invalid copy item node, group_set=%zu", item.group_set_id);
        return PrepareStatus::FAILED;
    }
    if ((item.source_tier != Tier::HOST && item.source_tier != Tier::DISK) || item.source_blocks.size() != 1) {
        RTP_LLM_LOG_WARNING(
            "invalid copy item, group_set=%zu source=%s", item.group_set_id, tierName(item.source_tier));
        return PrepareStatus::FAILED;
    }

    BlockIdxType source_host_block = NULL_BLOCK_IDX;
    if (item.source_tier == Tier::HOST && group_set->hostPool() != nullptr) {
        source_host_block = item.source_blocks[0];
    } else if (item.source_tier == Tier::DISK && group_set->hostPool() != nullptr && group_set->diskPool() != nullptr) {
        source_host_block = group_set->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
        if (isNullBlockIdx(source_host_block)) {
            return PrepareStatus::NEED_HOST_RECLAIM;
        }
        task.staging_host_blocks[item_index] = source_host_block;
        task.disk_to_host_descriptors.push_back(
            TransferDescriptor::diskToHost(item.group_set_id, item.source_blocks[0], source_host_block));
    }

    if (isNullBlockIdx(source_host_block)) {
        RTP_LLM_LOG_WARNING(
            "failed to prepare source, group_set=%zu source=%s", item.group_set_id, tierName(item.source_tier));
        return PrepareStatus::FAILED;
    }
    task.host_to_device_descriptors.push_back(
        TransferDescriptor::hostToDevice(item.group_set_id, source_host_block, item.target_device_blocks));
    return PrepareStatus::READY;
}

bool LoadTaskRunner::runTransfer(Task&                          task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms,
                                 bool                           prepared) {
    size_t host_transfer_blocks = 0;
    size_t disk_transfer_blocks = 0;
    for (const LoadTicket::PendingLoadItem& item : task.items) {
        if (item.source_tier == Tier::HOST) {
            ++host_transfer_blocks;
        } else if (item.source_tier == Tier::DISK) {
            ++disk_transfer_blocks;
        }
    }

    int64_t host_transfer_begin_time_us = 0;
    int64_t disk_transfer_begin_time_us = 0;
    bool    host_transfer_started       = false;
    bool    disk_transfer_started       = false;
    auto    finish_metrics              = [&](bool success) {
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
            copy_success = transfer_dispatcher.executeMultiRank(task.disk_to_host_descriptors, disk_timeout_ms);
        }
        if (copy_success) {
            copy_success = transfer_dispatcher.executeMultiRank(task.host_to_device_descriptors, host_timeout_ms);
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

void LoadTaskRunner::releaseTaskResources(Task& task) {
    releaseStagingBlocks(task);
    releaseUninstalledTargetHolders(task);
}

void LoadTaskRunner::releaseStagingBlocks(Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const GroupSetPtr& group_set = task.item_group_sets[item_index];
        if (group_set != nullptr && !isNullBlockIdx(task.staging_host_blocks[item_index])) {
            group_set->releaseSingleBlock(Tier::HOST, task.staging_host_blocks[item_index], BlockRefType::REQUEST);
            task.staging_host_blocks[item_index] = NULL_BLOCK_IDX;
        }
    }
}

void LoadTaskRunner::releaseUninstalledTargetHolders(const Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadTicket::PendingLoadItem& item      = task.items[item_index];
        const GroupSetPtr&                 group_set = task.item_group_sets[item_index];
        if (item.source_tier == Tier::DEVICE || task.target_installed[item_index] || group_set == nullptr) {
            continue;
        }
        group_set->unreferenceBlocks(MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}},
                                     BlockRefType::REQUEST);
    }
}

}  // namespace rtp_llm
