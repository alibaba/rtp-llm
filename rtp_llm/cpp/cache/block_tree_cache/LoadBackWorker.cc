#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackWorker.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LoadBackWorker::createTask(const LoadBackTicket::PendingLoadBackItems&  items,
                                const std::vector<GroupSetPtr>&              group_sets,
                                const std::shared_ptr<LoadBackAsyncContext>& context,
                                TaskPtr&                                     task) {
    task.reset();
    if (context == nullptr) {
        RTP_LLM_LOG_ERROR("invalid load-back task context");
        return false;
    }

    LoadBackTicket::PendingLoadBackItems task_items;
    std::vector<GroupSetPtr>              task_item_groups;
    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.group_set_id >= group_sets.size()) {
            RTP_LLM_LOG_ERROR("invalid load-back task group, group_set=%zu", item.group_set_id);
            return false;
        }
        const GroupSetPtr& group = group_sets[item.group_set_id];
        if (group == nullptr || group->groupSetId() != item.group_set_id) {
            RTP_LLM_LOG_ERROR("mismatched load-back task group, group_set=%zu", item.group_set_id);
            return false;
        }
        if (item.joined_load_back) {
            continue;
        }
        task_items.push_back(item);
        task_item_groups.push_back(group);
    }
    if (task_items.empty()) {
        return true;
    }

    task              = std::make_shared<Task>();
    task->items       = std::move(task_items);
    task->item_groups = std::move(task_item_groups);
    task->staging_host_blocks.assign(task->items.size(), NULL_BLOCK_IDX);
    task->target_installed.assign(task->items.size(), false);
    task->context = context;
    return true;
}

LoadBackWorker::PrepareStatus LoadBackWorker::prepareTransferItem(Task& task, size_t item_index) {
    if (item_index >= task.items.size() || item_index >= task.item_groups.size()) {
        RTP_LLM_LOG_WARNING("invalid load-back item index, index=%zu count=%zu", item_index, task.items.size());
        return PrepareStatus::FAILED;
    }

    const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
    const GroupSetPtr&                         group = task.item_groups[item_index];
    if (group == nullptr || group->groupSetId() != item.group_set_id) {
        RTP_LLM_LOG_WARNING("invalid group set id, group_set=%zu", item.group_set_id);
        return PrepareStatus::FAILED;
    }
    if (item.target_device_blocks.size() != group->devicePoolCount()) {
        RTP_LLM_LOG_WARNING("target block count mismatch, group_set=%zu expected=%zu actual=%zu",
                            item.group_set_id,
                            group->devicePoolCount(),
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
    if (item.source_tier == Tier::HOST && group->hostPool() != nullptr) {
        source_host_block = item.source_blocks[0];
    } else if (item.source_tier == Tier::DISK && group->hostPool() != nullptr && group->diskPool() != nullptr) {
        source_host_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
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

bool LoadBackWorker::runTransfer(Task&                          task,
                                 const BlockTransferDispatcher& transfer_dispatcher,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms,
                                 bool                           prepared) {
    size_t host_transfer_blocks = 0;
    size_t disk_transfer_blocks = 0;
    for (const LoadBackTicket::PendingLoadBackItem& item : task.items) {
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
    auto finish_metrics = [&](bool success) {
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
            RTP_LLM_LOG_ERROR("failed to finalize load-back metrics: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to finalize load-back metrics with unknown exception");
        }
        throw;
    }
}

void LoadBackWorker::releaseTaskResources(Task& task) {
    releaseStagingBlocks(task);
    releaseUninstalledTargetHolders(task);
}

void LoadBackWorker::releaseStagingBlocks(Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const GroupSetPtr& group = task.item_groups[item_index];
        if (group != nullptr && !isNullBlockIdx(task.staging_host_blocks[item_index])) {
            group->releaseSingleBlock(Tier::HOST, task.staging_host_blocks[item_index], BlockRefType::REQUEST);
            task.staging_host_blocks[item_index] = NULL_BLOCK_IDX;
        }
    }
}

void LoadBackWorker::releaseUninstalledTargetHolders(const Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
        const GroupSetPtr&                         group = task.item_groups[item_index];
        if (item.source_tier == Tier::DEVICE || task.target_installed[item_index] || group == nullptr) {
            continue;
        }
        group->unreferenceBlocks(MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}},
                                 BlockRefType::REQUEST);
    }
}

bool LoadBackWorker::cancelLoadBackNolock(const std::shared_ptr<AsyncContext>& context) {
    std::shared_ptr<LoadBackAsyncContext> load_back_context =
        std::dynamic_pointer_cast<LoadBackAsyncContext>(context);
    if (load_back_context == nullptr) {
        RTP_LLM_LOG_WARNING("context is not owned by BlockTreeCache");
        return false;
    }
    return !load_back_context->done() && load_back_context->requestCancel();
}

bool LoadBackWorker::startLoading(TreeNode*                                    node,
                                  size_t                                       group_set_id,
                                  const std::vector<BlockIdxType>&             target_blocks,
                                  const std::shared_ptr<LoadBackAsyncContext>& context) {
    if (node == nullptr || target_blocks.empty() || context == nullptr
        || std::any_of(
            target_blocks.begin(), target_blocks.end(), [](BlockIdxType block) { return isNullBlockIdx(block); })) {
        return false;
    }

    const LoadingKey                                  key{node, group_set_id};
    const std::pair<LoadingRecordMap::iterator, bool> insert_result =
        loading_records_.emplace(key, LoadingRecord{target_blocks, {context}});
    return insert_result.second;
}

std::optional<std::vector<BlockIdxType>>
LoadBackWorker::joinLoading(TreeNode* node,
                            size_t group_set_id,
                            const std::shared_ptr<LoadBackAsyncContext>& context) {
    if (node == nullptr || context == nullptr) {
        return std::nullopt;
    }

    const LoadingKey                 key{node, group_set_id};
    const LoadingRecordMap::iterator record_it = loading_records_.find(key);
    if (record_it == loading_records_.end()) {
        return std::nullopt;
    }

    for (const std::shared_ptr<LoadBackAsyncContext>& registered_context : record_it->second.contexts) {
        if (registered_context == context) {
            return record_it->second.target_blocks;
        }
    }
    record_it->second.contexts.push_back(context);
    return record_it->second.target_blocks;
}

bool LoadBackWorker::finishLoading(TreeNode* node, size_t group_set_id, bool success) {
    if (node == nullptr) {
        return false;
    }

    const LoadingKey                 key{node, group_set_id};
    const LoadingRecordMap::iterator record_it = loading_records_.find(key);
    if (record_it == loading_records_.end()) {
        return false;
    }

    std::vector<std::shared_ptr<LoadBackAsyncContext>> contexts = std::move(record_it->second.contexts);
    loading_records_.erase(record_it);

    bool all_completed = true;
    for (const std::shared_ptr<LoadBackAsyncContext>& context : contexts) {
        if (context == nullptr || !context->completeOne(success)) {
            all_completed = false;
            RTP_LLM_LOG_WARNING("failed to complete joined load-back context, group_set=%zu", group_set_id);
        }
    }
    return all_completed;
}

bool LoadBackWorker::eraseLoadingForOneContext(TreeNode*                                    node,
                                               size_t                                       group_set_id,
                                               const std::shared_ptr<LoadBackAsyncContext>& context) {
    if (node == nullptr || context == nullptr) {
        return false;
    }

    const LoadingKey                 key{node, group_set_id};
    const LoadingRecordMap::iterator record_it = loading_records_.find(key);
    if (record_it == loading_records_.end()) {
        return false;
    }
    const std::vector<std::shared_ptr<LoadBackAsyncContext>>::iterator context_it =
        std::find(record_it->second.contexts.begin(), record_it->second.contexts.end(), context);
    if (context_it == record_it->second.contexts.end()) {
        return false;
    }
    record_it->second.contexts.erase(context_it);
    if (record_it->second.contexts.empty()) {
        loading_records_.erase(record_it);
    }
    return true;
}

}  // namespace rtp_llm
