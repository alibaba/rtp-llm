#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackWorker.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LoadBackAsyncContext::requestCancel() {
    State expected = State::PENDING;
    if (state_.compare_exchange_strong(expected, State::CANCEL_REQUESTED)) {
        return true;
    }
    return expected == State::CANCEL_REQUESTED;
}

bool LoadBackAsyncContext::cancelRequested() const {
    return state_.load() == State::CANCEL_REQUESTED;
}

void LoadBackAsyncContext::onTaskComplete(bool ok) {
    State expected = State::PENDING;
    State terminal = ok ? State::SUCCEEDED : State::FAILED;
    if (!state_.compare_exchange_strong(expected, terminal)) {
        if (expected == State::CANCEL_REQUESTED) {
            state_.store(State::CANCELLED);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
}

void LoadBackAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done(); });
}

bool LoadBackAsyncContext::done() const {
    const State state = state_.load();
    return state == State::SUCCEEDED || state == State::FAILED || state == State::CANCELLED;
}

bool LoadBackAsyncContext::success() const {
    return state_.load() == State::SUCCEEDED;
}

LoadBackWorker::TaskPtr
LoadBackWorker::createTask(const LoadBackTicket::PendingLoadBackItems& items,
                           const std::vector<ComponentGroupPtr>&       item_groups) {
    if (items.size() != item_groups.size()) {
        RTP_LLM_LOG_ERROR(
            "load-back task group count mismatch, items=%zu groups=%zu", items.size(), item_groups.size());
        return nullptr;
    }

    TaskPtr task = std::make_shared<Task>();
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item = items[item_index];
        if (item_groups[item_index] == nullptr
            || item_groups[item_index]->component_group_id != item.group_id) {
            RTP_LLM_LOG_ERROR("invalid load-back task group, group=%d index=%zu", item.group_id, item_index);
            return nullptr;
        }
    }
    task->items       = items;
    task->item_groups = item_groups;
    task->staging_host_blocks.assign(items.size(), NULL_BLOCK_IDX);
    task->target_installed.assign(items.size(), false);
    task->context = std::make_shared<LoadBackAsyncContext>();
    return task;
}

LoadBackWorker::PrepareStatus LoadBackWorker::prepareTransferItem(Task& task, size_t item_index) {
    if (item_index >= task.items.size() || item_index >= task.item_groups.size()) {
        RTP_LLM_LOG_WARNING("invalid load-back item index, index=%zu count=%zu", item_index, task.items.size());
        return PrepareStatus::FAILED;
    }

    const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
    const ComponentGroupPtr&                   group = task.item_groups[item_index];
    if (group == nullptr || item.group_id < 0 || group->component_group_id != item.group_id) {
        RTP_LLM_LOG_WARNING("invalid group id, group=%d", item.group_id);
        return PrepareStatus::FAILED;
    }
    if (item.target_device_blocks.empty()) {
        RTP_LLM_LOG_WARNING("invalid item, group=%d", item.group_id);
        return PrepareStatus::FAILED;
    }
    if (item.source_tier == Tier::DEVICE) {
        if (item.source_blocks.empty() || item.source_blocks != item.target_device_blocks) {
            RTP_LLM_LOG_WARNING("resident identity changed, group=%d", item.group_id);
            return PrepareStatus::FAILED;
        }
        return PrepareStatus::READY;
    }
    if (item.node == nullptr) {
        RTP_LLM_LOG_WARNING("invalid copy item node, group=%d", item.group_id);
        return PrepareStatus::FAILED;
    }
    if ((item.source_tier != Tier::HOST && item.source_tier != Tier::DISK) || item.source_blocks.size() != 1) {
        RTP_LLM_LOG_WARNING("invalid copy item, group=%d source=%s", item.group_id, tierName(item.source_tier));
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
            TransferDescriptor::diskToHost(item.group_id, item.source_blocks[0], source_host_block));
    }

    if (isNullBlockIdx(source_host_block)) {
        RTP_LLM_LOG_WARNING(
            "failed to prepare source, group=%d source=%s", item.group_id, tierName(item.source_tier));
        return PrepareStatus::FAILED;
    }
    task.host_to_device_descriptors.push_back(
        TransferDescriptor::hostToDevice(item.group_id, source_host_block, item.target_device_blocks));
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
    if (host_transfer_blocks > 0) {
        host_transfer_begin_time_us = metrics_reporter.reportTransferStarted(Tier::HOST, Tier::DEVICE);
    }
    if (disk_transfer_blocks > 0) {
        disk_transfer_begin_time_us = metrics_reporter.reportTransferStarted(Tier::DISK, Tier::DEVICE);
    }

    bool copy_success = prepared;
    if (copy_success) {
        copy_success = transfer_dispatcher.executeMultiRank(task.disk_to_host_descriptors, disk_timeout_ms);
    }
    if (copy_success) {
        copy_success = transfer_dispatcher.executeMultiRank(task.host_to_device_descriptors, host_timeout_ms);
    }
    if (host_transfer_blocks > 0) {
        metrics_reporter.reportTransferFinished(
            Tier::HOST, Tier::DEVICE, host_transfer_blocks, host_transfer_begin_time_us, copy_success);
    }
    if (disk_transfer_blocks > 0) {
        metrics_reporter.reportTransferFinished(
            Tier::DISK, Tier::DEVICE, disk_transfer_blocks, disk_transfer_begin_time_us, copy_success);
    }

    releaseStagingBlocks(task);
    return copy_success;
}

void LoadBackWorker::completeTask(Task& task, bool success) {
    releaseUninstalledTargetHolders(task);
    if (task.context == nullptr) {
        RTP_LLM_LOG_ERROR("load-back task context is null");
        return;
    }
    task.context->onTaskComplete(success);
}

void LoadBackWorker::releaseStagingBlocks(Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const ComponentGroupPtr& group = task.item_groups[item_index];
        if (group != nullptr && !isNullBlockIdx(task.staging_host_blocks[item_index])) {
            group->releaseSingleBlock(Tier::HOST, task.staging_host_blocks[item_index], BlockRefType::REQUEST);
            task.staging_host_blocks[item_index] = NULL_BLOCK_IDX;
        }
    }
}

void LoadBackWorker::releaseUninstalledTargetHolders(const Task& task) {
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
        const ComponentGroupPtr&                   group = task.item_groups[item_index];
        if (item.source_tier == Tier::DEVICE || task.target_installed[item_index] || group == nullptr) {
            continue;
        }
        group->unreferenceBlocks(
            GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
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

}  // namespace rtp_llm
