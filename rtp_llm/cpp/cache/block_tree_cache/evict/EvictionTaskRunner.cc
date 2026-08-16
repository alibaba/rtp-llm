#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

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
    bool                   overall_success        = false;
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
                                                overall_success,
                                                transfer_bytes);
    };

    try {
        transfer_begin_time_us = metrics_reporter.reportTransferStarted(
            CacheTransferOperation::EVICT, task.primary_desc.source_tier, task.primary_desc.target_tier);
        transfer_started = true;

        if (!validateTransferDescriptors(task)) {
            task_result.cascade_success.assign(task.cascade_descs.size(), false);
            finish_metrics();
            return task_result;
        }

        const int transfer_timeout_ms = selectTransferTimeoutMs(task, memory_timeout_ms_, disk_timeout_ms_);
        auto      primary_context =
            transfer_dispatcher_->executeMultiRank({task.primary_desc}, transfer_timeout_ms);
        primary_context->waitDone();
        task_result.primary_success = primary_context->success();
        if (task_result.primary_success) {
            metrics_reporter.accumulateTransferBytes({task.primary_desc}, group_sets_, transfer_bytes);
        } else {
            task_result.cascade_success.assign(task.cascade_descs.size(), false);
            finish_metrics();
            return task_result;
        }

        overall_success = true;
        task_result.cascade_success.reserve(task.cascade_descs.size());
        for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
            auto cascade_context = transfer_dispatcher_->executeMultiRank({cascade_desc}, transfer_timeout_ms);
            cascade_context->waitDone();
            const bool cascade_success = cascade_context->success();
            task_result.cascade_success.push_back(cascade_success);
            overall_success = overall_success && cascade_success;
            if (cascade_success) {
                metrics_reporter.accumulateTransferBytes({cascade_desc}, group_sets_, transfer_bytes);
            }
        }
        finish_metrics();
        return task_result;
    } catch (...) {
        finish_metrics();
        throw;
    }
}

bool EvictionTaskRunner::validateTransferDescriptors(const EvictionTask& task) {
    if (!task.primary_desc.isExecutable()) {
        return false;
    }
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (!cascade_desc.isExecutable()) {
            return false;
        }
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
