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
    EvictionTaskResult task_result;
    int64_t            transfer_begin_time_us = 0;
    bool               transfer_started       = false;
    bool               overall_success        = false;
    const auto         finish_metrics         = [&]() {
        if (!transfer_started) {
            return;
        }
        transfer_started = false;
        std::vector<TransferDescriptor> successful_descriptors;
        successful_descriptors.reserve(task.cascade_descs.size() + 1);
        if (task_result.primary_success) {
            successful_descriptors.push_back(task.primary_desc);
        }
        for (size_t i = 0; i < task_result.cascade_success.size(); ++i) {
            if (task_result.cascade_success[i]) {
                successful_descriptors.push_back(task.cascade_descs[i]);
            }
        }
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                task.primary_desc.source_tier,
                                                task.primary_desc.target_tier,
                                                task.cascade_descs.size() + 1,
                                                transfer_begin_time_us,
                                                overall_success,
                                                successful_descriptors,
                                                group_sets_);
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
        task_result.primary_success = transfer_dispatcher_->runTransfer({task.primary_desc}, transfer_timeout_ms);
        if (!task_result.primary_success) {
            task_result.cascade_success.assign(task.cascade_descs.size(), false);
            finish_metrics();
            return task_result;
        }

        overall_success = true;
        task_result.cascade_success.reserve(task.cascade_descs.size());
        for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
            const bool cascade_success = transfer_dispatcher_->runTransfer({cascade_desc}, transfer_timeout_ms);
            task_result.cascade_success.push_back(cascade_success);
            overall_success = overall_success && cascade_success;
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
