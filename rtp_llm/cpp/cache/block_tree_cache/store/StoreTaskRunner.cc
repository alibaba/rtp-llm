#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <exception>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void StoreTaskRunner::Task::recordMockSubmission(BlockTreeCacheMetricsReporter& metrics_reporter) {
    if (!(mock_memory_cache && target_tier == Tier::HOST) && !(mock_disk_cache && target_tier == Tier::DISK)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mock_outcome_mutex);
    if (!mock_submission_recorded) {
        if (target_tier == Tier::DISK)
            metrics_reporter.recordMockDiskWriteSubmitted(descriptors().size());
        else
            metrics_reporter.recordMockMemoryD2HSubmitted(descriptors().size());
        mock_submission_recorded = true;
    }
}

void StoreTaskRunner::Task::recordMockTransferOutcome(BlockTreeCacheMetricsReporter& metrics_reporter, bool success) {
    if (!(mock_memory_cache && target_tier == Tier::HOST) && !(mock_disk_cache && target_tier == Tier::DISK)) {
        return;
    }
    // A callback proves acceptance, even when an already-expired timeout runs
    // synchronously before submit() returns. Publish submission before outcome.
    recordMockSubmission(metrics_reporter);
    std::lock_guard<std::mutex> lock(mock_outcome_mutex);
    if (!mock_outcome_recorded) {
        if (target_tier == Tier::DISK)
            metrics_reporter.recordMockDiskWriteFinished(descriptors().size(), success);
        else
            metrics_reporter.recordMockMemoryD2HFinished(descriptors().size(), success);
        mock_outcome_recorded = true;
    }
}

StoreTaskRunner::StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets, BlockTree* pressure_tree):
    pressure_tree_(pressure_tree), group_sets_(group_sets) {}

bool StoreTaskRunner::prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources) {
    for (size_t key_index = 0; key_index < task.cache_keys.size(); ++key_index) {
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& source = resources[key_index][group_set_id];
            if (source.device_blocks.empty()) {
                continue;
            }
            const GroupSetPtr& group_set    = group_sets_[group_set_id];
            const BlockIdxType target_block = group_set->allocateSingleBlock(task.target_tier, BlockTreeRefType::STORE);
            if (isNullBlockIdx(target_block)) {
                RTP_LLM_LOG_WARNING(
                    "store aborted: %s pool exhausted for group_set[%zu]", tierName(task.target_tier), group_set_id);
                return false;
            }
            group_set->referenceBlocks({group_set_id, Tier::DEVICE, {{nullptr, source.device_blocks}}},
                                       BlockTreeRefType::STORE);
            TransferDescriptor descriptor =
                task.target_tier == Tier::HOST ?
                    TransferDescriptor::deviceToHost(group_set_id, source.device_blocks, target_block) :
                    TransferDescriptor::deviceToDisk(group_set_id, source.device_blocks, target_block);
            descriptor.path_index = key_index;
            task.transfer_task.addDescriptor(std::move(descriptor));
        }
    }
    return true;
}

void StoreTaskRunner::freezeUnconfirmedEndpoints(const Task& task, const TransferResult& result) const {
    // A peer may still be using these slots. Convert the temporary ownership
    // into a permanent pool quarantine before queuing any STORE ref release.
    if (result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
        for (size_t i = 0; i < task.descriptors().size(); ++i) {
            if (!result.completionUnknown(i))
                continue;
            const auto& desc  = task.descriptors()[i];
            const auto& group = group_sets_[desc.group_set_id];
            for (size_t member = 0; member < desc.source_blocks.size(); ++member) {
                group->devicePools()[member]->freezeBlock(desc.source_blocks[member]);
            }
            IBlockPool* destination = desc.target_tier == Tier::HOST ?
                                          static_cast<IBlockPool*>(group->hostPool().get()) :
                                          static_cast<IBlockPool*>(group->diskPool().get());
            destination->freezeBlock(desc.singleBlockAt(desc.target_tier));
        }
    }
}

void StoreTaskRunner::runTransfer(TaskPtr                        task,
                                  const BlockTransferDispatcher& transfer_dispatcher,
                                  BlockTreeCacheMetricsReporter& metrics_reporter,
                                  TransferDoneCallback           callback) {
    try {
        task->phase = Task::Phase::TRANSFERRING;
        const int64_t transfer_begin =
            metrics_reporter.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, task->target_tier);

        auto complete = [this, task, &metrics_reporter, transfer_begin, callback](TransferResult result) mutable {
            freezeUnconfirmedEndpoints(*task, result);
            task->recordMockTransferOutcome(metrics_reporter, result.ok());
            try {
                if (pressure_tree_) {
                    pressure_tree_->recordPressureWorkflowFailure(
                        CacheTransferOperation::STORE, task->descriptors(), result);
                }
                static const std::vector<TransferDescriptor> empty_descriptors;
                metrics_reporter.reportTransferFinished(CacheTransferOperation::STORE,
                                                        Tier::DEVICE,
                                                        task->target_tier,
                                                        task->descriptors().size(),
                                                        transfer_begin,
                                                        result.ok(),
                                                        result.ok() ? task->descriptors() : empty_descriptors,
                                                        group_sets_);
                metrics_reporter.maybeReportCopyError(Tier::DEVICE, task->target_tier, result);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_WARNING("store transfer metrics failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("store transfer metrics failed");
            }
            task->phase = Task::Phase::FINISHED;
            callback(std::move(result));
        };
        if (task->descriptors().empty()) {
            complete(TransferResult{});
            return;
        }
        transfer_dispatcher.runTransfer(CacheTransferOperation::STORE, task->transfer_task, std::move(complete));
    } catch (const std::exception& error) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store submission exception"));
    }
}

void StoreTaskRunner::releaseTaskResources(const Task& task) {
    for (const TransferDescriptor& descriptor : task.descriptors()) {
        const GroupSetPtr& group_set = group_sets_[descriptor.group_set_id];
        group_set->releaseSingleBlock(
            task.target_tier, descriptor.singleBlockAt(task.target_tier), BlockTreeRefType::STORE);
        group_set->unreferenceBlocks(
            MultiNodeResource{descriptor.group_set_id, Tier::DEVICE, {{nullptr, descriptor.source_blocks}}},
            BlockTreeRefType::STORE);
    }
}

}  // namespace rtp_llm
