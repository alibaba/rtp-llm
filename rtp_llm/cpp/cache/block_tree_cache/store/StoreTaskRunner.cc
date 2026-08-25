#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"

#include <exception>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

StoreTaskRunner::StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets): group_sets_(group_sets) {}

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
            task.descriptors.push_back(std::move(descriptor));
        }
    }
    return true;
}

void StoreTaskRunner::runTransfer(TaskPtr                         task,
                                  const BlockTransferDispatcher& transfer_dispatcher,
                                  BlockTreeCacheMetricsReporter& metrics_reporter,
                                  int                            host_timeout_ms,
                                  int                            disk_timeout_ms,
                                  TransferDoneCallback           callback) {
    try {
        const int timeout_ms = task->target_tier == Tier::DISK ? disk_timeout_ms : host_timeout_ms;
        task->phase = Task::Phase::TRANSFERRING;
        task->transfer_begin_time_us =
            metrics_reporter.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, task->target_tier);

        auto stage_state = std::make_shared<TransferStageState>(
            [this, task, &metrics_reporter, callback](ErrorInfo error) mutable {
                try {
                    static const std::vector<TransferDescriptor> empty_descriptors;
                    metrics_reporter.reportTransferFinished(CacheTransferOperation::STORE,
                                                            Tier::DEVICE,
                                                            task->target_tier,
                                                            task->descriptors.size(),
                                                            task->transfer_begin_time_us,
                                                            error.ok(),
                                                            error.ok() ? task->descriptors : empty_descriptors,
                                                            group_sets_);
                    task->phase = Task::Phase::FINISHED;
                    callback(std::move(error));
                } catch (const std::exception& exception) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()));
                } catch (...) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store completion exception"));
                }
            });

        const auto submit_batch = [&](const std::vector<TransferDescriptor>& descriptors) {
            stage_state->addBatch();
            transfer_dispatcher.runTransfer(
                descriptors,
                timeout_ms,
                [stage_state](ErrorInfo error) { stage_state->completeBatch(std::move(error)); });
        };
        if (task->target_tier == Tier::DISK) {
            for (const auto& descriptor : task->descriptors) {
                submit_batch({descriptor});
            }
        } else if (!task->descriptors.empty()) {
            submit_batch(task->descriptors);
        }
        stage_state->finishSubmitting();
    } catch (const std::exception& error) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store submission exception"));
    }
}

void StoreTaskRunner::releaseTaskResources(const Task& task) {
    for (const TransferDescriptor& descriptor : task.descriptors) {
        const GroupSetPtr& group_set = group_sets_[descriptor.group_set_id];
        group_set->releaseSingleBlock(
            task.target_tier, descriptor.singleBlockAt(task.target_tier), BlockTreeRefType::STORE);
        group_set->unreferenceBlocks(
            MultiNodeResource{descriptor.group_set_id, Tier::DEVICE, {{nullptr, descriptor.source_blocks}}},
            BlockTreeRefType::STORE);
    }
}

}  // namespace rtp_llm
