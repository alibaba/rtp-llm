#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
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
            const BlockIdxType target_block = group_set->allocateSingleBlock(task.target_tier, BlockRefType::STORE);
            if (isNullBlockIdx(target_block)) {
                RTP_LLM_LOG_WARNING(
                    "store aborted: %s pool exhausted for group_set[%zu]", tierName(task.target_tier), group_set_id);
                return false;
            }
            group_set->referenceBlocks({group_set_id, Tier::DEVICE, {{nullptr, source.device_blocks}}}, BlockRefType::STORE);
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

bool StoreTaskRunner::runTransfer(Task&                          task,
                                  const BlockTransferDispatcher& transfer_dispatcher,
                                  BlockTreeCacheMetricsReporter& metrics_reporter,
                                  int                            host_timeout_ms,
                                  int                            disk_timeout_ms) {
    const int     timeout_ms = task.target_tier == Tier::DISK ? disk_timeout_ms : host_timeout_ms;
    const int64_t transfer_begin_time_us =
        metrics_reporter.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, task.target_tier);
    const std::vector<TransferDescriptor> empty_descriptors;
    bool                                  copy_success   = false;
    const auto                            finish_metrics = [&]() {
        metrics_reporter.reportTransferFinished(CacheTransferOperation::STORE,
                                                Tier::DEVICE,
                                                task.target_tier,
                                                task.descriptors.size(),
                                                transfer_begin_time_us,
                                                copy_success,
                                                copy_success ? task.descriptors : empty_descriptors,
                                                group_sets_);
    };

    try {
        std::vector<std::shared_ptr<AsyncContext>> contexts;
        if (task.target_tier == Tier::DISK) {
            contexts.reserve(task.descriptors.size());
            for (const auto& descriptor : task.descriptors) {
                contexts.push_back(transfer_dispatcher.executeMultiRank({descriptor}, timeout_ms));
            }
        } else {
            contexts.push_back(transfer_dispatcher.executeMultiRank(task.descriptors, timeout_ms));
        }
        FusedAsyncContext context(contexts);
        context.waitDone();
        copy_success = context.success();
        finish_metrics();
        return copy_success;
    } catch (...) {
        finish_metrics();
        throw;
    }
}

void StoreTaskRunner::releaseTaskResources(const Task& task) {
    for (const TransferDescriptor& descriptor : task.descriptors) {
        const GroupSetPtr& group_set = group_sets_[descriptor.group_set_id];
        group_set->releaseSingleBlock(
            task.target_tier, descriptor.singleBlockAt(task.target_tier), BlockRefType::STORE);
        group_set->unreferenceBlocks(
            MultiNodeResource{descriptor.group_set_id, Tier::DEVICE, {{nullptr, descriptor.source_blocks}}},
            BlockRefType::STORE);
    }
}

}  // namespace rtp_llm
