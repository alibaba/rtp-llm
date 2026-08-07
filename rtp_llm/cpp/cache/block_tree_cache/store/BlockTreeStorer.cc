#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeStorer::BlockTreeStorer(BlockTree*                     tree,
                                 BlockTreeEvictor&              evictor,
                                 BlockTransferDispatcher*       transfer_dispatcher,
                                 BlockTreeTaskPool*             task_pool,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 std::mutex&                    mutex,
                                 int                            host_timeout_ms,
                                 int                            disk_timeout_ms,
                                 SettledFn                      settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    settled_(std::move(settled)) {}

void BlockTreeStorer::stopAdmissionLocked() {
    stopping_.store(true);
}

void BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                  const std::vector<std::vector<GroupSetResource>>& resources,
                                  Tier                                              target_tier) {
    if (target_tier == Tier::DEVICE) {
        publishDeviceLocked(cache_keys, resources);
        return;
    }
    if (target_tier == Tier::HOST || target_tier == Tier::DISK) {
        submitLowerTierLocked(cache_keys, resources, target_tier);
        return;
    }
    RTP_LLM_FAIL("store reached an unsupported target tier=%s", tierName(target_tier));
}

void BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources) {
    const BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources);
    if (insert_result.inserted_nodes.empty() && insert_result.adopted_nodes.empty()) {
        return;
    }
    evictor_.onInsertCommitted(insert_result);
    settled_(true, true);
}

void BlockTreeStorer::submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            Tier                                              target_tier) {
    if (stopping_.load()) {
        return;
    }

    const std::vector<GroupSetPtr>& group_sets = tree_->groupSets();
    StoreTaskPtr                    task       = std::make_shared<StoreTask>();
    task->target_tier                          = target_tier;
    task->cache_keys                           = cache_keys;

    block_tree_cache_detail::ScopeRollback prepare_guard([this, &task]() { settleLocked(*task, /*publish=*/false); });

    for (size_t key_index = 0; key_index < cache_keys.size(); ++key_index) {
        for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
            const GroupSetResource& source = resources[key_index][group_set_id];
            if (!source.hasCompleteDeviceValue()) {
                continue;
            }

            const GroupSetPtr& group_set = group_sets[group_set_id];
            StoreTask::Entry   entry;
            entry.key_index            = key_index;
            entry.group_set_id         = group_set_id;
            entry.source_device_blocks = source.device_blocks;
            entry.target_block         = group_set->allocateSingleBlock(target_tier, BlockRefType::STORE);
            if (isNullBlockIdx(entry.target_block)) {
                RTP_LLM_LOG_WARNING(
                    "store aborted: %s pool exhausted for group_set[%zu]", tierName(target_tier), group_set_id);
                return;
            }

            const MultiNodeResource source_holder{group_set_id, Tier::DEVICE, {{nullptr, entry.source_device_blocks}}};
            bool                    source_referenced = false;
            block_tree_cache_detail::ScopeRollback entry_guard(
                [&group_set, &entry, &source_holder, &source_referenced, target_tier]() {
                    if (source_referenced) {
                        group_set->unreferenceBlocks(source_holder, BlockRefType::STORE);
                    }
                    group_set->releaseSingleBlock(target_tier, entry.target_block, BlockRefType::STORE);
                });

            group_set->referenceBlocks(source_holder, BlockRefType::STORE);
            source_referenced = true;

            task->descriptors.push_back(
                target_tier == Tier::HOST ?
                    TransferDescriptor::deviceToHost(group_set_id, entry.source_device_blocks, entry.target_block) :
                    TransferDescriptor::deviceToDisk(group_set_id, entry.source_device_blocks, entry.target_block));
            task->entries.push_back(entry);
            entry_guard.dismiss();
        }
    }

    if (task->entries.empty()) {
        return;
    }

    // Publishing beyond the last key that carries data would create empty tree nodes.
    task->cache_keys.resize(task->entries.back().key_index + 1);

    if (!task_pool_->submit([this, task]() { runStoreTask(task); })) {
        RTP_LLM_LOG_WARNING(
            "store aborted: cache task queue full, target=%s blocks=%zu", tierName(target_tier), task->entries.size());
        return;
    }
    prepare_guard.dismiss();
}

void BlockTreeStorer::runStoreTask(const StoreTaskPtr& task) {
    bool                                   copy_success = false;
    block_tree_cache_detail::ScopeRollback settle_guard(
        [this, &task, &copy_success]() { settleTask(*task, copy_success); });

    if (stopping_.load()) {
        return;
    }

    const int     timeout_ms = task->target_tier == Tier::DISK ? disk_timeout_ms_ : host_timeout_ms_;
    const int64_t transfer_begin_time_us =
        metrics_reporter_.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, task->target_tier);
    try {
        copy_success = transfer_dispatcher_->executeMultiRank(task->descriptors, timeout_ms);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("store copy threw: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("store copy threw an unknown exception");
    }
    std::vector<BlockTreeTransferBytesSnapshot> transfer_bytes;
    if (copy_success) {
        for (const TransferDescriptor& desc : task->descriptors) {
            metrics_reporter_.accumulateTransferBytes(
                desc, tree_->groupSets()[desc.group_set_id], transfer_bytes);
        }
    }
    metrics_reporter_.reportTransferFinished(CacheTransferOperation::STORE,
                                             Tier::DEVICE,
                                             task->target_tier,
                                             task->entries.size(),
                                             transfer_begin_time_us,
                                             copy_success,
                                             transfer_bytes);
}

void BlockTreeStorer::settleTask(const StoreTask& task, bool copy_success) {
    bool   stopping = false;
    bool   publish  = false;
    size_t accepted = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping = stopping_.load();
        publish  = copy_success && !stopping;
        accepted = settleLocked(task, publish);
    }

    if (publish) {
        metrics_reporter_.reportStorePublish(task.target_tier, accepted, task.entries.size() - accepted);
    }

    if (publish) {
        RTP_LLM_LOG_DEBUG("store published target=%s accepted=%zu duplicate=%zu",
                          tierName(task.target_tier),
                          accepted,
                          task.entries.size() - accepted);
    } else if (stopping) {
        RTP_LLM_LOG_INFO(
            "store rolled back during shutdown, target=%s blocks=%zu", tierName(task.target_tier), task.entries.size());
    } else {
        RTP_LLM_LOG_WARNING(
            "store copy failed, target=%s blocks=%zu; tree unchanged", tierName(task.target_tier), task.entries.size());
    }
}

size_t BlockTreeStorer::settleLocked(const StoreTask& task, bool publish) {
    BlockTreeInsertResult insert_result;
    if (publish) {
        std::vector<std::vector<GroupSetResource>> resources(task.cache_keys.size(),
                                                             std::vector<GroupSetResource>(tree_->groupSets().size()));
        for (const StoreTask::Entry& entry : task.entries) {
            resources[entry.key_index][entry.group_set_id].setBlocks(task.target_tier, {entry.target_block});
        }
        insert_result = tree_->insertNode(task.cache_keys, resources);
    }

    for (const StoreTask::Entry& entry : task.entries) {
        const GroupSetPtr& group_set = tree_->groupSets()[entry.group_set_id];
        // Release the temporary holder before candidate admission.
        group_set->releaseSingleBlock(task.target_tier, entry.target_block, BlockRefType::STORE);
        releaseSourceLocked(*group_set, entry);
    }

    if (insert_result.accepted_resource_count > 0) {
        evictor_.onInsertCommitted(insert_result);
    }
    if (!task.entries.empty()) {
        settled_(insert_result.accepted_resource_count > 0, true);
    }
    return insert_result.accepted_resource_count;
}

void BlockTreeStorer::releaseSourceLocked(const GroupSet& group_set, const StoreTask::Entry& entry) {
    group_set.unreferenceBlocks(
        MultiNodeResource{entry.group_set_id, Tier::DEVICE, {{nullptr, entry.source_device_blocks}}},
        BlockRefType::STORE);
    // Request-owned blocks are candidates only when they also serve a tree node.
    for (size_t member_group_id = 0; member_group_id < entry.source_device_blocks.size(); ++member_group_id) {
        TreeNode* node =
            group_set.findTreeNodeByDeviceBlock(member_group_id, entry.source_device_blocks[member_group_id]);
        if (node != nullptr) {
            evictor_.refreshCandidate(node, entry.group_set_id);
        }
    }
}

}  // namespace rtp_llm
