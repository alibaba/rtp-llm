#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
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
    store_task_runner_(tree_->groupSets()),
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
    } else {
        submitLowerTierLocked(cache_keys, resources, target_tier);
    }
}

void BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources) {
    const BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources, false);
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

    StoreTaskPtr task = std::make_shared<StoreTask>();
    task->target_tier = target_tier;
    task->cache_keys  = cache_keys;

    block_tree_cache_detail::ScopeRollback prepare_guard([this, &task]() { settleLocked(*task, /*publish=*/false); });

    if (!store_task_runner_.prepareTask(*task, resources)) {
        return;
    }
    if (!task_pool_->submit([this, task]() { runStoreTask(task); })) {
        RTP_LLM_LOG_WARNING("store aborted: cache task queue full, target=%s blocks=%zu",
                            tierName(target_tier),
                            task->descriptors.size());
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

    try {
        copy_success = store_task_runner_.runTransfer(
            *task, *transfer_dispatcher_, metrics_reporter_, host_timeout_ms_, disk_timeout_ms_);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("store copy threw: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("store copy threw an unknown exception");
    }
}

void BlockTreeStorer::settleTask(const StoreTask& task, bool copy_success) {
    bool   stopping = false;
    size_t accepted = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping = stopping_.load();
        accepted = settleLocked(task, copy_success && !stopping);
    }

    if (stopping) {
        RTP_LLM_LOG_INFO("store rolled back during shutdown, target=%s blocks=%zu",
                         tierName(task.target_tier),
                         task.descriptors.size());
    } else if (!copy_success) {
        RTP_LLM_LOG_WARNING("store copy failed, target=%s blocks=%zu; tree unchanged",
                            tierName(task.target_tier),
                            task.descriptors.size());
    } else {
        metrics_reporter_.reportStorePublish(task.target_tier, accepted, task.descriptors.size() - accepted);
        RTP_LLM_LOG_DEBUG("store published target=%s accepted=%zu duplicate=%zu",
                          tierName(task.target_tier),
                          accepted,
                          task.descriptors.size() - accepted);
    }
}

size_t BlockTreeStorer::settleLocked(const StoreTask& task, bool publish) {
    BlockTreeInsertResult insert_result;
    std::vector<TreeNode*>        fallback_path;
    const std::vector<TreeNode*>* source_path = &insert_result.path;
    if (publish) {
        std::vector<std::vector<GroupSetResource>> resources(task.cache_keys.size(),
                                                             std::vector<GroupSetResource>(tree_->groupSets().size()));
        for (const TransferDescriptor& descriptor : task.descriptors) {
            resources[descriptor.path_index][descriptor.group_set_id].setBlocks(
                task.target_tier, {descriptor.singleBlockAt(task.target_tier)});
        }
        insert_result = tree_->insertNode(task.cache_keys, resources, true);
    } else {
        fallback_path = tree_->findNode(task.cache_keys);
        source_path   = &fallback_path;
    }

    // Release temporary holders before candidate admission.
    store_task_runner_.releaseTaskResources(task);
    for (const TransferDescriptor& descriptor : task.descriptors) {
        if (descriptor.path_index >= source_path->size()) {
            continue;
        }
        TreeNode*               node     = (*source_path)[descriptor.path_index];
        const GroupSetResource& resource = node->group_set_resources[descriptor.group_set_id];
        if (!resource.device_blocks.empty() && resource.device_blocks[0] == descriptor.source_blocks[0]) {
            evictor_.refreshCandidate(node, descriptor.group_set_id);
        }
    }

    if (insert_result.accepted_resource_count > 0) {
        evictor_.onInsertCommitted(insert_result);
    }
    if (!task.descriptors.empty()) {
        settled_(insert_result.accepted_resource_count > 0, true);
    }
    return insert_result.accepted_resource_count;
}

}  // namespace rtp_llm
