#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
BlockTreeStorer::BlockTreeStorer(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             host_timeout_ms,
                                 int                             disk_timeout_ms,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    store_task_runner_(tree_->groupSets()),
    mutex_(mutex),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)) {}

void BlockTreeStorer::stopAdmissionLocked() {
    stopping_.store(true);
}

StorageWriteTask BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier                                              target_tier) {
    if (target_tier == Tier::DEVICE) {
        return publishDeviceLocked(cache_keys, resources);
    }
    if (target_tier == Tier::HOST || target_tier == Tier::DISK) {
        submitLowerTierLocked(cache_keys, resources, target_tier);
        return {};
    }
    RTP_LLM_FAIL("unsupported store target tier: %s", tierName(target_tier));
}

StorageWriteTask BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                                      const std::vector<std::vector<GroupSetResource>>& resources) {
    const BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources, false);
    if (!insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty()) {
        evictor_.onInserted(insert_result);
        settled_(true, true);
    }
    return storage_backend_ ? storage_backend_->prepareWrite(makeStorageRequest(cache_keys, resources)) :
                              StorageWriteTask{};
}

StorageRequest BlockTreeStorer::makeStorageRequest(const CacheKeysType&                              cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources) const {
    StorageRequest request{std::make_shared<CacheKeysType>(cache_keys),
                           std::vector<std::vector<StorageBlockHandle>>(cache_keys.size())};
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        auto& key_handles = request.handles[key_index];
        for (size_t group_set = 0; group_set < tree_->groupSets().size(); ++group_set) {
            const auto& resource = resources[key_index][group_set];
            if (!resource.hasCompleteDeviceValue()) {
                continue;
            }
            const auto& group_ids = tree_->groupSets()[group_set]->groupIds();
            for (size_t member = 0; member < group_ids.size(); ++member) {
                key_handles.push_back({group_ids[member], resource.device_blocks[member]});
            }
        }
    }
    return request;
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
    if (publish) {
        std::vector<std::vector<GroupSetResource>> resources(task.cache_keys.size(),
                                                             std::vector<GroupSetResource>(tree_->groupSets().size()));
        for (const TransferDescriptor& descriptor : task.descriptors) {
            resources[descriptor.path_index][descriptor.group_set_id].setBlocks(
                task.target_tier, {descriptor.singleBlockAt(task.target_tier)});
        }
        insert_result = tree_->insertNode(task.cache_keys, resources, true);
    }

    // Publication owns the target through BLOCK_CACHE; temporary STORE holders
    // are no longer needed once every target has been installed.
    store_task_runner_.releaseTaskResources(task);

    if (insert_result.accepted_resource_count > 0) {
        evictor_.onInserted(insert_result);
    }
    if (!task.descriptors.empty()) {
        settled_(insert_result.accepted_resource_count > 0, true);
    }
    return insert_result.accepted_resource_count;
}

}  // namespace rtp_llm
