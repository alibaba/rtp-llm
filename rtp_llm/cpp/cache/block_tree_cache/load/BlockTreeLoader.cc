#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeLoader::BlockTreeLoader(const std::vector<GroupSetPtr>& group_sets,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             disk_timeout_ms,
                                 int                             host_timeout_ms,
                                 bool                            enable_device_cache,
                                 SettledFn                       settled):
    group_sets_(group_sets),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    disk_timeout_ms_(disk_timeout_ms),
    host_timeout_ms_(host_timeout_ms),
    enable_device_cache_(enable_device_cache),
    settled_(std::move(settled)),
    load_context_coordinator_(std::make_shared<LoadContextCoordinator>(
        [this](const std::shared_ptr<LoadAsyncContext>& context) { return commitLoad(context); },
        [this](LoadAsyncContext& context) { abortLoad(context); })) {}

void BlockTreeLoader::prepareLoadLocked(const std::vector<TreeNode*>& matched_path, BlockTreeMatchResult& result) {
    const size_t        matched_block_count = matched_path.size();
    const size_t ready_matched_block_count = result.matched_blocks;
    if (matched_block_count == 0) {
        return;
    }

    LoadAsyncContext::PendingLoadItems pending_load_items;
    std::vector<bool>                  joined_load;
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        const size_t       ready_reuse_count =
            std::min(group_set->computeReuseBlockCount(ready_matched_block_count), ready_matched_block_count);
        const size_t ready_reuse_begin = ready_matched_block_count - ready_reuse_count;
        const size_t logical_reuse_count =
            std::min(group_set->computeReuseBlockCount(matched_block_count), matched_block_count);
        for (size_t i = matched_block_count - logical_reuse_count; i < matched_block_count; ++i) {
            if (i >= ready_reuse_begin && i < ready_matched_block_count) {
                continue;
            }
            TreeNode*         path_node          = matched_path[i];
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            prepareMatchedLoadItem(
                path_node, group_set, group_set_resource, i, result, pending_load_items, joined_load);
        }
    }

    if (!pending_load_items.empty()) {
        result.async_context = prepareLoadContext(pending_load_items, joined_load, matched_block_count);
        if (result.async_context == nullptr) {
            result.load_blocks      = 0;
            result.host_load_blocks = 0;
            result.disk_load_blocks = 0;
        }
    }
    return;
}

bool BlockTreeLoader::cancelLoadLocked(const std::shared_ptr<AsyncContext>& context) {
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(context);
    if (load_context == nullptr) {
        RTP_LLM_LOG_WARNING("context is not owned by BlockTreeCache");
        return false;
    }
    return !load_context->done() && load_context->requestCancel();
}

void BlockTreeLoader::shutdown() {
    load_context_coordinator_->shutdown();
}

void BlockTreeLoader::prepareMatchedLoadItem(TreeNode*                           path_node,
                                             const GroupSetPtr&                  group_set,
                                             const GroupSetResource&             group_set_resource,
                                             size_t                              path_index,
                                             BlockTreeMatchResult&               result,
                                             LoadAsyncContext::PendingLoadItems& pending_load_items,
                                             std::vector<bool>&                  joined_load) {
    // DEMOTING/LOAD_PENDING sources belong to another in-flight operation and
    // can neither be referenced nor joined; skip them like empty resources.
    if (!group_set_resource.isMatchUsable()) {
        RTP_LLM_LOG_DEBUG("skip busy resource for load planning, node_key=%ld group_set=%zu state=%d",
                          path_node->cache_key,
                          group_set->groupSetId(),
                          static_cast<int>(group_set_resource.transfer_state));
        return;
    }
    const Tier source_tier = group_set_resource.getTopTier();
    if (source_tier == Tier::NONE) {
        return;
    }

    const std::vector<BlockIdxType> source_blocks = group_set_resource.getBlocks(source_tier);

    LoadAsyncContext::PendingLoadItem pending_item;
    pending_item.node          = path_node;
    pending_item.group_set_id  = group_set->groupSetId();
    pending_item.path_index    = path_index;
    pending_item.source_tier   = source_tier;
    pending_item.source_blocks = source_blocks;
    pending_load_items.push_back(std::move(pending_item));
    joined_load.push_back(group_set_resource.transfer_state == GroupSetTransferState::LOADING);

    if (source_tier == Tier::HOST) {
        result.host_load_blocks++;
        result.load_blocks++;
    } else if (source_tier == Tier::DISK) {
        result.disk_load_blocks++;
        result.load_blocks++;
    }

    RTP_LLM_LOG_DEBUG("planned logical settlement from %s group_set[%zu] node_key=%ld",
                      tierName(source_tier),
                      group_set->groupSetId(),
                      path_node->cache_key);
    if (group_set_resource.transfer_state == GroupSetTransferState::LOADING) {
        RTP_LLM_LOG_DEBUG(
            "match joined LOADING, node_key=%ld group_set=%zu", path_node->cache_key, group_set->groupSetId());
    }
}

std::shared_ptr<LoadAsyncContext> BlockTreeLoader::prepareLoadContext(LoadAsyncContext::PendingLoadItems& items,
                                                                      const std::vector<bool>&            joined_load,
                                                                      size_t logical_matched_blocks) {
    if (!reserveLoadItems(items, joined_load)) {
        return nullptr;
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        if (!joined_load[item_index]) {
            continue;
        }
        if (!prepareJoinedLoadItem(items[item_index])) {
            abortLoadNolock(items, joined_load, 0, 0);
            return nullptr;
        }
    }

    size_t pending_transfer_count = 0;
    for (const LoadAsyncContext::PendingLoadItem& item : items) {
        if (item.source_tier == Tier::HOST || item.source_tier == Tier::DISK) {
            ++pending_transfer_count;
        }
    }

    const std::shared_ptr<LoadAsyncContext> context =
        load_context_coordinator_->create(items, joined_load, logical_matched_blocks, pending_transfer_count);
    if (context == nullptr) {
        abortLoadNolock(items, joined_load, 0, 0);
        return nullptr;
    }
    const uint64_t context_id = context->contextId();

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        if (!joined_load[item_index]) {
            continue;
        }
        const LoadAsyncContext::PendingLoadItem& item = items[item_index];
        std::vector<BlockIdxType>                joined_target_blocks;
        const bool joined = load_join_registry_.join(item.node, item.group_set_id, context, joined_target_blocks);
        if (!joined) {
            RTP_LLM_LOG_ERROR("failed to attach joined load context, group_set_id=%zu", item.group_set_id);
            abortLoadNolock(items, joined_load, 0, context_id);
            return nullptr;
        }
    }
    if (!load_context_coordinator_->registerContext(context)) {
        abortLoadNolock(items, joined_load, 0, context_id);
        return nullptr;
    }
    return context;
}

bool BlockTreeLoader::prepareJoinedLoadItem(LoadAsyncContext::PendingLoadItem& item) {
    std::vector<BlockIdxType> target_blocks;
    const bool                found = load_join_registry_.getTargetBlocks(item.node, item.group_set_id, target_blocks);
    if (!found) {
        RTP_LLM_LOG_ERROR("LOADING resource has no registry entry, group_set_id=%zu", item.group_set_id);
        return false;
    }
    group_sets_[item.group_set_id]->referenceBlocks(MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, target_blocks}}},
                                                    BlockRefType::REQUEST);
    item.target_device_blocks = std::move(target_blocks);
    return true;
}

bool BlockTreeLoader::reserveLoadItems(const LoadAsyncContext::PendingLoadItems& items,
                                       const std::vector<bool>&                  joined_load) {
    const bool has_lower_tier_item =
        std::any_of(items.begin(), items.end(), [](const LoadAsyncContext::PendingLoadItem& item) {
            return item.source_tier == Tier::HOST || item.source_tier == Tier::DISK;
        });
    if (!has_lower_tier_item) {
        return false;
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item = items[item_index];
        if (joined_load[item_index]) {
            continue;
        }
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, item.source_tier, {{item.node, item.source_blocks}}}, BlockRefType::REQUEST);
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item = items[item_index];
        if (item.source_tier == Tier::DEVICE || joined_load[item_index]) {
            continue;
        }
        if (!changeTransferState(
                item.node, item.group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING)) {
            RTP_LLM_LOG_ERROR("new load source is not IDLE, group_set_id=%zu", item.group_set_id);
            abortLoadNolock(items, joined_load, 0, 0);
            return false;
        }
        evictor_.refreshCandidate(item.node, item.group_set_id);
    }
    return true;
}

bool BlockTreeLoader::commitLoad(const std::shared_ptr<LoadAsyncContext>& context) {
    std::lock_guard<std::mutex>               lock(mutex_);
    const LoadAsyncContext::PendingLoadItems& items               = context->items();
    const std::vector<bool>&                  joined_load         = context->joinedLoadItems();
    const uint64_t                            context_id          = context->contextId();
    size_t                                    prepared_item_count = 0;
    block_tree_cache_detail::ScopeRollback    rollback_guard(
        [this, &items, &joined_load, &prepared_item_count, context_id]() {
            abortLoadNolock(items, joined_load, prepared_item_count, context_id);
        });

    LoadTaskRunner::TaskPtr task = load_task_runner_.createTask(items, joined_load, group_sets_, context);
    if (task != nullptr) {
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            const LoadAsyncContext::PendingLoadItem& item = task->items[item_index];
            if (item.source_tier != Tier::DEVICE
                && !task->item_group_sets[item_index]->hasAllocatedDeviceBlocks(item.target_device_blocks)) {
                RTP_LLM_LOG_WARNING("invalid load target blocks, group_set=%zu", item.group_set_id);
                return false;
            }
        }
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item = items[item_index];
        if (item.source_tier == Tier::DEVICE || joined_load[item_index]) {
            ++prepared_item_count;
            continue;
        }
        const bool started =
            load_join_registry_.start(item.node, item.group_set_id, item.target_device_blocks, context);
        if (!started) {
            RTP_LLM_LOG_ERROR("failed to register new load, group_set_id=%zu", item.group_set_id);
            return false;
        }
        if (!changeTransferState(
                item.node, item.group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING)) {
            const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context_id);
            if (!erased) {
                RTP_LLM_LOG_ERROR("failed to erase rejected load, group_set_id=%zu", item.group_set_id);
            }
            RTP_LLM_LOG_ERROR("committed load source is not LOAD_PENDING, group_set_id=%zu", item.group_set_id);
            return false;
        }
        // Add an in-flight copy holder. It becomes a cache holder only after
        // the target blocks are installed into the tree resource.
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}}, BlockRefType::REQUEST);
        ++prepared_item_count;
    }

    if (task != nullptr) {
        const bool submitted = task_pool_->submit([this, task]() { runLoadTask(task); });
        if (!submitted) {
            rollback_guard.run();
            return false;
        }
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item = items[item_index];
        if (joined_load[item_index]) {
            group_sets_[item.group_set_id]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}}, BlockRefType::REQUEST);
        }
    }
    rollback_guard.dismiss();
    return true;
}

void BlockTreeLoader::abortLoad(LoadAsyncContext& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadNolock(context.items(), context.joinedLoadItems(), 0, context.contextId());
}

void BlockTreeLoader::abortLoadNolock(const LoadAsyncContext::PendingLoadItems& items,
                                      const std::vector<bool>&                  joined_load,
                                      size_t                                    prepared_item_count,
                                      uint64_t                                  context_id) {
    bool device_refs_released = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item           = items[item_index];
        const size_t                             group_set_id   = item.group_set_id;
        const bool                               fully_prepared = item_index < prepared_item_count;
        if (joined_load[item_index]) {
            if (context_id != 0) {
                const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context_id);
                if (!erased) {
                    RTP_LLM_LOG_DEBUG("joined load context is no longer registered, group_set=%zu", item.group_set_id);
                }
            }
            if (!item.target_device_blocks.empty()) {
                group_sets_[group_set_id]->unreferenceBlocks(
                    MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}},
                    BlockRefType::REQUEST);
                device_refs_released = true;
            }
            continue;
        }
        if (item.source_tier != Tier::DEVICE && fully_prepared) {
            if (context_id != 0) {
                const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context_id);
                if (!erased) {
                    RTP_LLM_LOG_WARNING("failed to erase aborted load context, group_set=%zu", item.group_set_id);
                }
            }
            group_sets_[group_set_id]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}}, BlockRefType::REQUEST);
        }

        MultiNodeResource source_set{item.group_set_id, item.source_tier, {{item.node, item.source_blocks}}};
        group_sets_[group_set_id]->unreferenceBlocks(source_set, BlockRefType::REQUEST);
        if (item.source_tier != Tier::DEVICE) {
            const GroupSetTransferState expected_state =
                fully_prepared ? GroupSetTransferState::LOADING : GroupSetTransferState::LOAD_PENDING;
            if (!changeTransferState(item.node, item.group_set_id, expected_state, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_WARNING("load rollback state mismatch, group_set=%zu source=%s",
                                    item.group_set_id,
                                    tierName(item.source_tier));
            } else {
                if (fully_prepared) {
                    evictor_.refreshCandidate(item.node, group_set_id);
                }
            }
            if (!fully_prepared) {
                evictor_.refreshCandidatesAfterRelease(source_set);
            }
        } else {
            evictor_.refreshCandidatesAfterRelease(source_set);
            device_refs_released = true;
        }
    }
    if (device_refs_released) {
        settled_(false, true);
    }
}

void BlockTreeLoader::runLoadTask(const LoadTaskRunner::TaskPtr& task) {
    bool copy_success = false;
    try {
        bool prepared = !task->items.empty();
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            if (!load_task_runner_.prepareTransferItem(*task, item_index)) {
                prepared = false;
            }
        }

        copy_success = load_task_runner_.runTransfer(
            *task, *transfer_dispatcher_, metrics_reporter_, disk_timeout_ms_, host_timeout_ms_, prepared);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("load task runner failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("load task runner failed with unknown exception");
    }

    // Commit the copied batch only while every stateful item still belongs
    // to this load operation.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool                  settlement_success = settleLoad(*task, copy_success);
        if (!settlement_success) {
            RTP_LLM_LOG_DEBUG("load task settled unsuccessfully");
        }
    }
}

bool BlockTreeLoader::settleLoad(LoadTaskRunner::Task& task, bool copy_success) {
    bool settlement_success   = copy_success;
    bool state_settled        = false;
    bool tree_data_mutated    = false;
    bool device_refs_released = false;

    for (const LoadAsyncContext::PendingLoadItem& item : task.items) {
        if (settlement_success && item.source_tier != Tier::DEVICE
            && item.node->group_set_resources[item.group_set_id].transfer_state != GroupSetTransferState::LOADING) {
            RTP_LLM_LOG_WARNING("completion state mismatch, group_set=%zu", item.group_set_id);
            settlement_success = false;
        }
    }

    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadAsyncContext::PendingLoadItem& item         = task.items[item_index];
        const GroupSetPtr&                       group_set    = task.item_group_sets[item_index];
        const size_t                             group_set_id = item.group_set_id;

        MultiNodeResource source_protection{group_set_id, item.source_tier, {{item.node, item.source_blocks}}};
        group_set->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        if (item.source_tier == Tier::DEVICE) {
            evictor_.refreshCandidatesAfterRelease(source_protection);
            device_refs_released = true;
            continue;
        }
        GroupSetResource& resource = item.node->group_set_resources[group_set_id];
        if (settlement_success) {
            if (enable_device_cache_) {
                MultiNodeResource target_holder{group_set_id, Tier::DEVICE, {{item.node, item.target_device_blocks}}};
                resource.setBlocks(Tier::DEVICE, item.target_device_blocks);
                group_set->mapDeviceBlocksToTreeNode(target_holder);
                group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
                group_set->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
                group_set->unreferenceBlocks(MultiNodeResource{group_set_id, item.source_tier, {{item.node, item.source_blocks}}},
                                             BlockRefType::BLOCK_CACHE);
                resource.evictFromTier(item.source_tier);
                task.target_installed[item_index] = true;
                tree_data_mutated                 = true;
            }
            if (!changeTransferState(
                    item.node, group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_ERROR("load state changed after locked preflight, group_set_id=%zu", group_set_id);
                settlement_success = false;
            } else {
                state_settled = true;
                if (enable_device_cache_) {
                    evictor_.onTierEntered(item.node, group_set_id, Tier::DEVICE);
                } else {
                    evictor_.refreshCandidate(item.node, group_set_id);
                }
            }
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!changeTransferState(
                item.node, group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group_set=%zu source=%s", group_set_id, tierName(item.source_tier));
        } else {
            evictor_.refreshCandidate(item.node, group_set_id);
            state_settled = true;
        }
    }
    settled_(tree_data_mutated, device_refs_released || state_settled);
    load_task_runner_.releaseTaskResources(task);
    for (const LoadAsyncContext::PendingLoadItem& item : task.items) {
        if (item.source_tier == Tier::DEVICE) {
            continue;
        }
        const bool completed = load_join_registry_.finish(item.node, item.group_set_id, settlement_success);
        if (!completed) {
            RTP_LLM_LOG_WARNING("failed to finish loading record, group_set=%zu", item.group_set_id);
        }
    }
    return settlement_success;
}

bool BlockTreeLoader::changeTransferState(TreeNode*             node,
                                          size_t                group_set_id,
                                          GroupSetTransferState expected_state,
                                          GroupSetTransferState target_state) {
    GroupSetResource& resource = node->group_set_resources[group_set_id];
    if (resource.transfer_state != expected_state) {
        return false;
    }
    resource.transfer_state = target_state;
    return true;
}

}  // namespace rtp_llm
