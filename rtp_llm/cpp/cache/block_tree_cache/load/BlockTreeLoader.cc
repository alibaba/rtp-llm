#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeLoader::BlockTreeLoader(std::vector<GroupSetPtr>&      group_sets,
                                 BlockTreeEvictor&              evictor,
                                 BlockTransferDispatcher*       transfer_dispatcher,
                                 BlockTreeTaskPool*             task_pool,
                                 BlockTreeCacheMetricsReporter& metrics_reporter,
                                 std::mutex&                    mutex,
                                 int                            disk_timeout_ms,
                                 int                            host_timeout_ms,
                                 ReclaimOneFn                   reclaim_one,
                                 SettledFn                      settled):
    group_sets_(group_sets),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    disk_timeout_ms_(disk_timeout_ms),
    host_timeout_ms_(host_timeout_ms),
    reclaim_one_(std::move(reclaim_one)),
    settled_(std::move(settled)),
    load_ticket_registry_(
        std::make_shared<LoadTicketRegistry>([this](const LoadTicket& ticket) { return commitLoad(ticket); },
                                             [this](const LoadTicket& ticket) { abortLoad(ticket); })) {}

BlockTreeLoadResult BlockTreeLoader::prepareLoadLocked(const std::vector<TreeNode*>& matched_path,
                                                       size_t                        ready_matched_block_count) {
    BlockTreeLoadResult result;
    const size_t        logical_matched_block_count = matched_path.size();
    if (logical_matched_block_count == 0) {
        return result;
    }

    LoadTicket::PendingLoadItems pending_load_items;
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set         = group_sets_[group_set_id];
        const size_t       ready_reuse_count = std::min(
            group_set->computeReuseBlockCount(ready_matched_block_count, matched_path), ready_matched_block_count);
        const size_t ready_reuse_begin   = ready_matched_block_count - ready_reuse_count;
        const size_t logical_reuse_count = std::min(
            group_set->computeReuseBlockCount(logical_matched_block_count, matched_path), logical_matched_block_count);
        for (size_t i = logical_matched_block_count - logical_reuse_count; i < logical_matched_block_count; ++i) {
            if (i >= ready_reuse_begin && i < ready_matched_block_count) {
                continue;
            }
            TreeNode*         path_node          = matched_path[i];
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            prepareMatchedLoadItem(path_node, group_set, group_set_resource, i, result, pending_load_items);
        }
    }

    if (!pending_load_items.empty()) {
        result.load_ticket = prepareLoadTicket(pending_load_items, logical_matched_block_count);
        if (result.load_ticket == nullptr) {
            result.load_blocks      = 0;
            result.host_load_blocks = 0;
            result.disk_load_blocks = 0;
        }
    }
    return result;
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
    load_ticket_registry_->shutdown();
}

void BlockTreeLoader::prepareMatchedLoadItem(TreeNode*                     path_node,
                                             const GroupSetPtr&            group_set,
                                             const GroupSetResource&       group_set_resource,
                                             size_t                        path_index,
                                             BlockTreeLoadResult&          result,
                                             LoadTicket::PendingLoadItems& pending_load_items) {
    // DEMOTING/LOAD_PENDING sources belong to another in-flight operation and
    // can neither be referenced nor joined; skip them like empty slots.
    if (!group_set_resource.isMatchUsable()) {
        RTP_LLM_LOG_DEBUG("skip busy slot for load planning, node_key=%ld group_set=%zu state=%d",
                          path_node->cache_key,
                          group_set->groupSetId(),
                          static_cast<int>(group_set_resource.transfer_state));
        return;
    }
    const Tier source_tier = group_set->getTopTier(group_set_resource);
    if (source_tier == Tier::NONE) {
        return;
    }

    const std::vector<BlockIdxType> source_blocks = group_set->getBlocks(group_set_resource, source_tier);

    LoadTicket::PendingLoadItem pending_item;
    pending_item.node          = path_node;
    pending_item.group_set_id  = group_set->groupSetId();
    pending_item.path_index    = path_index;
    pending_item.source_tier   = source_tier;
    pending_item.source_blocks = source_blocks;
    pending_item.joined_load   = group_set_resource.transfer_state == GroupSetTransferState::LOADING;
    pending_load_items.push_back(std::move(pending_item));

    if (source_tier == Tier::HOST) {
        result.host_load_blocks++;
        result.load_blocks++;
    } else if (source_tier == Tier::DISK) {
        result.disk_load_blocks++;
        result.load_blocks++;
    }

    RTP_LLM_LOG_DEBUG("planned logical settlement from %s group[%zu] node_key=%ld",
                      tierName(source_tier),
                      group_set->groupSetId(),
                      path_node->cache_key);
    if (group_set_resource.transfer_state == GroupSetTransferState::LOADING) {
        RTP_LLM_LOG_DEBUG(
            "match joined LOADING, node_key=%ld group_set=%zu", path_node->cache_key, group_set->groupSetId());
    }
}

std::shared_ptr<LoadTicket> BlockTreeLoader::prepareLoadTicket(LoadTicket::PendingLoadItems& items,
                                                               size_t                        logical_matched_blocks) {
    if (!reserveLoadItems(items)) {
        return nullptr;
    }

    size_t pending_transfer_count = 0;
    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.source_tier == Tier::HOST || item.source_tier == Tier::DISK) {
            ++pending_transfer_count;
        }
    }
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(pending_transfer_count);
    for (LoadTicket::PendingLoadItem& item : items) {
        if (!item.joined_load) {
            continue;
        }
        if (!prepareJoinedLoadItem(item, context)) {
            abortLoadUnsafe(items, 0, context);
            return nullptr;
        }
    }

    std::shared_ptr<LoadTicket> ticket = load_ticket_registry_->createTicket(items, logical_matched_blocks, context);
    if (ticket == nullptr) {
        abortLoadUnsafe(items, 0, context);
    }
    return ticket;
}

bool BlockTreeLoader::prepareJoinedLoadItem(LoadTicket::PendingLoadItem&             item,
                                            const std::shared_ptr<LoadAsyncContext>& context) {
    const std::optional<std::vector<BlockIdxType>> target_blocks =
        load_join_registry_.join(item.node, item.group_set_id, context);
    if (!target_blocks.has_value()) {
        RTP_LLM_LOG_WARNING("failed to join active load, group_set=%zu", item.group_set_id);
        return false;
    }
    item.target_device_blocks = target_blocks.value();
    if (item.target_device_blocks.size() != group_sets_[item.group_set_id]->devicePoolCount()) {
        item.target_device_blocks.clear();
        return false;
    }
    group_sets_[item.group_set_id]->referenceBlocks(
        MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
    return true;
}

bool BlockTreeLoader::reserveLoadItems(const LoadTicket::PendingLoadItems& items) {
    const bool has_lower_tier_item =
        std::any_of(items.begin(), items.end(), [](const LoadTicket::PendingLoadItem& item) {
            return item.source_tier == Tier::HOST || item.source_tier == Tier::DISK;
        });
    if (items.empty() || !has_lower_tier_item) {
        return false;
    }

    for (const auto& item : items) {
        if (item.node == nullptr || item.group_set_id >= group_sets_.size()
            || (item.source_tier != Tier::DEVICE && item.source_tier != Tier::HOST && item.source_tier != Tier::DISK)) {
            return false;
        }
        const GroupSetPtr& group = group_sets_[item.group_set_id];
        if (group == nullptr || item.group_set_id >= item.node->group_set_resources.size()) {
            return false;
        }
        const GroupSetTransferState expected_state =
            item.joined_load ? GroupSetTransferState::LOADING : GroupSetTransferState::IDLE;
        if (item.node->group_set_resources[item.group_set_id].transfer_state != expected_state) {
            return false;
        }
        const size_t expected_source_count = item.source_tier == Tier::DEVICE ? group->devicePoolCount() : 1;
        if (item.source_blocks.size() != expected_source_count
            || group->getTopTier(item.node->group_set_resources[item.group_set_id]) != item.source_tier
            || group->getBlocks(item.node->group_set_resources[item.group_set_id], item.source_tier)
                   != item.source_blocks) {
            return false;
        }
    }

    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.joined_load) {
            continue;
        }
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, item.source_tier, {item.source_blocks}}, BlockRefType::REQUEST);
    }

    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.source_tier == Tier::DEVICE || item.joined_load) {
            continue;
        }
        if (!reserveLoad(item.node, item.group_set_id, item.source_tier, item.source_blocks)) {
            abortLoadUnsafe(items, /*prepared_item_count=*/0, nullptr);
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncContext> BlockTreeLoader::commitLoad(const LoadTicket& ticket) {
    std::lock_guard<std::mutex>         lock(mutex_);
    const LoadTicket::PendingLoadItems& items = ticket.items();

    size_t                                  prepared_item_count = 0;
    const std::shared_ptr<LoadAsyncContext> context             = ticket.context();
    block_tree_cache_detail::ScopeRollback  rollback_guard(
        [this, &items, &prepared_item_count, &context]() { abortLoadUnsafe(items, prepared_item_count, context); });
    if (context == nullptr) {
        RTP_LLM_LOG_WARNING("load ticket has no context");
        return nullptr;
    }

    LoadTaskRunner::TaskPtr task;
    if (!load_task_runner_.createTask(items, group_sets_, context, task)) {
        return nullptr;
    }
    if (task != nullptr) {
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            const LoadTicket::PendingLoadItem& item = task->items[item_index];
            if (item.source_tier != Tier::DEVICE
                && !task->item_groups[item_index]->hasAllocatedDeviceBlocks(item.target_device_blocks)) {
                RTP_LLM_LOG_WARNING("invalid load target blocks, group_set=%zu", item.group_set_id);
                return nullptr;
            }
        }
    }

    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.source_tier == Tier::DEVICE || item.joined_load) {
            ++prepared_item_count;
            continue;
        }
        if (!load_join_registry_.start(item.node, item.group_set_id, item.target_device_blocks, context)) {
            RTP_LLM_LOG_WARNING("failed to create loading record, group_set=%zu", item.group_set_id);
            return nullptr;
        }
        if (!beginLoad(item.node, item.group_set_id, item.source_tier)) {
            const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context);
            if (!erased) {
                RTP_LLM_LOG_ERROR("failed to erase load context, group_set=%zu", item.group_set_id);
            }
            RTP_LLM_LOG_WARNING("pending-to-loading transition failed, rolled back all %zu load items", items.size());
            return nullptr;
        }
        // Add an in-flight copy holder. It becomes a cache holder only after
        // the target blocks are installed into the tree slot.
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        ++prepared_item_count;
    }

    if (task != nullptr) {
        const bool submitted = task_pool_->submit([this, task]() { runLoadTask(task); });
        if (!submitted) {
            rollback_guard.run();
            const bool completed = context->onTaskFail();
            if (!completed) {
                RTP_LLM_LOG_ERROR("failed to complete rejected load task");
            }
            return context;
        }
    }

    for (const LoadTicket::PendingLoadItem& item : items) {
        if (item.joined_load) {
            group_sets_[item.group_set_id]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        }
    }
    rollback_guard.dismiss();
    return context;
}

void BlockTreeLoader::abortLoad(const LoadTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadUnsafe(ticket.items(), 0, ticket.context());
}

void BlockTreeLoader::abortLoadUnsafe(const LoadTicket::PendingLoadItems&      items,
                                      size_t                                   prepared_item_count,
                                      const std::shared_ptr<LoadAsyncContext>& context) {
    if (prepared_item_count > 0 && context == nullptr) {
        RTP_LLM_LOG_ERROR("missing context while aborting %zu prepared load items", prepared_item_count);
    }

    bool device_refs_released = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const auto&  item            = items[item_index];
        const size_t group_set_index = item.group_set_id;
        if (group_set_index >= group_sets_.size() || group_sets_[group_set_index] == nullptr) {
            continue;
        }
        const bool fully_prepared = item_index < prepared_item_count;
        if (item.joined_load) {
            if (context != nullptr) {
                const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context);
                if (!erased) {
                    RTP_LLM_LOG_DEBUG("joined load context is no longer registered, group_set=%zu", item.group_set_id);
                }
            }
            if (!item.target_device_blocks.empty()) {
                group_sets_[group_set_index]->unreferenceBlocks(
                    MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}},
                    BlockRefType::REQUEST);
                device_refs_released = true;
            }
            continue;
        }
        if (item.source_tier != Tier::DEVICE && fully_prepared) {
            if (context != nullptr) {
                const bool erased = load_join_registry_.eraseForContext(item.node, item.group_set_id, context);
                if (!erased) {
                    RTP_LLM_LOG_WARNING("failed to erase aborted load context, group_set=%zu", item.group_set_id);
                }
            }
            group_sets_[group_set_index]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        }

        MultiNodeResource source_set{item.group_set_id, item.source_tier, {item.source_blocks}};
        if (item.node != nullptr) {
            source_set.tree_nodes = {item.node};
        }
        group_sets_[group_set_index]->unreferenceBlocks(source_set, BlockRefType::REQUEST);
        if (item.source_tier != Tier::DEVICE) {
            if (fully_prepared) {
                if (!finishLoad(item.node, item.group_set_id, item.source_tier, false)) {
                    RTP_LLM_LOG_WARNING(
                        "loading state mismatch, group=%zu source=%s", item.group_set_id, tierName(item.source_tier));
                }
            } else {
                if (!abortPendingLoad(item.node, item.group_set_id, item.source_tier, item.source_blocks)) {
                    RTP_LLM_LOG_WARNING("reservation state mismatch, "
                                        "group=%zu source=%s",
                                        item.group_set_id,
                                        tierName(item.source_tier));
                }
                evictor_.refreshCandidatesAfterRelease(source_set);
            }
        } else {
            evictor_.refreshCandidatesAfterRelease(source_set);
            device_refs_released = true;
        }
    }
    if (device_refs_released && settled_) {
        settled_(false, true);
    }
}

void BlockTreeLoader::runLoadTask(const LoadTaskRunner::TaskPtr& task) {
    if (task == nullptr || task->context == nullptr) {
        RTP_LLM_LOG_ERROR("invalid load task");
        return;
    }

    bool copy_success = false;
    try {
        bool prepared = !task->items.empty();
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            LoadTaskRunner::PrepareStatus status = load_task_runner_.prepareTransferItem(*task, item_index);
            if (status == LoadTaskRunner::PrepareStatus::NEED_HOST_RECLAIM) {
                const size_t group_set_id = task->items[item_index].group_set_id;
                if (reclaim_one_(group_set_id, Tier::HOST)) {
                    status = load_task_runner_.prepareTransferItem(*task, item_index);
                }
            }
            if (status != LoadTaskRunner::PrepareStatus::READY) {
                if (status == LoadTaskRunner::PrepareStatus::NEED_HOST_RECLAIM) {
                    RTP_LLM_LOG_WARNING("failed to prepare host staging block, group_set=%zu",
                                        task->items[item_index].group_set_id);
                }
                prepared = false;
            }
        }

        copy_success = load_task_runner_.runTransfer(
            *task, *transfer_dispatcher_, metrics_reporter_, disk_timeout_ms_, host_timeout_ms_, prepared);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("load worker failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("load worker failed with unknown exception");
    }

    // Commit the copied batch only while every stateful item still belongs
    // to this load operation.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool                  settlement_success = settleLoadNolock(*task, copy_success);
        if (!settlement_success) {
            RTP_LLM_LOG_DEBUG("load task settled unsuccessfully");
        }
    }
}

bool BlockTreeLoader::settleLoadNolock(LoadTaskRunner::Task& task, bool copy_success) {
    bool settlement_success   = copy_success && task.context != nullptr;
    bool state_settled        = false;
    bool tree_data_mutated    = false;
    bool device_refs_released = false;

    RTP_LLM_CHECK_WITH_INFO(task.items.size() == task.item_groups.size()
                                && task.items.size() == task.target_installed.size(),
                            "malformed load task: items=%zu groups=%zu targets=%zu",
                            task.items.size(),
                            task.item_groups.size(),
                            task.target_installed.size());
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadTicket::PendingLoadItem& item  = task.items[item_index];
        const GroupSetPtr&                 group = task.item_groups[item_index];
        RTP_LLM_CHECK_WITH_INFO(group != nullptr && group->groupSetId() == item.group_set_id && item.node != nullptr
                                    && item.group_set_id < item.node->group_set_resources.size(),
                                "malformed load item: index=%zu group_set_id=%zu node=%p",
                                item_index,
                                item.group_set_id,
                                static_cast<void*>(item.node));
        if (settlement_success && item.source_tier != Tier::DEVICE
            && (item.target_device_blocks.size() != group->devicePoolCount()
                || item.node->group_set_resources[item.group_set_id].transfer_state
                       != GroupSetTransferState::LOADING)) {
            RTP_LLM_LOG_WARNING("completion state mismatch, group_set=%zu", item.group_set_id);
            settlement_success = false;
        }
    }

    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadTicket::PendingLoadItem& item         = task.items[item_index];
        const GroupSetPtr&                 group        = task.item_groups[item_index];
        const size_t                       group_set_id = item.group_set_id;

        MultiNodeResource source_protection{group_set_id, item.source_tier, {item.source_blocks}};
        source_protection.tree_nodes = {item.node};
        group->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        if (item.source_tier == Tier::DEVICE) {
            evictor_.refreshCandidatesAfterRelease(source_protection);
            device_refs_released = true;
            continue;
        }
        GroupSetResource& resource = item.node->group_set_resources[group_set_id];
        if (settlement_success) {
            MultiNodeResource target_holder{group_set_id, Tier::DEVICE, {item.target_device_blocks}};
            group->setBlocks(resource, Tier::DEVICE, item.target_device_blocks);
            group->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
            group->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
            group->unreferenceBlocks(MultiNodeResource{group_set_id, item.source_tier, {item.source_blocks}},
                                     BlockRefType::BLOCK_CACHE);
            group->evictFromTier(item.node, resource, item.source_tier);
            task.target_installed[item_index] = true;
            tree_data_mutated                 = true;
            RTP_LLM_CHECK_WITH_INFO(finishLoad(item.node, group_set_id, item.source_tier, true),
                                    "load state changed after locked preflight, group_set_id=%zu",
                                    group_set_id);
            state_settled = true;
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!finishLoad(item.node, group_set_id, item.source_tier, false)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group_set=%zu source=%s", group_set_id, tierName(item.source_tier));
        } else {
            state_settled = true;
        }
    }
    if (settled_) {
        settled_(tree_data_mutated, device_refs_released || state_settled);
    }
    load_task_runner_.releaseTaskResources(task);
    for (const LoadTicket::PendingLoadItem& item : task.items) {
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

bool BlockTreeLoader::reserveLoad(TreeNode*                        node,
                                  size_t                           group_set_id,
                                  Tier                             source,
                                  const std::vector<BlockIdxType>& source_blocks) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return false;
    }
    const GroupSetPtr& group = group_sets_[group_set_id];
    GroupSetResource&  slot  = node->group_set_resources[group_set_id];
    if (group == nullptr || (source != Tier::HOST && source != Tier::DISK)
        || slot.transfer_state != GroupSetTransferState::IDLE || group->getTopTier(slot) != source
        || group->getBlocks(slot, source) != source_blocks) {
        return false;
    }
    slot.transfer_state = GroupSetTransferState::LOAD_PENDING;
    evictor_.refreshCandidate(node, group_set_id);
    return true;
}

bool BlockTreeLoader::abortPendingLoad(TreeNode*                        node,
                                       size_t                           group_set_id,
                                       Tier                             source,
                                       const std::vector<BlockIdxType>& source_blocks) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return false;
    }
    const GroupSetPtr& group = group_sets_[group_set_id];
    GroupSetResource&  slot  = node->group_set_resources[group_set_id];
    if (group == nullptr || slot.transfer_state != GroupSetTransferState::LOAD_PENDING
        || group->getTopTier(slot) != source || group->getBlocks(slot, source) != source_blocks) {
        return false;
    }
    slot.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(group->isValidSteadyState(slot),
                            "load abort produced invalid steady state: group_set_id=%zu node_key=%ld",
                            group_set_id,
                            node->cache_key);
    evictor_.refreshCandidate(node, group_set_id);
    return true;
}

bool BlockTreeLoader::beginLoad(TreeNode* node, size_t group_set_id, Tier source) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return false;
    }
    const GroupSetPtr& group = group_sets_[group_set_id];
    GroupSetResource&  slot  = node->group_set_resources[group_set_id];
    if (group == nullptr || (source != Tier::HOST && source != Tier::DISK) || group->getTopTier(slot) != source
        || slot.transfer_state != GroupSetTransferState::LOAD_PENDING) {
        return false;
    }
    slot.transfer_state = GroupSetTransferState::LOADING;
    evictor_.refreshCandidate(node, group_set_id);
    return true;
}

bool BlockTreeLoader::finishLoad(TreeNode* node, size_t group_set_id, Tier source, bool copy_ok) {
    if (node == nullptr || group_set_id >= group_sets_.size() || group_set_id >= node->group_set_resources.size()) {
        return false;
    }
    const GroupSetPtr& group = group_sets_[group_set_id];
    GroupSetResource&  slot  = node->group_set_resources[group_set_id];
    if (group == nullptr || slot.transfer_state != GroupSetTransferState::LOADING) {
        RTP_LLM_LOG_WARNING("state mismatch, group=%zu node_key=%ld state=%d",
                            group_set_id,
                            node->cache_key,
                            static_cast<int>(slot.transfer_state));
        return false;
    }
    slot.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(group->isValidSteadyState(slot),
                            "load settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            group_set_id,
                            node->cache_key);
    if (copy_ok) {
        evictor_.onTierEntered(node, group_set_id, Tier::DEVICE);
    } else {
        evictor_.refreshCandidate(node, group_set_id);
    }
    return true;
}

}  // namespace rtp_llm
