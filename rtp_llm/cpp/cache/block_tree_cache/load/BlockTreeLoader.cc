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

    std::vector<TransferDescriptor> pending_load_descs;
    std::vector<bool> joined_load;
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
            prepareMatchedLoadDescriptor(
                path_node, group_set, group_set_resource, i, result, pending_load_descs, joined_load);
        }
    }

    if (!pending_load_descs.empty()) {
        result.async_context = prepareLoadContext(pending_load_descs, joined_load, matched_block_count);
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

void BlockTreeLoader::prepareMatchedLoadDescriptor(TreeNode*                           path_node,
                                             const GroupSetPtr&                  group_set,
                                             const GroupSetResource&             group_set_resource,
                                             size_t                              path_index,
                                             BlockTreeMatchResult&            result,
                                             std::vector<TransferDescriptor>& pending_load_descs,
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

    TransferDescriptor pending_desc;
    pending_desc.node          = path_node;
    pending_desc.group_set_id  = group_set->groupSetId();
    pending_desc.path_index    = path_index;
    pending_desc.source_tier   = source_tier;
    pending_desc.target_tier   = Tier::DEVICE;
    pending_desc.source_blocks = source_blocks;
    pending_load_descs.push_back(std::move(pending_desc));
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

std::shared_ptr<LoadAsyncContext> BlockTreeLoader::prepareLoadContext(std::vector<TransferDescriptor>& load_descs,
                                                                      const std::vector<bool>&            joined_load,
                                                                      size_t logical_matched_blocks) {
    if (!reserveLoadDescriptors(load_descs, joined_load)) {
        return nullptr;
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        if (!joined_load[desc_index]) {
            continue;
        }
        if (!prepareJoinedLoadDescriptor(load_descs[desc_index])) {
            abortLoadNolock(load_descs, joined_load, 0, 0);
            return nullptr;
        }
    }

    size_t pending_transfer_count = 0;
    for (const TransferDescriptor& desc : load_descs) {
        if (desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK) {
            ++pending_transfer_count;
        }
    }

    const std::shared_ptr<LoadAsyncContext> context =
        load_context_coordinator_->create(load_descs, joined_load, logical_matched_blocks, pending_transfer_count);
    if (context == nullptr) {
        abortLoadNolock(load_descs, joined_load, 0, 0);
        return nullptr;
    }
    const uint64_t context_id = context->contextId();

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        if (!joined_load[desc_index]) {
            continue;
        }
        const TransferDescriptor& desc = load_descs[desc_index];
        std::vector<BlockIdxType>                joined_target_blocks;
        const bool joined = load_join_registry_.join(desc.node, desc.group_set_id, context, joined_target_blocks);
        if (!joined) {
            RTP_LLM_LOG_ERROR("failed to attach joined load context, group_set_id=%zu", desc.group_set_id);
            abortLoadNolock(load_descs, joined_load, 0, context_id);
            return nullptr;
        }
    }
    if (!load_context_coordinator_->registerContext(context)) {
        abortLoadNolock(load_descs, joined_load, 0, context_id);
        return nullptr;
    }
    return context;
}

bool BlockTreeLoader::prepareJoinedLoadDescriptor(TransferDescriptor& desc) {
    std::vector<BlockIdxType> target_blocks;
    const bool                found = load_join_registry_.getTargetBlocks(desc.node, desc.group_set_id, target_blocks);
    if (!found) {
        RTP_LLM_LOG_ERROR("LOADING resource has no registry entry, group_set_id=%zu", desc.group_set_id);
        return false;
    }
    group_sets_[desc.group_set_id]->referenceBlocks(
        MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, target_blocks}}},
                                                    BlockRefType::REQUEST);
    desc.target_blocks = std::move(target_blocks);
    return true;
}

bool BlockTreeLoader::reserveLoadDescriptors(const std::vector<TransferDescriptor>& load_descs,
                                       const std::vector<bool>&                  joined_load) {
    const bool has_lower_tier_desc =
        std::any_of(load_descs.begin(), load_descs.end(), [](const TransferDescriptor& desc) {
            return desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK;
        });
    if (!has_lower_tier_desc) {
        return false;
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (joined_load[desc_index]) {
            continue;
        }
        group_sets_[desc.group_set_id]->referenceBlocks(
            MultiNodeResource{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}},
            BlockRefType::REQUEST);
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (desc.source_tier == Tier::DEVICE || joined_load[desc_index]) {
            continue;
        }
        if (!changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING)) {
            RTP_LLM_LOG_ERROR("new load source is not IDLE, group_set_id=%zu", desc.group_set_id);
            abortLoadNolock(load_descs, joined_load, 0, 0);
            return false;
        }
        evictor_.refreshCandidate(desc.node, desc.group_set_id);
    }
    return true;
}

bool BlockTreeLoader::commitLoad(const std::shared_ptr<LoadAsyncContext>& context) {
    std::lock_guard<std::mutex>               lock(mutex_);
    const std::vector<TransferDescriptor>& load_descs               = context->loadDescs();
    const std::vector<bool>&                  joined_load         = context->joinedLoads();
    const uint64_t                            context_id          = context->contextId();
    size_t                                    prepared_desc_count = 0;
    block_tree_cache_detail::ScopeRollback    rollback_guard(
        [this, &load_descs, &joined_load, &prepared_desc_count, context_id]() {
            abortLoadNolock(load_descs, joined_load, prepared_desc_count, context_id);
        });

    LoadTaskRunner::TaskPtr task = load_task_runner_.createTask(load_descs, joined_load, group_sets_, context);
    if (task != nullptr) {
        for (size_t desc_index = 0; desc_index < task->load_descs.size(); ++desc_index) {
            const TransferDescriptor& desc = task->load_descs[desc_index];
            if (desc.source_tier != Tier::DEVICE
                && !task->desc_group_sets[desc_index]->hasAllocatedDeviceBlocks(desc.target_blocks)) {
                RTP_LLM_LOG_WARNING("invalid load target blocks, group_set=%zu", desc.group_set_id);
                return false;
            }
        }
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (desc.source_tier == Tier::DEVICE || joined_load[desc_index]) {
            ++prepared_desc_count;
            continue;
        }
        const bool started =
            load_join_registry_.start(desc.node, desc.group_set_id, desc.target_blocks, context);
        if (!started) {
            RTP_LLM_LOG_ERROR("failed to register new load, group_set_id=%zu", desc.group_set_id);
            return false;
        }
        if (!changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING)) {
            const bool erased = load_join_registry_.eraseForContext(desc.node, desc.group_set_id, context_id);
            if (!erased) {
                RTP_LLM_LOG_ERROR("failed to erase rejected load, group_set_id=%zu", desc.group_set_id);
            }
            RTP_LLM_LOG_ERROR("committed load source is not LOAD_PENDING, group_set_id=%zu", desc.group_set_id);
            return false;
        }
        // Add an in-flight copy holder. It becomes a cache holder only after
        // the target blocks are installed into the tree resource.
        group_sets_[desc.group_set_id]->referenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockRefType::REQUEST);
        ++prepared_desc_count;
    }

    if (task != nullptr) {
        const bool submitted = task_pool_->submit([this, task]() { runLoadTask(task); });
        if (!submitted) {
            rollback_guard.run();
            return false;
        }
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (joined_load[desc_index]) {
            group_sets_[desc.group_set_id]->unreferenceBlocks(
                MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
                BlockRefType::REQUEST);
        }
    }
    rollback_guard.dismiss();
    return true;
}

void BlockTreeLoader::abortLoad(LoadAsyncContext& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadNolock(context.loadDescs(), context.joinedLoads(), 0, context.contextId());
}

void BlockTreeLoader::abortLoadNolock(const std::vector<TransferDescriptor>& load_descs,
                                      const std::vector<bool>&                  joined_load,
                                      size_t                                    prepared_desc_count,
                                      uint64_t                                  context_id) {
    bool device_refs_released = false;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        const size_t                             group_set_id   = desc.group_set_id;
        const bool                               fully_prepared = desc_index < prepared_desc_count;
        if (joined_load[desc_index]) {
            if (context_id != 0) {
                const bool erased = load_join_registry_.eraseForContext(desc.node, desc.group_set_id, context_id);
                if (!erased) {
                    RTP_LLM_LOG_DEBUG("joined load context is no longer registered, group_set=%zu", desc.group_set_id);
                }
            }
            if (!desc.target_blocks.empty()) {
                group_sets_[group_set_id]->unreferenceBlocks(
                    MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
                    BlockRefType::REQUEST);
                device_refs_released = true;
            }
            continue;
        }
        if (desc.source_tier != Tier::DEVICE && fully_prepared) {
            if (context_id != 0) {
                const bool erased = load_join_registry_.eraseForContext(desc.node, desc.group_set_id, context_id);
                if (!erased) {
                    RTP_LLM_LOG_WARNING("failed to erase aborted load context, group_set=%zu", desc.group_set_id);
                }
            }
            group_sets_[group_set_id]->unreferenceBlocks(
                MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
                BlockRefType::REQUEST);
        }

        MultiNodeResource source_set{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        group_sets_[group_set_id]->unreferenceBlocks(source_set, BlockRefType::REQUEST);
        if (desc.source_tier != Tier::DEVICE) {
            const GroupSetTransferState expected_state =
                fully_prepared ? GroupSetTransferState::LOADING : GroupSetTransferState::LOAD_PENDING;
            if (!changeTransferState(desc.node, desc.group_set_id, expected_state, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_WARNING("load rollback state mismatch, group_set=%zu source=%s",
                                    desc.group_set_id,
                                    tierName(desc.source_tier));
            } else {
                if (fully_prepared) {
                    evictor_.refreshCandidate(desc.node, group_set_id);
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
        bool prepared = !task->load_descs.empty();
        for (size_t desc_index = 0; desc_index < task->load_descs.size(); ++desc_index) {
            if (!load_task_runner_.prepareTransferDescriptor(*task, desc_index)) {
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

    // Commit the copied batch only while every stateful descriptor still belongs
    // to this load operation.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool                  settlement_success = settleLoadLocked(*task, copy_success);
        if (!settlement_success) {
            RTP_LLM_LOG_DEBUG("load task settled unsuccessfully");
        }
    }
}

bool BlockTreeLoader::settleLoadLocked(LoadTaskRunner::Task& task, bool copy_success) {
    bool settlement_success   = copy_success;
    bool state_settled        = false;
    bool tree_data_mutated    = false;
    bool device_refs_released = false;

    for (const TransferDescriptor& desc : task.load_descs) {
        if (settlement_success && desc.source_tier != Tier::DEVICE
            && desc.node->group_set_resources[desc.group_set_id].transfer_state != GroupSetTransferState::LOADING) {
            RTP_LLM_LOG_WARNING("completion state mismatch, group_set=%zu", desc.group_set_id);
            settlement_success = false;
        }
    }

    for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = task.load_descs[desc_index];
        const GroupSetPtr&                       group_set    = task.desc_group_sets[desc_index];
        const size_t                             group_set_id = desc.group_set_id;

        MultiNodeResource source_protection{
            group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        group_set->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        if (desc.source_tier == Tier::DEVICE) {
            evictor_.refreshCandidatesAfterRelease(source_protection);
            device_refs_released = true;
            continue;
        }
        GroupSetResource& resource = desc.node->group_set_resources[group_set_id];
        if (settlement_success) {
            if (enable_device_cache_) {
                MultiNodeResource target_holder{
                    group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}};
                resource.setBlocks(Tier::DEVICE, desc.target_blocks);
                group_set->mapDeviceBlocksToTreeNode(target_holder);
                group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
                group_set->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
                group_set->unreferenceBlocks(
                    MultiNodeResource{group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}},
                    BlockRefType::BLOCK_CACHE);
                resource.evictFromTier(desc.source_tier);
                task.target_installed[desc_index] = true;
                tree_data_mutated                 = true;
            }
            if (!changeTransferState(
                    desc.node, group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_ERROR("load state changed after locked preflight, group_set_id=%zu", group_set_id);
                settlement_success = false;
            } else {
                state_settled = true;
                if (enable_device_cache_) {
                    evictor_.onTierEntered(desc.node, group_set_id, Tier::DEVICE);
                } else {
                    evictor_.refreshCandidate(desc.node, group_set_id);
                }
            }
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!changeTransferState(
                desc.node, group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group_set=%zu source=%s", group_set_id, tierName(desc.source_tier));
        } else {
            evictor_.refreshCandidate(desc.node, group_set_id);
            state_settled = true;
        }
    }
    settled_(tree_data_mutated, device_refs_released || state_settled);
    load_task_runner_.releaseTaskResources(task);
    for (const TransferDescriptor& desc : task.load_descs) {
        if (desc.source_tier == Tier::DEVICE) {
            continue;
        }
        const bool completed = load_join_registry_.finish(desc.node, desc.group_set_id, settlement_success);
        if (!completed) {
            RTP_LLM_LOG_WARNING("failed to finish loading record, group_set=%zu", desc.group_set_id);
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
