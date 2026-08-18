#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

BlockTreeLoader::BlockTreeLoader(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             disk_timeout_ms,
                                 int                             host_timeout_ms,
                                 bool                            enable_device_cache,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    disk_timeout_ms_(disk_timeout_ms),
    host_timeout_ms_(host_timeout_ms),
    enable_device_cache_(enable_device_cache),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)),
    load_task_runner_(tree_->groupSets()),
    load_join_registry_(tree),
    load_context_coordinator_(std::make_shared<LoadContextCoordinator>(
        [this](const std::shared_ptr<LoadAsyncContext>& context) { return commitLoad(context); },
        [this](LoadAsyncContext& context) {
            std::lock_guard<std::mutex> lock(mutex_);
            abortLoadLocked(
                context.loadDescs(), context.joinedLoads(), 0, context.contextId(), /*release_transferred_refs=*/false);
        })) {}

BlockTreeMatchResult BlockTreeLoader::matchLocked(const CacheKeysType& cache_keys) {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("empty cache_keys, returning empty result");
        return {};
    }

    std::vector<TreeNode*> path   = tree_->findNode(cache_keys);
    BlockTreeMatchResult   result = createMatchResult(path, cache_keys);
    RTP_LLM_LOG_DEBUG("matched %zu device blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_device_blocks,
                      cache_keys.size(),
                      tree_->size());
    return result;
}

bool BlockTreeLoader::validMatch(std::vector<TreeNode*>& path, std::vector<bool>& candidate_valid) const {
    size_t valid_block_count = 0;
    candidate_valid.reserve(path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(tree_->groupSets().size());
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    for (size_t i = 0; i < path.size(); ++i) {
        TreeNode* node             = path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            if (!match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_block_count = i + 1;
        }
        candidate_valid.push_back(all_groups_valid);
    }

    path.resize(valid_block_count);
    candidate_valid.resize(valid_block_count);
    return valid_block_count > 0;
}

BlockIndicesType BlockTreeLoader::matchedBlocksForGroup(size_t                                group_id,
                                                        const std::vector<MultiNodeResource>& matched_resources) const {
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_id);
    if (location == nullptr) {
        return {};
    }
    for (const MultiNodeResource& resource : matched_resources) {
        if (resource.group_set_id != location->group_set_id) {
            continue;
        }
        BlockIndicesType blocks;
        blocks.reserve(resource.node_blocks.size());
        for (const auto& [_, node_blocks] : resource.node_blocks) {
            blocks.push_back(node_blocks[location->member_group_id]);
        }
        return blocks;
    }
    return {};
}

std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> BlockTreeLoader::collectReuseTimeSnapshots(
    const std::vector<TreeNode*>& path, size_t matched_device_blocks, int64_t access_time_us) const {
    std::vector<BlockTreeCacheReuseTimeSample> reuse_time_samples;
    reuse_time_samples.reserve(path.size() * tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
        const size_t       ready_reuse_count =
            std::min(group_set->computeReuseBlockCount(matched_device_blocks), matched_device_blocks);
        for (size_t i = matched_device_blocks - ready_reuse_count; i < matched_device_blocks; ++i) {
            const GroupSetResource& resource       = path[i]->group_set_resources[group_set_id];
            const CandidateMeta&    candidate_meta = resource.candidate_meta;
            reuse_time_samples.push_back({Tier::DEVICE,
                                          group_set->groupType(),
                                          candidate_meta.insert_time_us,
                                          candidate_meta.last_access_time_us,
                                          access_time_us});
        }

        const size_t logical_reuse_count = std::min(group_set->computeReuseBlockCount(path.size()), path.size());
        for (size_t i = std::max(path.size() - logical_reuse_count, matched_device_blocks); i < path.size(); ++i) {
            const GroupSetResource& resource       = path[i]->group_set_resources[group_set_id];
            const CandidateMeta&    candidate_meta = resource.candidate_meta;
            reuse_time_samples.push_back({resource.getTopTier(),
                                          group_set->groupType(),
                                          candidate_meta.insert_time_us,
                                          candidate_meta.last_access_time_us,
                                          access_time_us});
        }
    }
    return metrics_reporter_.collectCacheReuseTimeMetrics(reuse_time_samples);
}

BlockTreeMatchResult BlockTreeLoader::createMatchResult(std::vector<TreeNode*>& path, const CacheKeysType& cache_keys) {
    BlockTreeMatchResult result;
    std::vector<bool>    candidate_valid;
    if (!path.empty() && !validMatch(path, candidate_valid) && !storage_backend_) {
        return result;
    }
    const int64_t access_time_us = currentTimeUs();

    for (size_t candidate_count = path.size(); candidate_count > 0; --candidate_count) {
        if (!candidate_valid[candidate_count - 1]) {
            continue;
        }
        bool all_groups_ready = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
            const size_t reuse_count = std::min(group_set->computeReuseBlockCount(candidate_count), candidate_count);
            for (size_t i = candidate_count - reuse_count; i < candidate_count; ++i) {
                if (!path[i]->group_set_resources[group_set_id].hasCompleteDeviceValue()) {
                    all_groups_ready = false;
                    break;
                }
            }
            if (!all_groups_ready) {
                break;
            }
        }
        if (all_groups_ready) {
            result.matched_device_blocks = candidate_count;
            break;
        }
    }

    if (metrics_reporter_.enabled()) {
        result.reuse_time_metrics_snapshots =
            collectReuseTimeSnapshots(path, result.matched_device_blocks, access_time_us);
    }
    if (result.matched_device_blocks > 0) {
        evictor_.onMatched(
            std::vector<TreeNode*>(path.begin(), path.begin() + static_cast<ptrdiff_t>(result.matched_device_blocks)));
    }

    std::vector<TransferDescriptor> pending_load_descs;
    std::vector<bool>               joined_loads;
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
        const size_t       ready_reuse_count =
            std::min(group_set->computeReuseBlockCount(result.matched_device_blocks), result.matched_device_blocks);
        MultiNodeResource matched_device_resource{group_set_id, Tier::DEVICE};
        for (size_t i = result.matched_device_blocks - ready_reuse_count; i < result.matched_device_blocks; ++i) {
            const GroupSetResource& resource = path[i]->group_set_resources[group_set_id];
            matched_device_resource.node_blocks.emplace_back(path[i], resource.getBlocks(Tier::DEVICE));
        }
        if (!matched_device_resource.node_blocks.empty()) {
            group_set->referenceBlocks(matched_device_resource, BlockRefType::REQUEST);
            result.matched_device_resources.push_back(std::move(matched_device_resource));
        }

        const size_t logical_reuse_count = std::min(group_set->computeReuseBlockCount(path.size()), path.size());
        for (size_t i = std::max(path.size() - logical_reuse_count, result.matched_device_blocks); i < path.size();
             ++i) {
            GroupSetResource&  resource    = path[i]->group_set_resources[group_set_id];
            const Tier         source_tier = resource.getTopTier();
            TransferDescriptor desc{
                path[i], group_set_id, i, source_tier, Tier::DEVICE, resource.getBlocks(source_tier)};
            const bool         is_joined = resource.transfer_state == GroupSetTransferState::LOADING;
            if (!is_joined) {
                group_set->referenceBlocks(
                    MultiNodeResource{group_set_id, source_tier, {{path[i], desc.source_blocks}}},
                    BlockRefType::REQUEST);
                if (source_tier != Tier::DEVICE) {
                    resource.transfer_state = GroupSetTransferState::LOAD_PENDING;
                    evictor_.refreshCandidate(path[i], group_set_id);
                }
            }
            pending_load_descs.emplace_back(std::move(desc));
            joined_loads.push_back(is_joined);
        }
    }

    StorageRequest storage_request;
    if (storage_backend_ && path.size() < cache_keys.size()) {
        storage_request = makeStorageRequest(cache_keys, path.size());
    }
    const bool use_storage = !storage_request.empty();
    if (!pending_load_descs.empty() || use_storage) {
        result.async_context = load_context_coordinator_->create(pending_load_descs,
                                                                 joined_loads,
                                                                 path.size(),
                                                                 use_storage ? storage_backend_ : nullptr,
                                                                 std::move(storage_request));
        if (result.async_context == nullptr) {
            abortLoadLocked(pending_load_descs, joined_loads, 0, 0, /*release_transferred_refs=*/true);
        } else if (!load_join_registry_.join(result.async_context)
                   || !load_context_coordinator_->registerContext(result.async_context)) {
            abortLoadLocked(result.async_context->loadDescs(),
                            result.async_context->joinedLoads(),
                            0,
                            result.async_context->contextId(),
                            /*release_transferred_refs=*/true);
            result.async_context = nullptr;
        }
    }
    return result;
}

StorageRequest BlockTreeLoader::makeStorageRequest(const CacheKeysType& cache_keys,
                                                   size_t               local_matched_blocks_num) const {
    StorageRequest request{std::make_shared<CacheKeysType>(cache_keys),
                           std::vector<std::vector<StorageBlockHandle>>(cache_keys.size()),
                           local_matched_blocks_num};
    for (auto& key_handles : request.handles) {
        for (const auto& group_set : tree_->groupSets()) {
            for (size_t group_id : group_set->groupIds()) {
                key_handles.push_back({group_id, NULL_BLOCK_IDX});
            }
        }
    }
    return request;
}

bool BlockTreeLoader::cancelLoad(const std::shared_ptr<AsyncContext>& context) {
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

bool BlockTreeLoader::commitLoad(const std::shared_ptr<LoadAsyncContext>& context) {
    std::lock_guard<std::mutex>            lock(mutex_);
    const std::vector<TransferDescriptor>& load_descs          = context->loadDescs();
    const std::vector<bool>&               joined_loads        = context->joinedLoads();
    const uint64_t                         context_id          = context->contextId();
    size_t                                 prepared_desc_count = 0;
    block_tree_cache_detail::ScopeRollback rollback_guard(
        [this, &load_descs, &joined_loads, &prepared_desc_count, context_id]() {
            abortLoadLocked(load_descs,
                            joined_loads,
                            prepared_desc_count,
                            context_id,
                            /*release_transferred_refs=*/false);
        });

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (desc.source_tier == Tier::DEVICE || joined_loads[desc_index]) {
            ++prepared_desc_count;
            continue;
        }
        if (desc.node->group_set_resources[desc.group_set_id].transfer_detached
            || !changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING)) {
            RTP_LLM_LOG_ERROR("committed load source is not LOAD_PENDING, group_set_id=%zu", desc.group_set_id);
            return false;
        }
        RTP_LLM_CHECK(load_join_registry_.start(desc.node, desc.group_set_id, desc.target_blocks, context));
        tree_->groupSets()[desc.group_set_id]->referenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockRefType::REQUEST);
        ++prepared_desc_count;
    }

    LoadTaskRunner::TaskPtr task = load_task_runner_.createTask(context);
    if (task && !task_pool_->submit([this, task]() { runLoadTask(task); })) {
        return false;
    }

    rollback_guard.dismiss();
    return true;
}

void BlockTreeLoader::abortLoadLocked(const std::vector<TransferDescriptor>& load_descs,
                                      const std::vector<bool>&               joined_loads,
                                      size_t                                 prepared_desc_count,
                                      uint64_t                               context_id,
                                      bool                                   release_transferred_refs) {
    bool device_refs_released = false;
    bool tree_data_mutated    = false;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc           = load_descs[desc_index];
        const bool                joined_load    = joined_loads[desc_index];
        const bool                fully_prepared = desc_index < prepared_desc_count;
        if (joined_load || (desc.source_tier != Tier::DEVICE && fully_prepared)) {
            const bool erased = load_join_registry_.eraseForContext(desc.node, desc.group_set_id, context_id);
            if (!erased) {
                if (joined_load) {
                    RTP_LLM_LOG_DEBUG("joined load context is no longer registered, group_set=%zu", desc.group_set_id);
                } else {
                    RTP_LLM_LOG_WARNING("failed to erase aborted load context, group_set=%zu", desc.group_set_id);
                }
            }
            if (!joined_load || release_transferred_refs) {
                tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(
                    MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
                    BlockRefType::REQUEST);
            }
        }
        if (joined_load) {
            device_refs_released = device_refs_released || release_transferred_refs;
            continue;
        }

        MultiNodeResource resource{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        if (desc.source_tier != Tier::DEVICE || release_transferred_refs) {
            tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(resource, BlockRefType::REQUEST);
        }
        if (desc.node->group_set_resources[desc.group_set_id].transfer_detached) {
            evictor_.discardDetachedTransfer(desc);
            tree_data_mutated = true;
            continue;
        }
        if (desc.source_tier != Tier::DEVICE) {
            const GroupSetTransferState expected_state =
                fully_prepared ? GroupSetTransferState::LOADING : GroupSetTransferState::LOAD_PENDING;
            if (!changeTransferState(desc.node, desc.group_set_id, expected_state, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_WARNING("load rollback state mismatch, group_set=%zu source=%s",
                                    desc.group_set_id,
                                    tierName(desc.source_tier));
            } else if (fully_prepared) {
                evictor_.refreshCandidate(desc.node, desc.group_set_id);
            }
            if (!fully_prepared) {
                evictor_.refreshCandidatesAfterRelease(resource);
            }
        } else if (release_transferred_refs) {
            evictor_.refreshCandidatesAfterRelease(resource);
            device_refs_released = true;
        }
    }
    if (tree_data_mutated || device_refs_released) {
        settled_(tree_data_mutated, device_refs_released);
    }
}

void BlockTreeLoader::runLoadTask(const LoadTaskRunner::TaskPtr& task) {
    bool copy_success = false;
    try {
        copy_success = load_task_runner_.runTransfer(
            *task, *transfer_dispatcher_, metrics_reporter_, disk_timeout_ms_, host_timeout_ms_);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("load task runner failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("load task runner failed with unknown exception");
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settleLoadLocked(*task, copy_success)) {
            RTP_LLM_LOG_DEBUG("load task settled unsuccessfully");
        }
    }
}

bool BlockTreeLoader::settleLoadLocked(LoadTaskRunner::Task& task, bool copy_success) {
    bool       settlement_success   = copy_success;
    bool       state_settled        = false;
    bool       tree_data_mutated    = false;

    if (copy_success) {
        for (const TransferDescriptor& desc : task.load_descs) {
            const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
            if (resource.transfer_detached) {
                RTP_LLM_LOG_WARNING("load transfer detached before completion, group_set=%zu", desc.group_set_id);
                settlement_success = false;
                continue;
            }
            if (resource.transfer_state != GroupSetTransferState::LOADING) {
                RTP_LLM_LOG_ERROR("load state mismatch during settlement, group_set_id=%zu", desc.group_set_id);
                settlement_success = false;
            }
        }
    }

    for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc      = task.load_descs[desc_index];
        const GroupSetPtr&        group_set = tree_->groupSets()[desc.group_set_id];
        MultiNodeResource source_protection{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        group_set->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (resource.transfer_detached) {
            evictor_.discardDetachedTransfer(desc);
            tree_data_mutated = true;
            state_settled     = true;
            continue;
        }
        if (settlement_success) {
            if (enable_device_cache_) {
                MultiNodeResource target_holder{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}};
                resource.setBlocks(Tier::DEVICE, desc.target_blocks);
                group_set->mapDeviceBlocksToTreeNode(target_holder);
                group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
                group_set->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
                group_set->unreferenceBlocks(source_protection, BlockRefType::BLOCK_CACHE);
                resource.evictFromTier(desc.source_tier);
                task.target_installed[desc_index] = true;
                tree_data_mutated                 = true;
            }
            const bool state_changed = changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE);
            assert(state_changed);
            (void)state_changed;
            state_settled = true;
            if (enable_device_cache_) {
                evictor_.onLoaded(desc.node, desc.group_set_id);
            } else {
                evictor_.refreshCandidate(desc.node, desc.group_set_id);
            }
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group_set=%zu source=%s", desc.group_set_id, tierName(desc.source_tier));
        } else {
            evictor_.refreshCandidate(desc.node, desc.group_set_id);
            state_settled = true;
        }
    }
    settled_(tree_data_mutated, state_settled);
    load_task_runner_.releaseTaskResources(task);
    for (const TransferDescriptor& desc : task.load_descs) {
        if (!load_join_registry_.finish(desc.node, desc.group_set_id, settlement_success)) {
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
