#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <limits>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"
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
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    enable_device_cache_(enable_device_cache),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)),
    load_task_runner_(tree->groupSets(), transfer_dispatcher, *task_pool, metrics_reporter),
    load_context_coordinator_(std::make_shared<LoadContextCoordinator>(
        [this](const std::shared_ptr<LoadAsyncContext>& context) { return commitLoad(context); },
        [this](LoadAsyncContext& context) {
            CacheLockGuard lock(mutex_, metrics_reporter_, "load_abort");
            abortLoadLocked(context.loadDescs(),
                            context.joinedLoads(),
                            context.loadOperations(),
                            /*release_transferred_refs=*/false);
        },
        [task_pool](std::function<void()> cleanup) { return task_pool->submitCompletion(std::move(cleanup)); },
        &metrics_reporter)) {}

BlockTreeMatchResult BlockTreeLoader::matchLocked(const CacheKeysType&               cache_keys,
                                                  const std::function<bool(size_t)>& valid_prefix,
                                                  bool                               enable_storage_backend,
                                                  std::optional<int64_t>             request_id,
                                                  bool                               skip_linear) {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("empty cache_keys, returning empty result");
        return {};
    }

    std::vector<TreeNode*> path = tree_->findNode(cache_keys);
    const bool             has_considered_group_set =
        std::any_of(tree_->groupSets().begin(), tree_->groupSets().end(), [skip_linear](const GroupSetPtr& group_set) {
            return !skip_linear || group_set->groupType() != CacheGroupType::LINEAR;
        });
    if (!has_considered_group_set) {
        return {};
    }
    std::vector<bool> candidate_valid;
    if (!path.empty() && !validMatch(path, candidate_valid, valid_prefix, skip_linear)
        && (!storage_backend_ || !enable_storage_backend)) {
        return {};
    }
    BlockTreeMatchResult result = createMatchResult(
        path, candidate_valid, cache_keys, valid_prefix, enable_storage_backend, request_id, skip_linear);
    RTP_LLM_LOG_DEBUG("matched %zu device blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_device_blocks,
                      cache_keys.size(),
                      tree_->size());
    return result;
}

bool BlockTreeLoader::validMatch(std::vector<TreeNode*>&            path,
                                 std::vector<bool>&                 candidate_valid,
                                 const std::function<bool(size_t)>& valid_prefix,
                                 bool                               skip_linear) const {
    size_t valid_block_count = 0;
    candidate_valid.reserve(path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(tree_->groupSets().size());
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    bool has_considered_group_set = false;
    for (size_t i = 0; i < path.size(); ++i) {
        TreeNode* node             = path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            if (skip_linear && tree_->groupSets()[group_set_id]->groupType() == CacheGroupType::LINEAR) {
                continue;
            }
            has_considered_group_set = true;
            if (!match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                all_groups_valid = false;
            }
        }
        all_groups_valid = all_groups_valid && (!valid_prefix || valid_prefix(i + 1));
        if (all_groups_valid) {
            valid_block_count = i + 1;
        }
        candidate_valid.push_back(all_groups_valid);
    }

    if (!has_considered_group_set) {
        path.clear();
        candidate_valid.clear();
        return false;
    }
    path.resize(valid_block_count);
    candidate_valid.resize(valid_block_count);
    return valid_block_count > 0;
}

BlockIndicesType BlockTreeLoader::matchedBlocksForGroup(std::string_view                      group_tag,
                                                        const std::vector<MultiNodeResource>& matched_resources) const {
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
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

void BlockTreeLoader::collectReuseTimeMetrics(TreeNode*                                    node,
                                              size_t                                       group_set_id,
                                              uint64_t                                     access_seq,
                                              int64_t                                      access_time_us,
                                              std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list) {
    GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier        tier     = resource.getTopTier();
    if (tier == Tier::NONE) {
        return;
    }
    CandidateMeta&                 meta             = resource.candidate_meta;
    const bool                     first_hit        = meta.tier_last_hit_time_us == 0;
    const int64_t                  previous_time_us = first_hit ? meta.tier_enter_time_us : meta.tier_last_hit_time_us;
    BlockTreeCacheReuseTimeMetrics metrics_data;
    metrics_data.tier                        = tier;
    metrics_data.group_type                  = tree_->groupSets()[group_set_id]->groupType();
    metrics_data.reuse_block_hit_idletime_ms = std::max<int64_t>(0, access_time_us - previous_time_us) / 1000;
    if (first_hit) {
        metrics_data.reuse_block_first_hit_idletime_ms = metrics_data.reuse_block_hit_idletime_ms;
    }
    metrics_list.push_back(metrics_data);

    meta.tier_last_hit_time_us = access_time_us;
    meta.last_access_time_us   = access_time_us;
    meta.last_access_seq       = access_seq;
    ++meta.hit_count;
    evictor_.onMatched(node, group_set_id);
}

BlockTreeMatchResult BlockTreeLoader::createMatchResult(std::vector<TreeNode*>&            path,
                                                        const std::vector<bool>&           candidate_valid,
                                                        const CacheKeysType&               cache_keys,
                                                        const std::function<bool(size_t)>& valid_prefix,
                                                        bool                               enable_storage_backend,
                                                        std::optional<int64_t>             request_id,
                                                        bool                               skip_linear) {
    BlockTreeMatchResult result;
    const int64_t        access_time_us = currentTimeUs();
    const uint64_t       access_seq     = evictor_.nextAccessSeq();
    result.reuse_time_metrics.reserve(path.size() * tree_->groupSets().size());

    std::vector<size_t> device_runs(tree_->groupSets().size(), 0);
    std::vector<size_t> device_host_runs(tree_->groupSets().size(), 0);
    std::vector<size_t> load_begins;
    load_begins.reserve(tree_->groupSets().size());
    for (const auto& group_set : tree_->groupSets()) {
        load_begins.push_back(path.size() - std::min(group_set->computeReuseBlockCount(path.size()), path.size()));
    }
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i]->unaccounted_writeback_version >= 0) {
            result.reuse_accounting.unaccounted_blocks.push_back({i, path[i]->unaccounted_writeback_version});
        }
        const size_t candidate_count        = i + 1;
        bool         all_groups_device      = true;
        bool         all_groups_device_host = true;
        for (size_t group_id = 0; group_id < tree_->groupSets().size(); ++group_id) {
            // PD decode handoff: LINEAR groups never gate prefix matching when
            // the caller supplies the linear state out of band.
            if (skip_linear && tree_->groupSets()[group_id]->groupType() == CacheGroupType::LINEAR) {
                continue;
            }
            const auto& resource       = path[i]->group_set_resources[group_id];
            const bool  device_ready   = resource.hasCompleteDeviceValue();
            device_runs[group_id]      = device_ready ? device_runs[group_id] + 1 : 0;
            const bool host_will_load  = i >= load_begins[group_id] && resource.hasTier(Tier::HOST);
            device_host_runs[group_id] = device_ready || host_will_load ? device_host_runs[group_id] + 1 : 0;
            const size_t required =
                std::min(tree_->groupSets()[group_id]->computeReuseBlockCount(candidate_count), candidate_count);
            all_groups_device      = all_groups_device && device_runs[group_id] >= required;
            all_groups_device_host = all_groups_device_host && device_host_runs[group_id] >= required;
        }
        if (candidate_valid[i]) {
            if (all_groups_device) {
                result.matched_device_blocks = candidate_count;
            }
            if (all_groups_device_host) {
                result.matched_device_host_blocks = candidate_count;
            }
        }
    }

    std::vector<MultiNodeResource>              request_refs;
    std::vector<MultiNodeResource>              load_refs;
    std::vector<TransferDescriptor>             pending_load_descs;
    std::vector<bool>                           joined_loads;
    std::vector<std::shared_ptr<LoadOperation>> operations;
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
        if (skip_linear && group_set->groupType() == CacheGroupType::LINEAR) {
            continue;
        }
        const size_t       ready_reuse_count =
            std::min(group_set->computeReuseBlockCount(result.matched_device_blocks), result.matched_device_blocks);
        MultiNodeResource matched_device_resource{group_set_id, Tier::DEVICE};
        for (size_t i = result.matched_device_blocks - ready_reuse_count; i < result.matched_device_blocks; ++i) {
            collectReuseTimeMetrics(path[i], group_set_id, access_seq, access_time_us, result.reuse_time_metrics);
            const GroupSetResource& resource = path[i]->group_set_resources[group_set_id];
            matched_device_resource.node_blocks.emplace_back(nullptr, resource.getBlocks(Tier::DEVICE));
        }
        if (!matched_device_resource.node_blocks.empty()) {
            group_set->referenceBlocks(matched_device_resource);
            result.matched_device_resources.push_back(std::move(matched_device_resource));
        }

        const size_t host_prefix      = result.matched_device_host_blocks;
        const size_t host_reuse_count = std::min(group_set->computeReuseBlockCount(host_prefix), host_prefix);
        const auto   append_range     = [&](size_t begin, size_t end) {
            for (size_t i = std::max(begin, result.matched_device_blocks); i < end; ++i) {
                GroupSetResource&  resource    = path[i]->group_set_resources[group_set_id];
                const Tier         source_tier = resource.getTopTier();
                TreeNode* const    source_node = source_tier == Tier::DEVICE ? nullptr : path[i];
                const bool initial_owner       = source_node && resource.transfer_state == GroupSetTransferState::IDLE;
                TransferDescriptor desc{source_node,
                                        group_set_id,
                                        i,
                                        source_tier,
                                        Tier::DEVICE,
                                        resource.getBlocks(source_tier),
                                        initial_owner};
                if (resource.load_operation) {
                    desc.target_blocks = resource.load_operation->descriptor().target_blocks;
                    appendBlocks(request_refs, group_set_id, Tier::DEVICE, desc.target_blocks);
                } else {
                    if (source_tier == Tier::DEVICE) {
                        appendBlocks(request_refs, group_set_id, source_tier, desc.source_blocks);
                    } else {
                        appendBlocks(load_refs, group_set_id, source_tier, desc.source_blocks);
                        if (initial_owner) {
                            resource.transfer_state = GroupSetTransferState::LOAD_PENDING;
                            evictor_.suspendCandidate(path[i], group_set_id, source_tier);
                        } else {
                            tree_->referenceNodeForNonOwnerTransfer(source_node);
                        }
                    }
                }
                collectReuseTimeMetrics(path[i], group_set_id, access_seq, access_time_us, result.reuse_time_metrics);
                pending_load_descs.emplace_back(std::move(desc));
                joined_loads.emplace_back(resource.load_operation != nullptr);
                operations.emplace_back(resource.load_operation);
            }
        };
        // Outside the normal load window, fallback only pins existing device values.
        append_range(host_prefix - host_reuse_count, std::min(host_prefix, load_begins[group_set_id]));
        append_range(load_begins[group_set_id], path.size());
    }
    StorageRequest storage_request;
    if (enable_storage_backend && storage_backend_ && path.size() < cache_keys.size()) {
        storage_request = makeStorageRequest(cache_keys, path.size(), skip_linear);
    }
    const bool use_storage     = !storage_request.empty();
    storage_request.request_id = request_id;
    if (!pending_load_descs.empty() || use_storage) {
        referenceBlocks(tree_->groupSets(), request_refs);
        referenceBlocks(tree_->groupSets(), load_refs, BlockTreeRefType::LOAD);
        result.async_context = load_context_coordinator_->create(pending_load_descs,
                                                                 joined_loads,
                                                                 operations,
                                                                 path.size(),
                                                                 result.matched_device_blocks,
                                                                 result.matched_device_host_blocks,
                                                                 valid_prefix,
                                                                 use_storage ? storage_backend_ : nullptr,
                                                                 std::move(storage_request));
        if (result.async_context == nullptr) {
            abortLoadLocked(pending_load_descs, joined_loads, operations, /*release_transferred_refs=*/true);
        }
    }
    return result;
}

StorageRequest BlockTreeLoader::makeStorageRequest(const CacheKeysType& cache_keys,
                                                   size_t               local_matched_blocks_num,
                                                   bool                 skip_linear) const {
    StorageRequest request{std::make_shared<CacheKeysType>(cache_keys),
                           std::vector<std::vector<StorageBlockHandle>>(cache_keys.size()),
                           local_matched_blocks_num};
    for (auto& key_handles : request.handles) {
        for (const auto& group_set : tree_->groupSets()) {
            if (skip_linear && group_set->groupType() == CacheGroupType::LINEAR) {
                continue;
            }
            for (const auto& tag : group_set->groupTags()) {
                key_handles.push_back({tag, NULL_BLOCK_IDX});
            }
        }
    }
    return request;
}

bool BlockTreeLoader::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(context);
    if (load_context == nullptr) {
        RTP_LLM_LOG_WARNING("context is not owned by BlockTreeCache");
        return false;
    }
    return load_context->abortPending();
}

void BlockTreeLoader::shutdown() {
    load_context_coordinator_->shutdown();
}

bool BlockTreeLoader::commitLoad(const std::shared_ptr<LoadAsyncContext>& context) {
    if (!context->newLoadIndices().empty()) {
        CacheLockGuard lock(mutex_, metrics_reporter_, "load_commit");
        if (std::none_of(context->newLoadIndices().begin(), context->newLoadIndices().end(), [&context](size_t index) {
                const auto& desc = context->loadDescs()[index];
                return desc.isCurrentOwner() && desc.node->group_set_resources[desc.group_set_id].transfer_detached;
            })) {
            if (context->createLoadOperations(*tree_)) {
                auto task = LoadTask::create(context, host_timeout_ms_, disk_timeout_ms_);
                context->setSettlementReadyCallback(
                    [this, task](const auto& ready) { scheduleContextSettlement(task, ready); });
                if (load_task_runner_.submit(task)) {
                    return true;
                }
            }
            abortLoadLocked(context->loadDescs(), context->joinedLoads(), context->loadOperations(), false);
            return false;
        }
    }
    context->setSettlementReadyCallback([this](const auto& ready) { scheduleContextSettlement(nullptr, ready); });
    if (!context->newLoadIndices().empty()) {
        context->cancelUnsubmittedTransfers();
    }
    return true;
}

void BlockTreeLoader::abortLoadLocked(const std::vector<TransferDescriptor>&             load_descs,
                                      const std::vector<bool>&                           joined_loads,
                                      const std::vector<std::shared_ptr<LoadOperation>>& operations,
                                      bool                                               release_transferred_refs) {
    bool                                  device_refs_released = false;
    std::vector<MultiNodeResource>        request_refs;
    std::vector<MultiNodeResource>        load_refs;
    std::unordered_map<TreeNode*, size_t> pending_nodes;
    std::vector<TransferDescriptor>       detached_descs;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (joined_loads[desc_index]) {
            if (release_transferred_refs) {
                appendBlocks(request_refs, desc.group_set_id, Tier::DEVICE, desc.target_blocks);
                device_refs_released = true;
            }
            continue;
        }
        if (desc.source_tier == Tier::DEVICE) {
            if (release_transferred_refs) {
                appendBlocks(request_refs, desc.group_set_id, desc.source_tier, desc.source_blocks);
                device_refs_released = true;
            }
            continue;
        }

        appendBlocks(load_refs, desc.group_set_id, desc.source_tier, desc.source_blocks);
        const auto& operation = operations[desc_index];
        if (operation) {
            auto& slot = desc.node->group_set_resources[desc.group_set_id].load_operation;
            if (slot == operation) {
                slot.reset();
            }
            appendBlocks(load_refs, desc.group_set_id, Tier::DEVICE, desc.target_blocks);
        }
        if (!desc.isCurrentOwner()) {
            ++pending_nodes[desc.node];
            continue;
        }
        if (desc.node->group_set_resources[desc.group_set_id].transfer_detached) {
            detached_descs.push_back(desc);
            pending_nodes.try_emplace(desc.node, 0);
            continue;
        }
        const GroupSetTransferState expected_state =
            operation ? GroupSetTransferState::LOADING : GroupSetTransferState::LOAD_PENDING;
        if (!changeTransferState(desc.node, desc.group_set_id, expected_state, GroupSetTransferState::IDLE)) {
            RTP_LLM_LOG_WARNING(
                "load rollback state mismatch, group_set=%zu source=%s", desc.group_set_id, tierName(desc.source_tier));
        } else {
            evictor_.admitCandidate(
                desc.node, desc.group_set_id, desc.node->group_set_resources[desc.group_set_id].getTopTier());
        }
    }
    unreferenceBlocks(tree_->groupSets(), load_refs, BlockTreeRefType::LOAD);
    unreferenceBlocks(tree_->groupSets(), request_refs);
    evictor_.discardDetachedTransfer(detached_descs);
    tree_->removePendingNodes(pending_nodes);
    if (device_refs_released) {
        settled_(true);
    }
}

void BlockTreeLoader::scheduleContextSettlement(const std::shared_ptr<LoadTask>&         task,
                                                const std::shared_ptr<LoadAsyncContext>& context) {
    const auto&    descriptors   = task ? task->load_descs : context->loadDescs();
    const uint64_t transfer_id   = task ? task->completionTransfer().traceId() : context->workflowId();
    auto stage_context = CacheStageContext::forWorkflow(
        metrics_reporter_, descriptors, "load", context->requestId().value_or(-1), transfer_id);
    const bool transfer_success = context->localTransfersSucceeded();
    auto       timers           = CompletionStageTimers::create(stage_context);
    auto       settle           = [this, task, context, stage_context, timers, transfer_success]() {
        timers.finishQueue(transfer_success);
        bool                                   success = false;
        block_tree_cache_detail::ScopeRollback finish([&] { timers.finish(success); });
        auto                                   execute =
            TransferStageTimer::scope(stage_context, "completion_execute", "business_completion", transfer_success);
        bool settlement_success = transfer_success;
        bool host_success       = context->hostTransfersSucceeded();
        if (task) {
            {
                CacheLockGuard lock(mutex_, "load_settle", stage_context);
                settlement_success = settleLoadLocked(*task, settlement_success, host_success, stage_context);
            }
            auto notify = TransferStageTimer::scope(
                stage_context, "completion_join_notify", "business_completion", transfer_success);
            for (const auto& operation : task->operations) {
                operation->complete(operation->descriptor().source_tier == Tier::HOST ? host_success :
                                                                                        settlement_success);
            }
        }
        {
            auto notify =
                TransferStageTimer::scope(stage_context, "completion_notify", "business_completion", transfer_success);
            if (!context->settle(settlement_success, host_success)) {
                RTP_LLM_LOG_WARNING("failed to publish load context settlement, context_id=%lu", context->contextId());
            }
            success = context->success();
            notify.setSuccess(success);
        }
        execute.setSuccess(success);
    };
    if (!task_pool_->submitCompletion(settle)) {
        RTP_LLM_LOG_WARNING("load completion queue is closed; settling context inline");
        settle();
    }
}

bool BlockTreeLoader::validateLoadTaskLocked(const LoadTask& task) const {
    bool valid = true;
    for (const TransferDescriptor& desc : task.load_descs) {
        if (!desc.isCurrentOwner()) {
            continue;
        }
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (resource.transfer_detached) {
            RTP_LLM_LOG_WARNING("load transfer detached before completion, group_set=%zu", desc.group_set_id);
            valid = false;
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::LOADING) {
            RTP_LLM_LOG_ERROR("load state mismatch during settlement, group_set_id=%zu", desc.group_set_id);
            valid = false;
        }
    }
    return valid;
}

bool BlockTreeLoader::settleLoadLocked(LoadTask&                task,
                                       bool                     aggregate_success,
                                       bool&                    host_success,
                                       const CacheStageContext& stage_context) {
    std::vector<BlockTreeEvictor::InvalidationSource> corrupted;
    const auto collect_corrupted = [this, &corrupted](const TransferTask& transfer, const TransferResult& result) {
        for (size_t index = 0; index < result.block_errors.size(); ++index) {
            const auto& desc = transfer.descriptors()[index];
            if (result.block_errors[index] != CacheCopyError::CRC_MISMATCH || !desc.node->parent) {
                continue;
            }
            auto& resource = desc.node->group_set_resources[desc.group_set_id];
            if (desc.generation != resource.generation || resource.transfer_detached
                || resource.getBlocks(desc.source_tier) != desc.source_blocks) {
                continue;
            }
            CacheKeysType cache_keys;
            for (TreeNode* node = desc.node; node != tree_->root(); node = node->parent) {
                cache_keys.push_back(node->cache_key);
            }
            std::reverse(cache_keys.begin(), cache_keys.end());
            corrupted.push_back({std::move(cache_keys), desc.group_set_id, desc.source_tier});
        }
    };
    // Capture both tiers before resource settlement or a cascading drop can destroy a source node.
    collect_corrupted(task.host_to_device_task, task.host_result);
    collect_corrupted(task.disk_to_device_task, task.disk_result);

    const bool valid                            = validateLoadTaskLocked(task);
    host_success                                = host_success && valid;
    const bool                            settlement_success = aggregate_success && valid;
    bool                                  state_settled = false;
    std::unordered_map<TreeNode*, size_t> pending_nodes;
    std::vector<MultiNodeResource>        target_load_refs;
    std::vector<MultiNodeResource>        target_cache_refs;
    std::vector<MultiNodeResource>        source_cache_refs;

    {
        auto update = TransferStageTimer::scope(
            stage_context, "completion_tree_update", "business_completion", settlement_success);
        std::vector<TransferDescriptor> detached_descs;
        for (size_t desc_index = 0; desc_index < task.load_descs.size(); ++desc_index) {
            const TransferDescriptor& desc      = task.load_descs[desc_index];
            const GroupSetPtr&        group_set = tree_->groupSets()[desc.group_set_id];
            const auto&               operation = task.operations[desc_index];
            GroupSetResource&         resource  = desc.node->group_set_resources[desc.group_set_id];
            if (resource.load_operation == operation) {
                resource.load_operation.reset();
            }
            operation->unreferenceSource();
            appendBlocks(target_load_refs, desc.group_set_id, Tier::DEVICE, desc.target_blocks);

            if (!desc.isCurrentOwner()) {
                ++pending_nodes[desc.node];
                state_settled = true;
                continue;
            }
            if (resource.transfer_detached) {
                detached_descs.push_back(desc);
                pending_nodes.try_emplace(desc.node, 0);
                state_settled = true;
                continue;
            }
            if (desc.source_tier == Tier::HOST ? host_success : settlement_success) {
                const bool install_target =
                    enable_device_cache_
                    && (group_set->groupType() != CacheGroupType::FULL || desc.node->parent == tree_->root()
                        || desc.node->parent->group_set_resources[desc.group_set_id].hasCompleteDeviceValue());
                if (install_target) {
                    resource.setBlocks(Tier::DEVICE, desc.target_blocks);
                    appendBlocks(target_cache_refs, desc.group_set_id, Tier::DEVICE, desc.target_blocks);
                    appendBlocks(source_cache_refs, desc.group_set_id, desc.source_tier, desc.source_blocks);
                    resource.evictFromTier(desc.source_tier);
                }
                const bool state_changed = changeTransferState(
                    desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE);
                assert(state_changed);
                (void)state_changed;
                state_settled = true;
                if (install_target) {
                    evictor_.onLoaded(desc.node, desc.group_set_id);
                } else {
                    evictor_.admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
                }
                continue;
            }

            // Ordinary copy failures still retain a reusable source.
            if (!changeTransferState(
                    desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
                RTP_LLM_LOG_WARNING(
                    "loading state mismatch, group_set=%zu source=%s", desc.group_set_id, tierName(desc.source_tier));
            } else {
                evictor_.admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
                state_settled = true;
            }
        }
        referenceBlocks(tree_->groupSets(), target_cache_refs, BlockTreeRefType::CACHE);
        unreferenceBlocks(tree_->groupSets(), source_cache_refs, BlockTreeRefType::CACHE);
        evictor_.discardDetachedTransfer(detached_descs);
    }
    {
        auto refs = TransferStageTimer::scope(
            stage_context, "completion_release_refs", "business_completion", settlement_success);
        for (const auto& resource : target_load_refs) {
            tree_->groupSets()[resource.group_set_id]->unreferenceBlocks(resource, BlockTreeRefType::LOAD);
        }
    }
    {
        auto cleanup =
            TransferStageTimer::scope(stage_context, "completion_cleanup", "business_completion", settlement_success);
        tree_->removePendingNodes(pending_nodes);
        evictor_.invalidateSources(corrupted);
    }
    {
        auto watermark =
            TransferStageTimer::scope(stage_context, "completion_watermark", "business_completion", settlement_success);
        settled_(state_settled);
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
    tree_->refreshPublishedState(node);
    return true;
}

}  // namespace rtp_llm
