#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTransferConverter.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// AsyncContext for load_back: waits until all copy tasks complete.
class LoadBackAsyncContext: public AsyncContext {
public:
    void addTask() {
        remaining_++;
    }
    void onTaskComplete(bool ok) {
        if (!ok)
            all_success_.store(false);
        if (--remaining_ <= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void waitDone() override {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return remaining_.load() <= 0; });
    }
    bool done() const override {
        return remaining_.load() <= 0;
    }
    bool success() const override {
        return all_success_.load();
    }

private:
    std::atomic<int>        remaining_{0};
    std::atomic<bool>       all_success_{true};
    std::mutex              mu_;
    std::condition_variable cv_;
};

}  // anonymous namespace

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>         tree,
                               std::vector<ComponentGroupPtr>     component_groups,
                               std::vector<Component>             components,
                               BlockTreeCacheConfig               config,
                               std::shared_ptr<StorageBackend>    storage_backend,
                               std::shared_ptr<BroadcastManager>  broadcast_manager,
                               std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups,
                               std::vector<PerTagMapping>         per_tag_mapping):
    config_(std::move(config)),
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::move(components)),
    per_tag_device_groups_(std::move(per_tag_device_groups)),
    per_tag_mapping_(std::move(per_tag_mapping)),
    aggregated_(!per_tag_mapping_.empty()),
    copy_engine_(std::make_shared<CopyEngine>(component_groups_, components_)),
    storage_backend_(std::move(storage_backend)),
    broadcast_manager_(std::move(broadcast_manager)),
    evictor_(
        component_groups_,
        [this](const TransferDescriptor& descriptor) { return executeTransfer(descriptor); },
        config_.enable_reverse_eviction) {
    // Validate tier dependencies: Disk requires Host (design doc section 2.7)
    if (config_.enable_disk_cache && !config_.enable_memory_cache) {
        throw std::invalid_argument("BlockTreeCache: enable_disk_cache requires enable_memory_cache = true");
    }
    evictor_.init(components_);

    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        static_cast<size_t>(config_.eviction_thread_pool_size), 1000, nullptr, "BlockTreeEvictionPool");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: failed to start eviction thread pool, size=%d",
                          config_.eviction_thread_pool_size);
    }
    RTP_LLM_LOG_INFO("BlockTreeCache: constructed with %zu component groups, %zu components, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     component_groups_.size(),
                     components_.size(),
                     config_.eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_memory_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const auto& g : component_groups_) {
        RTP_LLM_LOG_INFO("BlockTreeCache:   group[%d] type=%s host_pool=%s disk_pool=%s",
                         g->component_group_id,
                         cacheGroupTypeName(g->group_type),
                         g->hostPool() ? "enabled" : "null",
                         g->diskPool() ? "enabled" : "null");
    }
    // Wire this cache into each owned DeviceKVCacheGroup so ensureFreeBlocks() can call
    // back into evictForGroup() for cross-group device eviction via std::function callback.
    // In aggregated mode the per-tag registry (which also covers NON_REUSABLE tags) is the
    // full set of device groups; in legacy mode they live under each ComponentGroup.
    auto wireEviction = [this](const DeviceKVCacheGroupPtr& dkv) {
        if (dkv) {
            dkv->setEvictionCallback(
                [this](int group_id, size_t num_blocks) { return evictForGroup(group_id, num_blocks); });
        }
    };
    if (aggregated_) {
        for (const auto& dkv : per_tag_device_groups_) {
            wireEviction(dkv);
        }
    } else {
        for (const auto& g : component_groups_) {
            if (!g) {
                continue;
            }
            for (const auto& dkv : g->deviceKVGroups()) {
                wireEviction(dkv);
            }
        }
    }
}

BlockTreeCache::~BlockTreeCache() {
    RTP_LLM_LOG_INFO("BlockTreeCache: destroying, waiting for pending tasks...");
    waitForPendingTasks();
    if (thread_pool_) {
        thread_pool_->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        thread_pool_->join();
    }
    RTP_LLM_LOG_INFO("BlockTreeCache: destroyed");
}

CopyStatus BlockTreeCache::executeTransfer(const TransferDescriptor& descriptor) {
    if (!copy_engine_) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::executeTransfer: copy engine is not initialized");
        return CopyStatus::INVALID_ARGS;
    }

    TransferHandle handle = copy_engine_->submit(descriptor);
    return handle.status();
}

bool BlockTreeCache::broadcastTransfer(const ::MemoryOperationRequestPB& request, int timeout_ms) const {
    if (!broadcast_manager_) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: broadcast manager is not initialized");
        return false;
    }
    if (request.copy_items_size() == 0 || timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: invalid request, item_count=%d, timeout_ms=%d",
                            request.copy_items_size(),
                            timeout_ms);
        return false;
    }

    const size_t worker_count = broadcast_manager_->workerNum();
    if (worker_count == 0) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: no worker configured");
        return false;
    }

    FunctionRequestPB         function_request;
    MemoryOperationRequestPB* memory_request = function_request.mutable_mem_request();
    if (memory_request == nullptr) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: failed to create memory request");
        return false;
    }
    memory_request->CopyFrom(request);
    std::vector<FunctionRequestPB> requests(worker_count, function_request);

    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> broadcast_result =
        broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests,
            timeout_ms,
            [](const std::shared_ptr<RpcService::Stub>&    stub,
               const std::shared_ptr<grpc::ClientContext>& context,
               const FunctionRequestPB&                    rpc_request,
               grpc::CompletionQueue*                      completion_queue) {
                return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
            });
    if (!broadcast_result) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: failed to start broadcast");
        return false;
    }

    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: worker RPC failed");
        return false;
    }

    const std::vector<FunctionResponsePB> responses = broadcast_result->responses();
    if (responses.size() != worker_count) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: response count mismatch, expected=%zu, actual=%zu",
                            worker_count,
                            responses.size());
        return false;
    }
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const FunctionResponsePB& response = responses[rank];
        if (!response.has_mem_response() || !response.mem_response().success()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: worker transfer failed, rank=%zu", rank);
            return false;
        }
    }
    return true;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType& cache_keys) {
    BlockTreeMatchResult result;
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: empty cache_keys, returning empty result");
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeFindResult         tree_find_result = tree_->findNode(cache_keys);
    if (tree_find_result.matched_node == nullptr) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: no match found for %zu cache_keys", cache_keys.size());
        return result;
    }

    size_t                                       valid_matched_block_count = 0;
    TreeNode*                                    best_matched_node         = nullptr;
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(component_groups_.size());
    for (ComponentGroupPtr& component_group : component_groups_) {
        match_validators.push_back(component_group->createMatchValidator());
    }
    for (size_t i = 0; i < tree_find_result.path.size(); ++i) {
        TreeNode* path_node        = tree_find_result.path[i];
        bool      all_groups_valid = true;
        for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
            const size_t component_group_index =
                static_cast<size_t>(component_groups_[group_index]->component_group_id);
            GroupSlot& group_slot  = path_node->group_slots[component_group_index];
            const bool group_valid = match_validators[group_index]->validate(path_node, group_slot);
            if (!group_valid) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_matched_block_count = i + 1;
            best_matched_node         = path_node;
        }
    }

    result.matched_node   = best_matched_node;
    result.matched_blocks = valid_matched_block_count;
    std::vector<TreeNode*> matched_path(tree_find_result.path.begin(),
                                        tree_find_result.path.begin()
                                            + static_cast<ptrdiff_t>(valid_matched_block_count));
    if (config_.enable_load_back && best_matched_node != nullptr) {
        result.load_back_ticket = std::make_shared<LoadBackTicket>(
            [this](const std::vector<PendingLoadBackItem>& items) { return commitLoadBack(items); },
            [this](const std::vector<PendingLoadBackItem>& items) { abortLoadBack(items); });
    }
    prepareMatchedBlocks(matched_path, result);

    RTP_LLM_LOG_DEBUG("BlockTreeCache::match: matched %zu blocks, cache_keys=%zu, tree_nodes=%zu",
                      valid_matched_block_count,
                      cache_keys.size(),
                      tree_->nodeCount());
    return result;
}

void BlockTreeCache::insert(TreeNode*                                  parent,
                            const CacheKeysType&                       cache_keys,
                            const std::vector<std::vector<GroupSlot>>& slots) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    // Aggregated mode: the caller supplies slots indexed by per-tag gid (one device block
    // per tag). Translate them into per-ComponentGroup slots whose device_blocks are indexed
    // by the group's local device pool index before handing them to the tree.
    std::vector<std::vector<GroupSlot>>        translated;
    const std::vector<std::vector<GroupSlot>>* effective_slots = &slots;
    if (aggregated_) {
        const size_t num_cg = component_groups_.size();
        translated.assign(cache_keys.size(), std::vector<GroupSlot>(num_cg));
        for (size_t i = 0; i < cache_keys.size(); ++i) {
            for (size_t cg = 0; cg < num_cg; ++cg) {
                const size_t pools = component_groups_[cg] ? component_groups_[cg]->devicePools().size() : 0;
                translated[i][cg].device_blocks.assign(pools, NULL_BLOCK_IDX);
            }
            if (i >= slots.size()) {
                continue;
            }
            const auto& per_tag = slots[i];
            for (size_t tag_gid = 0; tag_gid < per_tag.size() && tag_gid < per_tag_mapping_.size(); ++tag_gid) {
                const auto m = per_tag_mapping_[tag_gid];
                if (m.component_group_id < 0) {
                    continue;
                }
                const auto& src = per_tag[tag_gid].device_blocks;
                if (src.empty() || isNullBlockIdx(src.front())) {
                    continue;
                }
                auto& dst = translated[i][static_cast<size_t>(m.component_group_id)].device_blocks;
                if (static_cast<size_t>(m.local_pool_index) < dst.size()) {
                    dst[static_cast<size_t>(m.local_pool_index)] = src.front();
                }
            }
        }
        effective_slots = &translated;
    }

    auto      result = tree_->insertNode(parent, cache_keys, *effective_slots);
    TreeNode* leaf   = result.leaf;

    // incRef cache-hold on new nodes' device blocks (balanced by releaseBlocks on
    // eviction). Reused nodes keep theirs; their demoted data comes from load_back.
    for (const auto& inserted : result.inserted_nodes) {
        TreeNode* node = inserted.node;
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            auto& slot = node->group_slots[gid];
            if (slot.has_value(Tier::DEVICE)) {
                auto blocks = group->getBlocks(slot, Tier::DEVICE);
                group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}});
            }
        }
    }

    // Only the leaf is offered to the device heap (group Leaf/Any policy decides).
    // Non-leaf nodes enter later via parent promotion on eviction.
    for (auto& group : component_groups_) {
        auto gid = static_cast<size_t>(group->component_group_id);
        if (gid < leaf->group_slots.size()) {
            group->tryAddToDeviceHeap(leaf);
        }
    }

    // Update overlap for existing nodes along the path (walk up from leaf)
    for (TreeNode* node = leaf->parent; node != nullptr && node != tree_->root(); node = node->parent) {
        for (auto& group : component_groups_) {
            auto gid = static_cast<size_t>(group->component_group_id);
            if (gid < node->group_slots.size()) {
                group->updateOnInsertOverlap(node, node->group_slots[gid]);
            }
        }
    }

    RTP_LLM_LOG_DEBUG(
        "BlockTreeCache::insert: inserted %zu cache_keys, tree_nodes=%zu", cache_keys.size(), tree_->nodeCount());

    // Phase 3: check watermark after insert
    checkWatermark();
}

std::shared_ptr<HostBlockPool> BlockTreeCache::hostPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->hostPool() : nullptr;
}

std::shared_ptr<DiskBlockPool> BlockTreeCache::diskPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->diskPool() : nullptr;
}

// reclaimBlocks: directly reclaim (drop) num_blocks blocks at the given tier.
// Forces target_tier=NONE, so no copy/demotion happens; block content is
// discarded instead of being moved down to a lower tier.
int BlockTreeCache::reclaimBlocks(size_t num_blocks, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(tier)) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::reclaimBlocks: tier %s is disabled, skipping", tierName(tier));
        return 0;
    }

    int total_reclaimed = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        auto eviction_move = evictor_.chooseVictim(tier);
        if (!eviction_move.has_value()) {
            RTP_LLM_LOG_DEBUG("BlockTreeCache::reclaimBlocks: no more candidates at tier=%s, reclaimed %d/%zu blocks",
                              tierName(tier),
                              total_reclaimed,
                              num_blocks);
            break;
        }

        // Force direct reclaim: no demotion, no copy; drop block content directly.
        eviction_move->target_tier = Tier::NONE;

        if (submitEvictionLocked(*eviction_move)) {
            ++total_reclaimed;
        }
    }

    RTP_LLM_LOG_INFO(
        "BlockTreeCache::reclaimBlocks: reclaimed %d blocks from %s tier", total_reclaimed, tierName(tier));
    return total_reclaimed;
}

size_t BlockTreeCache::evictableBlocksNum(int component_group_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const int                   resolved = resolveComponentGroupId(component_group_id);
    if (resolved < 0) {
        return 0;
    }
    const auto gid = static_cast<size_t>(resolved);
    if (gid >= component_groups_.size()) {
        return 0;
    }
    const auto& group = component_groups_[gid];
    if (!group->device_heap) {
        return 0;
    }
    size_t count = 0;
    for (TreeNode* node : group->device_heap->nodes()) {
        if (node == nullptr || gid >= node->group_slots.size()) {
            continue;
        }
        const auto& slot = node->group_slots[gid];
        if (slot.has_value(Tier::DEVICE) && group->isSlotEvictable(slot, Tier::DEVICE)) {
            ++count;
        }
    }
    return count;
}

int BlockTreeCache::evictForGroup(int component_group_id, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const int resolved = resolveComponentGroupId(component_group_id);
    if (resolved < 0) {
        return 0;
    }
    const auto gid = static_cast<size_t>(resolved);
    if (gid >= component_groups_.size()) {
        return 0;
    }
    auto& group = component_groups_[gid];

    int total_reclaimed = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        auto eviction_move = group->driveEviction(1, Tier::DEVICE);
        if (!eviction_move.has_value()) {
            break;
        }
        // Force direct reclaim: drop block content, do not demote to a lower tier.
        eviction_move->target_tier = Tier::NONE;
        if (submitEvictionLocked(*eviction_move)) {
            ++total_reclaimed;
        }
    }
    RTP_LLM_LOG_DEBUG("BlockTreeCache::evictForGroup: group[%d] reclaimed %d/%zu device blocks",
                      component_group_id,
                      total_reclaimed,
                      num_blocks);
    return total_reclaimed;
}

void BlockTreeCache::releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& set : sets) {
        auto gid = static_cast<size_t>(set.component_group_id);
        if (gid < component_groups_.size()) {
            component_groups_[gid]->unreferenceBlocks(set);
        }
    }
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count = tree_->nodeCount();
    for (const auto& group : component_groups_) {
        if (group->device_heap)
            stats.device_heap_total_size += group->device_heap->size();
        if (group->host_heap)
            stats.host_heap_total_size += group->host_heap->size();
        if (group->disk_heap)
            stats.disk_heap_total_size += group->disk_heap->size();
    }
    return stats;
}

void BlockTreeCache::waitForPendingTasks() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cv_.wait(lock, [this] { return pending_tasks_.load() <= 0; });
}

void BlockTreeCache::taskStarted() {
    pending_tasks_++;
}

void BlockTreeCache::taskFinished() {
    int remaining = --pending_tasks_;
    if (remaining <= 0) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

void BlockTreeCache::prepareMatchedBlocks(const std::vector<TreeNode*>& matched_path, BlockTreeMatchResult& result) {
    const size_t matched_block_count = matched_path.size();
    for (ComponentGroupPtr& component_group : component_groups_) {
        const size_t reference_count =
            std::min(component_group->computeReferenceCount(matched_block_count, matched_path), matched_block_count);
        const size_t  start_index           = matched_block_count - reference_count;
        const size_t  component_group_index = static_cast<size_t>(component_group->component_group_id);
        GroupBlockSet matched_device_blocks{component_group->component_group_id, Tier::DEVICE};

        for (size_t i = start_index; i < matched_block_count; ++i) {
            TreeNode* path_node = matched_path[i];
            if (component_group_index >= path_node->group_slots.size()) {
                continue;
            }
            GroupSlot& group_slot = path_node->group_slots[component_group_index];
            if (group_slot.has_value(Tier::DEVICE)) {
                const std::vector<BlockIdxType> device_blocks = component_group->getBlocks(group_slot, Tier::DEVICE);
                bool                            collected_device_block = false;
                if (aggregated_) {
                    for (size_t tag_group_index = 0; tag_group_index < per_tag_mapping_.size(); ++tag_group_index) {
                        const PerTagMapping& tag_mapping = per_tag_mapping_[tag_group_index];
                        if (tag_mapping.component_group_id != component_group->component_group_id
                            || tag_mapping.local_pool_index < 0) {
                            continue;
                        }
                        const size_t local_pool_index = static_cast<size_t>(tag_mapping.local_pool_index);
                        if (local_pool_index >= device_blocks.size()
                            || device_blocks[local_pool_index] == NULL_BLOCK_IDX) {
                            continue;
                        }
                        result.group_block_indices[static_cast<int>(tag_group_index)].push_back(
                            device_blocks[local_pool_index]);
                        collected_device_block = true;
                    }
                } else {
                    BlockIndicesType& collected_block_indices =
                        result.group_block_indices[component_group->component_group_id];
                    for (BlockIdxType device_block : device_blocks) {
                        if (device_block != NULL_BLOCK_IDX) {
                            collected_block_indices.push_back(device_block);
                            collected_device_block = true;
                        }
                    }
                }

                matched_device_blocks.per_node.push_back(device_blocks);
                if (collected_device_block && group_slot.in_device_heap && component_group->device_heap != nullptr) {
                    component_group->device_heap->onAccess(path_node);
                }
                continue;
            }

            if (result.load_back_ticket == nullptr) {
                continue;
            }

            Tier source_tier = Tier::NONE;
            if (group_slot.has_value(Tier::HOST)) {
                source_tier = Tier::HOST;
            } else if (group_slot.has_value(Tier::DISK)) {
                source_tier = Tier::DISK;
            }
            if (source_tier == Tier::NONE) {
                continue;
            }

            const std::vector<BlockIdxType> source_blocks = component_group->getBlocks(group_slot, source_tier);
            if (source_blocks.empty()) {
                continue;
            }

            // Keep the source non-evictable until the ticket is committed or aborted.
            component_group->referenceBlocks(
                GroupBlockSet{component_group->component_group_id, source_tier, {source_blocks}});

            result.load_back_ticket->items().push_back(
                PendingLoadBackItem{path_node, component_group->component_group_id, source_tier, source_blocks});

            if (source_tier == Tier::HOST) {
                result.host_load_back_blocks++;
            } else {
                result.disk_load_back_blocks++;
            }
            result.load_back_blocks++;

            RTP_LLM_LOG_DEBUG("BlockTreeCache::match: planned load_back from %s "
                              "group[%d] node_key=%ld",
                              tierName(source_tier),
                              component_group->component_group_id,
                              path_node->cache_key);
        }

        if (!matched_device_blocks.per_node.empty()) {
            component_group->referenceBlocks(matched_device_blocks);
            result.matched_block_sets.push_back(std::move(matched_device_blocks));
        }
    }
}

std::shared_ptr<AsyncContext> BlockTreeCache::commitLoadBack(const std::vector<PendingLoadBackItem>& items) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (items.empty()) {
        return nullptr;
    }

    class LoadBackRollbackGuard {
    public:
        LoadBackRollbackGuard(std::vector<ComponentGroupPtr>&         component_groups,
                              const std::vector<PendingLoadBackItem>& pending_items,
                              const std::vector<LoadBackItem>&        allocated_items):
            component_groups_(component_groups), pending_items_(pending_items), allocated_items_(allocated_items) {}

        ~LoadBackRollbackGuard() {
            if (!rollback_required_) {
                return;
            }
            for (const LoadBackItem& item : allocated_items_) {
                if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
                    continue;
                }
                ComponentGroupPtr& component_group = component_groups_[static_cast<size_t>(item.group_id)];
                if (component_group != nullptr) {
                    component_group->releaseBlocks(
                        GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
                }
            }
            for (const PendingLoadBackItem& item : pending_items_) {
                if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
                    continue;
                }
                ComponentGroupPtr& component_group = component_groups_[static_cast<size_t>(item.group_id)];
                if (component_group != nullptr) {
                    component_group->unreferenceBlocks(
                        GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
                }
            }
        }
        void dismiss() {
            rollback_required_ = false;
        }

    private:
        std::vector<ComponentGroupPtr>&         component_groups_;
        const std::vector<PendingLoadBackItem>& pending_items_;
        const std::vector<LoadBackItem>&        allocated_items_;
        bool                                    rollback_required_{true};
    };

    auto async_items = std::make_shared<std::vector<LoadBackItem>>();
    async_items->reserve(items.size());
    LoadBackRollbackGuard rollback_guard(component_groups_, items, *async_items);
    bool                  allocation_succeeded = true;
    for (const PendingLoadBackItem& item : items) {
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size() || item.node == nullptr
            || item.source_tier == Tier::NONE || item.source_blocks.empty()) {
            allocation_succeeded = false;
            break;
        }
        const size_t       component_group_index = static_cast<size_t>(item.group_id);
        ComponentGroupPtr& component_group       = component_groups_[component_group_index];
        if (component_group->devicePoolCount() == 0) {
            allocation_succeeded = false;
            break;
        }

        // Allocate the target after allocator malloc succeeds.
        // Keep the slot unchanged until the copy succeeds.
        GroupBlockSet allocated_device_blocks = component_group->allocateBlocks(Tier::DEVICE, 1);
        if (allocated_device_blocks.per_node.size() != 1
            || allocated_device_blocks.per_node[0].size() != component_group->devicePoolCount()) {
            component_group->releaseBlocks(allocated_device_blocks);
            allocation_succeeded = false;
            break;
        }
        const std::vector<BlockIdxType>& target_device_blocks = allocated_device_blocks.per_node[0];
        async_items->push_back(
            LoadBackItem{item.node, item.group_id, item.source_tier, item.source_blocks, target_device_blocks});
    }

    if (!allocation_succeeded || async_items->size() != items.size()) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: device allocation failed, "
                            "rolled back all %zu load_back items",
                            items.size());
        return nullptr;
    }

    auto lb_ctx = std::make_shared<LoadBackAsyncContext>();
    lb_ctx->addTask();

    taskStarted();
    autil::LambdaWorkItem*              work_item = new autil::LambdaWorkItem([this, async_items, lb_ctx]() {
        performLoadBack(std::move(*async_items), lb_ctx);
        taskFinished();
    });
    const autil::ThreadPool::ERROR_TYPE error     = thread_pool_->pushWorkItem(work_item);
    if (error != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        lb_ctx->onTaskComplete(false);
        taskFinished();
        return lb_ctx;
    }
    rollback_guard.dismiss();
    return lb_ctx;
}

void BlockTreeCache::abortLoadBack(const std::vector<PendingLoadBackItem>& items) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : items) {
        auto gid = static_cast<size_t>(item.group_id);
        if (gid >= component_groups_.size()) {
            continue;
        }
        // Release the source reference taken while preparing the match.
        // No device block was allocated, so there is nothing else to undo.
        component_groups_[gid]->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
    }
}

bool BlockTreeCache::executeLoadBackTransferBatch(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) {
    if (descriptors.empty()) {
        return true;
    }

    if (broadcast_manager_ == nullptr) {
        for (const TransferDescriptor& descriptor : descriptors) {
            const CopyStatus status = executeTransfer(descriptor);
            if (status != CopyStatus::OK) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::executeLoadBackTransferBatch: local transfer failed, "
                                    "group=%d source=%s target=%s status=%d",
                                    descriptor.component_group_id,
                                    tierName(descriptor.source_tier),
                                    tierName(descriptor.target_tier),
                                    static_cast<int>(status));
                return false;
            }
        }
        return true;
    }

    MemoryOperationRequestPB request;
    for (const TransferDescriptor& descriptor : descriptors) {
        const bool appended = BlockTreeTransferConverter::appendTransfer(descriptor, request);
        if (!appended) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::executeLoadBackTransferBatch: failed to encode transfer, "
                                "group=%d source=%s target=%s",
                                descriptor.component_group_id,
                                tierName(descriptor.source_tier),
                                tierName(descriptor.target_tier));
            return false;
        }
    }
    return broadcastTransfer(request, timeout_ms);
}

void BlockTreeCache::performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx) {
    std::shared_ptr<LoadBackAsyncContext> load_back_context = std::dynamic_pointer_cast<LoadBackAsyncContext>(ctx);
    std::vector<BlockIdxType>             staging_host_blocks(items.size(), NULL_BLOCK_IDX);
    std::vector<TransferDescriptor>       disk_to_host_descriptors;
    std::vector<TransferDescriptor>       host_to_device_descriptors;
    bool                                  prepared = !items.empty();

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        LoadBackItem& item = items[item_index];
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid group id, group=%d", item.group_id);
            prepared = false;
            continue;
        }

        ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (item.node == nullptr || item.source_blocks.size() != 1 || item.target_device_blocks.empty()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid item, group=%d", item.group_id);
            prepared = false;
            continue;
        }

        BlockIdxType source_host_block = NULL_BLOCK_IDX;
        if (item.source_tier == Tier::HOST && group->hostPool() != nullptr) {
            source_host_block = item.source_blocks[0];
        } else if (item.source_tier == Tier::DISK && group->hostPool() != nullptr && group->diskPool() != nullptr) {
            source_host_block = group->allocateSingleBlock(Tier::HOST);
            if (!isNullBlockIdx(source_host_block)) {
                staging_host_blocks[item_index] = source_host_block;
                disk_to_host_descriptors.push_back(
                    TransferDescriptor::diskToHost(item.group_id, item.source_blocks[0], source_host_block));
            }
        }

        if (isNullBlockIdx(source_host_block)) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: failed to prepare source, group=%d source=%s",
                                item.group_id,
                                tierName(item.source_tier));
            prepared = false;
            continue;
        }
        host_to_device_descriptors.push_back(
            TransferDescriptor::hostToDevice(item.group_id, source_host_block, item.target_device_blocks));
    }

    bool copy_success = prepared;
    if (copy_success) {
        copy_success =
            executeLoadBackTransferBatch(disk_to_host_descriptors, config_.memory_cache_disk_sync_timeout_ms);
    }
    if (copy_success) {
        copy_success = executeLoadBackTransferBatch(host_to_device_descriptors, config_.memory_cache_sync_timeout_ms);
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        LoadBackItem& item = items[item_index];
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
            continue;
        }
        ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (!isNullBlockIdx(staging_host_blocks[item_index])) {
            group->releaseSingleBlock(Tier::HOST, staging_host_blocks[item_index]);
        }
        group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
        if (!copy_success) {
            group->releaseBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        }
    }

    if (copy_success) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (LoadBackItem& item : items) {
            ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
            GroupSlot&         slot  = item.node->group_slots[static_cast<size_t>(item.group_id)];
            group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
            group->releaseBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
            group->evictFromTier(item.node, slot, item.source_tier);
            group->tryAddToDeviceHeap(item.node);
        }
    }

    if (load_back_context != nullptr) {
        load_back_context->onTaskComplete(copy_success);
    }
}

bool BlockTreeCache::submitEvictionLocked(EvictionMove& eviction_move) {
    if (eviction_move.target_tier != Tier::NONE && !config_.isTierEnabled(eviction_move.target_tier)) {
        eviction_move.target_tier = Tier::NONE;
    }

    auto plan = evictor_.buildPlan(eviction_move);
    if (!plan.has_value()) {
        return false;
    }

    if (!plan->needsCopy()) {
        BlockTreeEvictor::CopyResultSet results;
        results.primary_success = true;
        results.cascade_success.assign(plan->cascade_moves.size(), true);
        evictor_.complete(*tree_, *plan, results);
        return true;
    }

    auto plan_ptr = std::make_shared<BlockTreeEvictor::EvictionPlan>(std::move(*plan));
    taskStarted();
    auto* work_item = new autil::LambdaWorkItem([this, plan_ptr]() { performEvictionCopy(*plan_ptr); });
    auto  err       = thread_pool_->pushWorkItem(work_item);
    if (err != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        evictor_.rollbackPreparedPlan(*plan_ptr);
        taskFinished();
        return false;
    }
    return true;
}

void BlockTreeCache::performEvictionCopy(const BlockTreeEvictor::EvictionPlan& plan) {
    BlockTreeEvictor::CopyResultSet copy_results;
    if (broadcast_manager_ == nullptr) {
        copy_results = evictor_.performCopy(plan);
    } else {
        MemoryOperationRequestPB request;
        const bool               request_ready = buildEvictionTransferRequest(plan, request);
        const bool copy_success      = request_ready && broadcastTransfer(request, evictionTransferTimeoutMs(plan));
        copy_results.primary_success = copy_success;
        copy_results.cascade_success.assign(plan.cascade_moves.size(), copy_success);
    }

    bool         copy_ok          = copy_results.primary_success;
    CacheKeyType remote_cache_key = 0;
    int          remote_group_id  = -1;
    if (copy_ok && plan.primary.node != nullptr) {
        remote_cache_key = plan.primary.node->cache_key;
        remote_group_id  = plan.primary.component_group_id;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        evictor_.complete(*tree_, plan, copy_results);
    }

    if (copy_ok && config_.enable_remote_cache && remote_group_id >= 0) {
        evictor_.writeRemoteThrough(storage_backend_, remote_cache_key, remote_group_id);
    }
    taskFinished();
}

bool BlockTreeCache::buildEvictionTransferRequest(const BlockTreeEvictor::EvictionPlan& plan,
                                                  MemoryOperationRequestPB&             request) const {
    TransferDescriptor primary_descriptor;
    if (!BlockTreeEvictor::buildTransferDescriptor(plan.primary, primary_descriptor)
        || !BlockTreeTransferConverter::appendTransfer(primary_descriptor, request)) {
        return false;
    }

    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        TransferDescriptor cascade_descriptor;
        if (!BlockTreeEvictor::buildTransferDescriptor(cascade_move, cascade_descriptor)
            || !BlockTreeTransferConverter::appendTransfer(cascade_descriptor, request)) {
            return false;
        }
    }
    return true;
}

int BlockTreeCache::evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const {
    bool uses_disk = plan.primary.source_tier == Tier::DISK || plan.primary.target_tier == Tier::DISK;
    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        if (cascade_move.source_tier == Tier::DISK || cascade_move.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    if (!uses_disk) {
        return config_.memory_cache_sync_timeout_ms;
    }
    return std::max(config_.memory_cache_sync_timeout_ms, config_.memory_cache_disk_sync_timeout_ms);
}

void BlockTreeCache::checkWatermark() {
    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        auto wm = config_.watermarkForTier(tier);
        if (wm.ratio <= 0.0 || !config_.isTierEnabled(tier))
            continue;

        for (auto& group : component_groups_) {
            auto victims = evictor_.chooseWatermarkVictims(*group, tier, wm.ratio);
            for (auto& eviction_move : victims) {
                submitEvictionLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
