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

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>        tree,
                               std::vector<ComponentGroupPtr>    component_groups,
                               std::vector<Component>            components,
                               BlockTreeCacheConfig              config,
                               std::shared_ptr<StorageBackend>   storage_backend,
                               std::shared_ptr<BroadcastManager> broadcast_manager):
    config_(std::move(config)),
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::move(components)),
    copy_engine_(std::make_shared<CopyEngine>(component_groups_, components_)),
    storage_backend_(std::move(storage_backend)),
    broadcast_manager_(std::move(broadcast_manager)),
    evictor_(component_groups_,
             [this](const TransferDescriptor& descriptor) {
                 return executeTransfer(descriptor);
             },
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
    for (const auto& g : component_groups_) {
        if (!g) {
            continue;
        }
        for (const auto& dkv : g->deviceKVGroups()) {
            if (dkv) {
                dkv->setEvictionCallback(
                    [this](int group_id, size_t num_blocks) { return evictForGroup(group_id, num_blocks); });
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

    FunctionRequestPB function_request;
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
    std::lock_guard<std::mutex> lock(mutex_);

    BlockTreeMatchResult result;

    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: empty cache_keys, returning empty result");
        return result;
    }

    auto find_result = tree_->findNode(cache_keys);
    if (find_result.matched_node == nullptr) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: no match found for %zu cache_keys", cache_keys.size());
        return result;
    }

    size_t    valid_matched_block_count = 0;
    TreeNode* best_match                = nullptr;

    std::vector<std::unique_ptr<MatchValidator>> validators;
    validators.reserve(component_groups_.size());
    for (auto& group : component_groups_) {
        validators.push_back(group->createMatchValidator());
    }

    for (size_t i = 0; i < find_result.path.size(); ++i) {
        TreeNode* node      = find_result.path[i];
        bool      all_valid = true;

        // Run ALL validators without short-circuit because SWA validator keeps window state.
        // A node is a reusable match boundary only when every group accepts it.
        for (size_t g = 0; g < component_groups_.size(); ++g) {
            const size_t group_id = static_cast<size_t>(component_groups_[g]->component_group_id);
            GroupSlot&   slot     = node->group_slots[group_id];
            const bool   valid    = validators[g]->validate(node, slot);
            if (!valid) {
                all_valid = false;
            }
        }

        if (all_valid) {
            valid_matched_block_count = i + 1;
            best_match                = node;
        }
    }

    result.matched_node   = best_match;
    result.matched_blocks = valid_matched_block_count;

    // Find FULL group's component_group_id for block_indices collection
    int full_group_id = -1;
    for (const auto& group : component_groups_) {
        if (group->group_type == CacheGroupType::FULL) {
            full_group_id = group->component_group_id;
            break;
        }
    }

    if (full_group_id >= 0) {
        auto gidx = static_cast<size_t>(full_group_id);
        for (size_t i = 0; i < valid_matched_block_count; ++i) {
            TreeNode* node = find_result.path[i];
            if (gidx < node->group_slots.size()) {
                auto& slot = node->group_slots[gidx];
                for (auto block : slot.device_blocks) {
                    if (block != NULL_BLOCK_IDX) {
                        result.block_indices.push_back(block);
                    }
                }
            }
        }
    }

    // Collect per-group device block indices for ALL component groups. Whole-sequence
    // consumers (allocators) pick reuse blocks per group from result.group_block_indices;
    // result.block_indices above keeps the FULL aggregate for the load_back path.
    for (const auto& group : component_groups_) {
        const auto gid     = static_cast<size_t>(group->component_group_id);
        auto&      indices = result.group_block_indices[group->component_group_id];
        for (size_t i = 0; i < valid_matched_block_count; ++i) {
            TreeNode* node = find_result.path[i];
            if (gid >= node->group_slots.size()) {
                continue;
            }
            auto& slot = node->group_slots[gid];
            for (auto block : slot.device_blocks) {
                if (block != NULL_BLOCK_IDX) {
                    indices.push_back(block);
                }
            }
        }
    }

    for (size_t i = 0; i < valid_matched_block_count; ++i) {
        TreeNode* node = find_result.path[i];
        for (auto& group : component_groups_) {
            auto  gid  = static_cast<size_t>(group->component_group_id);
            auto& slot = node->group_slots[gid];
            if (slot.in_device_heap && group->device_heap) {
                group->device_heap->onAccess(node);
            }
        }
    }

    std::vector<TreeNode*> match_path(find_result.path.begin(),
                                      find_result.path.begin() + static_cast<ptrdiff_t>(valid_matched_block_count));
    referenceMatchedDeviceBlocks(match_path, result);

    // Phase 2: load_back — detect and transfer Host/Disk data to GPU
    if (config_.enable_load_back && best_match != nullptr) {
        std::vector<LoadBackItem> lb_items;
        prepareMatchedLoadBack(match_path, lb_items, result);

        // Submit async load_back task
        if (!lb_items.empty()) {
            auto lb_ctx = std::make_shared<LoadBackAsyncContext>();
            lb_ctx->addTask();
            result.async_context = lb_ctx;
            auto items           = std::make_shared<std::vector<LoadBackItem>>(std::move(lb_items));

            taskStarted();
            auto* work_item = new autil::LambdaWorkItem([this, items, lb_ctx]() {
                performLoadBack(std::move(*items), lb_ctx);
                taskFinished();
            });
            auto  err       = thread_pool_->pushWorkItem(work_item);
            if (err != autil::ThreadPool::ERROR_NONE) {
                work_item->destroy();
                for (const auto& item : *items) {
                    auto gid = static_cast<size_t>(item.group_id);
                    if (gid >= component_groups_.size()) {
                        continue;
                    }
                    auto& group = component_groups_[gid];
                    group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
                    group->releaseBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
                }
                lb_ctx->onTaskComplete(false);
                taskFinished();
            }
        }
    }

    for (auto& group : component_groups_) {
        group->finalizeMatchResult(result);
    }

    RTP_LLM_LOG_DEBUG("BlockTreeCache::match: matched %zu blocks, %zu block_indices, "
                      "cache_keys=%zu, tree_nodes=%zu",
                      valid_matched_block_count,
                      result.block_indices.size(),
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

    auto      result = tree_->insertNode(parent, cache_keys, slots);
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
            if (slot.has_device_value()) {
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
    const auto                  gid = static_cast<size_t>(component_group_id);
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
        if (slot.has_device_value() && group->isSlotEvictable(slot, Tier::DEVICE)) {
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
    const auto gid = static_cast<size_t>(component_group_id);
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

void BlockTreeCache::referenceMatchedDeviceBlocks(const std::vector<TreeNode*>& match_path,
                                                  BlockTreeMatchResult&         result) {
    const size_t matched_block_count = match_path.size();
    for (ComponentGroupPtr& group : component_groups_) {
        const size_t ref_count =
            std::min(group->computeReferenceCount(matched_block_count, match_path), matched_block_count);
        const size_t start_idx = matched_block_count - ref_count;
        const size_t gid       = static_cast<size_t>(group->component_group_id);

        GroupBlockSet set{group->component_group_id, Tier::DEVICE};
        for (size_t i = start_idx; i < matched_block_count; ++i) {
            TreeNode* node = match_path[i];
            if (gid >= node->group_slots.size()) {
                continue;
            }
            GroupSlot& slot = node->group_slots[gid];
            if (slot.has_device_value()) {
                set.per_node.push_back(group->getBlocks(slot, Tier::DEVICE));
            }
        }
        if (!set.per_node.empty()) {
            group->referenceBlocks(set);
            result.matched_block_sets.push_back(std::move(set));
        }
    }
}

void BlockTreeCache::prepareMatchedLoadBack(const std::vector<TreeNode*>& match_path,
                                            std::vector<LoadBackItem>&    lb_items,
                                            BlockTreeMatchResult&         result) {
    const size_t matched_block_count = match_path.size();
    for (ComponentGroupPtr& group : component_groups_) {
        const size_t ref_count =
            std::min(group->computeReferenceCount(matched_block_count, match_path), matched_block_count);
        const size_t start_idx = matched_block_count - ref_count;
        const size_t gid       = static_cast<size_t>(group->component_group_id);

        for (size_t i = start_idx; i < matched_block_count; ++i) {
            TreeNode* node = match_path[i];
            if (gid >= node->group_slots.size()) {
                continue;
            }

            GroupSlot& slot = node->group_slots[gid];
            if (slot.has_device_value()) {
                continue;
            }

            Tier source = Tier::NONE;
            if (slot.has_host_value()) {
                source = Tier::HOST;
            } else if (slot.has_disk_value()) {
                source = Tier::DISK;
            }
            if (source == Tier::NONE) {
                continue;
            }

            const auto source_blocks = group->getBlocks(slot, source);
            if (source_blocks.empty()) {
                continue;
            }
            group->referenceBlocks(GroupBlockSet{group->component_group_id, source, {source_blocks}});

            // cache self-allocated path: malloc + incRef. Do NOT write slot yet;
            // the block is installed only after the copy succeeds.
            auto set = group->allocateBlocks(Tier::DEVICE, 1);
            if (set.per_node.empty()) {
                group->unreferenceBlocks(GroupBlockSet{group->component_group_id, source, {source_blocks}});
                RTP_LLM_LOG_WARNING("BlockTreeCache::match: load_back allocate failed "
                                    "group[%d] node_key=%ld",
                                    group->component_group_id,
                                    node->cache_key);
                continue;
            }
            const auto& dev_blocks = set.per_node[0];

            group->referenceBlocks(set);
            result.matched_block_sets.push_back(set);

            lb_items.push_back(LoadBackItem{node, group->component_group_id, source, source_blocks, dev_blocks});

            if (source == Tier::HOST) {
                result.host_load_back_blocks++;
            } else {
                result.disk_load_back_blocks++;
            }
            result.load_back_blocks++;

            for (BlockIdxType block : dev_blocks) {
                if (block != NULL_BLOCK_IDX) {
                    result.block_indices.push_back(block);
                }
            }

            RTP_LLM_LOG_DEBUG("BlockTreeCache::match: load_back from %s "
                              "group[%d] node_key=%ld",
                              tierName(source),
                              group->component_group_id,
                              node->cache_key);
        }
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
