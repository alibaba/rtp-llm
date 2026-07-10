#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/AsyncContextStub.h"  // TODO(block_tree_cache refactor): restore connector/AsyncContext.h
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
    evictor_(component_groups_, copy_engine_, config_.enable_reverse_eviction) {
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

    std::vector<TreeNode*> match_path(
        find_result.path.begin(), find_result.path.begin() + static_cast<ptrdiff_t>(valid_matched_block_count));
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
            auto items = std::make_shared<std::vector<LoadBackItem>>(std::move(lb_items));

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

    auto result = tree_->insertNode(parent, cache_keys, slots);
    TreeNode* leaf = result.leaf;

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

    RTP_LLM_LOG_INFO("BlockTreeCache::reclaimBlocks: reclaimed %d blocks from %s tier", total_reclaimed, tierName(tier));
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

void BlockTreeCache::performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx) {
    auto lb_ctx = std::dynamic_pointer_cast<LoadBackAsyncContext>(ctx);
    bool all_ok = true;

    std::vector<char> item_ok(items.size(), 0);

    for (size_t idx = 0; idx < items.size(); ++idx) {
        auto& item = items[idx];
        auto  gid  = static_cast<size_t>(item.group_id);
        if (gid >= component_groups_.size()) {
            all_ok = false;
            continue;
        }

        auto& group   = component_groups_[gid];
        bool  copy_ok = false;
        if (item.node == nullptr || item.source_blocks.empty() || item.target_device_blocks.empty()) {
            all_ok = false;
            group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
            group->releaseBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
            continue;
        }

        if (item.source_tier == Tier::HOST && group->hostPool()) {
            // Host → GPU (direct H2D)
            auto desc = TransferDescriptor::hostToDevice(
                item.node, item.group_id, item.source_blocks[0], item.target_device_blocks);

            auto status = copy_engine_->submit(desc).status();
            copy_ok     = status == CopyStatus::OK;
            if (!copy_ok) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: H2D FAILED "
                                    "group[%d] node_key=%ld status=%d",
                                    item.group_id,
                                    item.node->cache_key,
                                    static_cast<int>(status));
            }
        } else if (item.source_tier == Tier::DISK && group->hostPool() && group->diskPool()) {
            // Disk → GPU (cross-layer: Disk → temp Host → GPU, Host not cached)
            BlockIdxType temp_host = group->allocateSingleBlock(Tier::HOST);
            if (!isNullBlockIdx(temp_host)) {
                // Step 1: Disk → temp Host
                auto d2h_desc =
                    TransferDescriptor::diskToHost(item.node, item.group_id, item.source_blocks[0], temp_host);

                auto d2h_status = copy_engine_->submit(d2h_desc).status();
                if (d2h_status == CopyStatus::OK) {
                    // Step 2: temp Host → GPU
                    auto h2d_desc = TransferDescriptor::hostToDevice(
                        item.node, item.group_id, temp_host, item.target_device_blocks);

                    auto h2d_status = copy_engine_->submit(h2d_desc).status();
                    copy_ok         = h2d_status == CopyStatus::OK;
                    if (!copy_ok) {
                        RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: Disk2D H2D step FAILED "
                                            "group[%d] node_key=%ld status=%d",
                                            item.group_id,
                                            item.node->cache_key,
                                            static_cast<int>(h2d_status));
                    }
                } else {
                    RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: Disk2D D2H step FAILED "
                                        "group[%d] node_key=%ld status=%d",
                                        item.group_id,
                                        item.node->cache_key,
                                        static_cast<int>(d2h_status));
                }
                group->releaseSingleBlock(Tier::HOST, temp_host);  // Release temp buffer
            }
            if (!copy_ok) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: Disk2D FAILED "
                                    "group[%d] node_key=%ld",
                                    item.group_id,
                                    item.node->cache_key);
            }
        }

        item_ok[idx] = copy_ok ? 1 : 0;
        group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
        if (!copy_ok) {
            all_ok = false;
            // Rollback: release the cache-holding reference on target device blocks.
            group->releaseBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        }
    }

    // Phase 3: re-acquire lock to update tree state
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t idx = 0; idx < items.size(); ++idx) {
            auto& item = items[idx];
            auto  gid  = static_cast<size_t>(item.group_id);
            if (!item_ok[idx] || gid >= component_groups_.size() || item.node == nullptr)
                continue;

            auto& group = component_groups_[gid];
            auto& slot  = item.node->group_slots[gid];

            // Move to DEVICE, then retire source: release its cache-hold (saved ids)
            // before evictFromTier clears the slot. load_back is async (await ctx).
            group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
            group->releaseBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
            group->evictFromTier(item.node, slot, item.source_tier);
            group->tryAddToDeviceHeap(item.node);
        }
    }

    if (lb_ctx) {
        lb_ctx->onTaskComplete(all_ok);
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
    auto         copy_results     = evictor_.performCopy(plan);
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
