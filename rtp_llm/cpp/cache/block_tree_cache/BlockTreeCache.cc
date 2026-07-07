#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <stdexcept>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
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
    copy_engine_(std::make_shared<CopyEngine>()),
    storage_backend_(std::move(storage_backend)),
    broadcast_manager_(std::move(broadcast_manager)) {
    // Validate tier dependencies: Disk requires Host (design doc section 2.7)
    if (config_.enable_disk_cache && !config_.enable_memory_cache) {
        throw std::invalid_argument("BlockTreeCache: enable_disk_cache requires enable_memory_cache = true");
    }

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
        RTP_LLM_LOG_INFO(
            "BlockTreeCache:   group[%d] type=%s host_pool=%s disk_pool=%s",
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

    size_t    valid_matched_blocks = 0;
    TreeNode* best_match           = nullptr;

    std::vector<std::unique_ptr<MatchValidator>> validators;
    validators.reserve(component_groups_.size());
    for (auto& group : component_groups_) {
        validators.push_back(group->createMatchValidator());
    }

    for (size_t i = 0; i < find_result.path.size(); ++i) {
        TreeNode* node      = find_result.path[i];
        bool      all_valid = true;

        // Run ALL validators (no short-circuit) — SWA validator is stateful.
        // Only FULL validator gates match validity; SWA/LINEAR track state only.
        for (size_t g = 0; g < component_groups_.size(); ++g) {
            auto& slot  = node->group_slots[static_cast<size_t>(component_groups_[g]->component_group_id)];
            bool  valid = validators[g]->validate(node, slot);
            if (component_groups_[g]->group_type == CacheGroupType::FULL && !valid) {
                all_valid = false;
            }
        }

        if (all_valid) {
            valid_matched_blocks = i + 1;
            best_match           = node;
        }
    }

    result.matched_node   = best_match;
    result.matched_blocks = valid_matched_blocks;

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
        for (size_t i = 0; i < valid_matched_blocks; ++i) {
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

    for (size_t i = 0; i < valid_matched_blocks; ++i) {
        TreeNode* node = find_result.path[i];
        for (auto& group : component_groups_) {
            auto  gid  = static_cast<size_t>(group->component_group_id);
            auto& slot = node->group_slots[gid];
            if (slot.in_device_heap && group->device_heap) {
                group->device_heap->onAccess(node);
            }
        }
    }

    // Phase 2: path lock — per-group reference strategy (SWA uses window lock)
    // Use device_pools_ for per-block reference counting
    {
        std::vector<TreeNode*> match_path(find_result.path.begin(),
                                          find_result.path.begin() + static_cast<ptrdiff_t>(valid_matched_blocks));

        for (auto& group : component_groups_) {
            size_t ref_count = group->computeReferenceCount(valid_matched_blocks, match_path);
            size_t start_idx = (valid_matched_blocks > ref_count) ? (valid_matched_blocks - ref_count) : 0;
            auto   gid       = static_cast<size_t>(group->component_group_id);
            for (size_t i = start_idx; i < valid_matched_blocks; ++i) {
                TreeNode* node = match_path[i];
                if (gid < node->group_slots.size()) {
                    auto& slot = node->group_slots[gid];
                    if (slot.has_device_value()) {
                        group->referenceDeviceBlocks(slot.device_blocks);
                    }
                }
            }
        }
    }

    // Phase 2: load_back — detect and transfer Host/Disk data to GPU
    if (config_.enable_load_back && best_match != nullptr) {
        std::vector<LoadBackItem> lb_items;

        for (size_t i = 0; i < valid_matched_blocks; ++i) {
            TreeNode* node = find_result.path[i];
            for (auto& group : component_groups_) {
                auto  gid  = static_cast<size_t>(group->component_group_id);
                auto& slot = node->group_slots[gid];

                if (slot.has_device_value())
                    continue;  // Already on GPU

                Tier source = Tier::NONE;
                if (slot.has_host_value()) {
                    source = Tier::HOST;
                } else if (slot.has_disk_value()) {
                    source = Tier::DISK;
                }
                if (source == Tier::NONE)
                    continue;

                // Allocate device blocks (if allocator available)
                std::vector<BlockIdxType> dev_blocks;
                if (device_block_allocator_) {
                    size_t count = group->component_indices.empty() ? 1 : group->component_indices.size();
                    dev_blocks   = device_block_allocator_(group->component_group_id, count);
                    if (dev_blocks.empty()) {
                        RTP_LLM_LOG_WARNING("BlockTreeCache::match: load_back allocator failed "
                                            "group[%d] node_key=%ld",
                                            group->component_group_id,
                                            node->cache_key);
                        continue;
                    }
                    slot.device_blocks = dev_blocks;
                }

                lb_items.push_back({node, group->component_group_id, source, dev_blocks});

                if (source == Tier::HOST) {
                    result.host_load_back_blocks++;
                } else {
                    result.disk_load_back_blocks++;
                }
                result.load_back_blocks++;

                // Add new device blocks to block_indices
                for (auto b : dev_blocks) {
                    if (b != NULL_BLOCK_IDX)
                        result.block_indices.push_back(b);
                }

                RTP_LLM_LOG_DEBUG("BlockTreeCache::match: load_back from %s "
                                  "group[%d] node_key=%ld",
                                  tierName(source),
                                  group->component_group_id,
                                  node->cache_key);
            }
        }

        // Submit async load_back task
        if (!lb_items.empty() && device_block_allocator_) {
            auto lb_ctx = std::make_shared<LoadBackAsyncContext>();
            lb_ctx->addTask();
            result.async_context = lb_ctx;

            taskStarted();
            auto* work_item = new autil::LambdaWorkItem([this, items = std::move(lb_items), lb_ctx]() {
                performLoadBack(std::move(items), lb_ctx);
                taskFinished();
            });
            auto  err       = thread_pool_->pushWorkItem(work_item);
            if (err != autil::ThreadPool::ERROR_NONE) {
                work_item->destroy();
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
                      valid_matched_blocks,
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

    TreeNode* leaf = tree_->insertNode(parent, cache_keys, slots);

    // Commit data and add leaf to heaps
    for (auto& group : component_groups_) {
        auto gid = static_cast<size_t>(group->component_group_id);
        if (gid < leaf->group_slots.size()) {
            auto& slot = leaf->group_slots[gid];
            if (slot.has_device_value()) {
                group->commitInsertData(leaf, slot, slot.device_blocks);
            }
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

int BlockTreeCache::evict(size_t num_blocks, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reject eviction on disabled tiers
    if (!config_.isTierEnabled(tier)) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::evict: tier %s is disabled, skipping", tierName(tier));
        return 0;
    }

    int total_evicted = 0;

    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        bool found_candidate = false;

        for (auto& group : component_groups_) {
            auto eviction_result = group->driveEviction(1, tier);
            if (!eviction_result.has_value()) {
                continue;
            }

            found_candidate = true;
            auto er         = eviction_result.value();

            RTP_LLM_LOG_DEBUG("BlockTreeCache::evict: Phase 1 selected candidate, "
                              "group[%d] type=%s tier=%s target=%s node_key=%ld",
                              er.component_group_id,
                              cacheGroupTypeName(group->group_type),
                              tierName(er.source_tier),
                              tierName(er.target_tier),
                              er.node ? er.node->cache_key : 0);

            submitEviction(er);

            total_evicted++;
            break;
        }

        if (!found_candidate) {
            RTP_LLM_LOG_DEBUG("BlockTreeCache::evict: no more candidates at tier=%s, "
                              "evicted %d/%zu blocks",
                              tierName(tier),
                              total_evicted,
                              num_blocks);
            break;
        }
    }

    RTP_LLM_LOG_INFO("BlockTreeCache::evict: evicted %d blocks from %s tier", total_evicted, tierName(tier));
    return total_evicted;
}

bool BlockTreeCache::executeTierCopy(int component_group_id, Tier source_tier, Tier target_tier,
                                     const std::vector<BlockIdxType>& source_blocks,
                                     BlockIdxType target_block) {
    if (!copy_engine_ || source_blocks.empty() || isNullBlockIdx(target_block))
        return false;

    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= component_groups_.size())
        return false;

    auto& group = component_groups_[gid];
    auto  hp    = group->hostPool();
    auto  dp    = group->diskPool();

    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        if (!hp) return false;
        std::vector<MemoryBlockLayerTagSlot> layer_slots;
        for (int ci : group->component_indices) {
            if (ci >= 0 && static_cast<size_t>(ci) < components_.size()) {
                for (const auto& lts : components_[static_cast<size_t>(ci)].memory_block_layer_tag_slots)
                    layer_slots.push_back(lts);
            }
        }
        std::sort(layer_slots.begin(), layer_slots.end(),
                  [](const MemoryBlockLayerTagSlot& a, const MemoryBlockLayerTagSlot& b) {
                      return a.layer_id < b.layer_id;
                  });
        DeviceBufferResolver resolver =
            device_buffer_resolver_ ? device_buffer_resolver_
                                   : DeviceBufferResolver([](int, BlockIdxType) { return BlockInfo{}; });
        return copy_engine_->deviceToHost(source_blocks, target_block, layer_slots, resolver, *hp);
    } else if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        if (!hp || !dp || isNullBlockIdx(source_blocks[0])) return false;
        return copy_engine_->hostToDisk(source_blocks[0], target_block, *hp, *dp);
    } else if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
        if (!hp || !dp || isNullBlockIdx(source_blocks[0])) return false;
        return copy_engine_->diskToHost(source_blocks[0], target_block, *hp, *dp);
    }
    return false;
}

void BlockTreeCache::releaseBlocksFromPool(int component_group_id, Tier tier,
                                           const std::vector<BlockIdxType>& blocks) {
    if (blocks.empty())
        return;

    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= component_groups_.size())
        return;

    auto& group = component_groups_[gid];
    if (tier == Tier::DEVICE) {
        group->releaseDeviceBlocks(blocks);
    } else if (tier == Tier::HOST) {
        if (auto hp = group->hostPool()) {
            for (auto b : blocks)
                if (!isNullBlockIdx(b)) hp->free(b);
        }
    } else if (tier == Tier::DISK) {
        if (auto dp = group->diskPool()) {
            for (auto b : blocks)
                if (!isNullBlockIdx(b)) dp->free(b);
        }
    }
}

void BlockTreeCache::freeTargetBlock(int component_group_id, Tier target_tier, BlockIdxType block) {
    if (isNullBlockIdx(block))
        return;

    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= component_groups_.size())
        return;

    auto& group = component_groups_[gid];
    if (target_tier == Tier::HOST) {
        if (auto hp = group->hostPool()) hp->free(block);
    } else if (target_tier == Tier::DISK) {
        if (auto dp = group->diskPool()) dp->free(block);
    }
}

void BlockTreeCache::setTargetSlot(ComponentGroupPtr& group, GroupSlot& slot,
                                   TreeNode* node, Tier target_tier, BlockIdxType target_block) {
    if (isNullBlockIdx(target_block))
        return;
    if (target_tier == Tier::HOST) {
        slot.host_block = target_block;
        // Establish cache hold: block becomes tree-visible (refcount 0 -> 1).
        if (auto hp = group->hostPool()) hp->incRef(target_block);
        group->tryAddToHostHeap(node);
    } else if (target_tier == Tier::DISK) {
        slot.disk_slot = target_block;
        // Establish cache hold: block becomes tree-visible (refcount 0 -> 1).
        if (auto dp = group->diskPool()) dp->incRef(target_block);
        group->tryAddToDiskHeap(node);
    }
}

void BlockTreeCache::performEvictionCopy(EvictionResult er) {
    // Phase 2: perform data movement (no lock held)
    bool copy_ok = false;

    if (er.node != nullptr) {
        const auto& desc  = er.transfer;
        const auto* entry = desc.entries.empty() ? nullptr : &desc.entries.front();

        // Build source_blocks from transfer entry
        std::vector<BlockIdxType> source_blocks;
        if (entry) {
            if (desc.source_tier == Tier::DEVICE) {
                source_blocks = entry->device_blocks;
            } else if (desc.source_tier == Tier::HOST) {
                source_blocks = {entry->host_block};
            } else if (desc.source_tier == Tier::DISK) {
                source_blocks = {entry->disk_block};
            }
        }

        copy_ok = executeTierCopy(er.component_group_id, desc.source_tier, desc.target_tier,
                                  source_blocks, er.target_block);

        if (!copy_ok) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performEvictionCopy: copy FAILED "
                                "group[%d] node_key=%ld %s→%s",
                                er.component_group_id, er.node->cache_key,
                                tierName(desc.source_tier), tierName(desc.target_tier));
        } else {
            RTP_LLM_LOG_DEBUG("BlockTreeCache::performEvictionCopy: copy OK "
                              "group[%d] node_key=%ld %s→%s target_block=%d",
                              er.component_group_id, er.node->cache_key,
                              tierName(desc.source_tier), tierName(desc.target_tier),
                              er.target_block);
        }
    }

    // Phase 3: completion callback (re-acquires lock)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (copy_ok) {
            onEvictionComplete(er, /*cascade_with_copy=*/true);
        } else {
            // Rollback: free allocated target block, restore source tier heap
            freeTargetBlock(er.component_group_id, er.target_tier, er.target_block);
            auto gid = static_cast<size_t>(er.component_group_id);
            if (gid < component_groups_.size() && er.node) {
                if (er.source_tier == Tier::DEVICE) {
                    component_groups_[gid]->tryAddToDeviceHeap(er.node);
                } else if (er.source_tier == Tier::HOST) {
                    component_groups_[gid]->tryAddToHostHeap(er.node);
                }
            }
        }
    }

    // Remote write-through (async, after local copy completes, only if copy succeeded)
    if (copy_ok && config_.enable_remote_cache && storage_backend_ && er.node != nullptr) {
        auto key = std::to_string(er.node->cache_key) + "_g" + std::to_string(er.component_group_id);
        std::vector<std::pair<std::string, std::vector<char>>> items;
        items.emplace_back(std::move(key), std::vector<char>{});
        if (!items.back().second.empty()) {
            storage_backend_->batchWrite(items);
            RTP_LLM_LOG_DEBUG("BlockTreeCache::performEvictionCopy: remote write-through "
                              "group[%d] node_key=%ld",
                              er.component_group_id,
                              er.node->cache_key);
        } else {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performEvictionCopy: remote write-through SKIPPED "
                                "(no data serialization yet) group[%d] node_key=%ld",
                                er.component_group_id,
                                er.node->cache_key);
        }
    }

    taskFinished();
}

void BlockTreeCache::onEvictionComplete(const EvictionResult& er, bool cascade_with_copy) {
    if (er.node == nullptr)
        return;

    auto gid = static_cast<size_t>(er.component_group_id);
    if (gid >= component_groups_.size())
        return;

    auto& group = component_groups_[gid];
    auto& slot  = er.node->group_slots[gid];

    RTP_LLM_LOG_DEBUG("BlockTreeCache::onEvictionComplete: group[%d] node_key=%ld "
                      "source=%s target=%s cascade_copy=%d",
                      er.component_group_id,
                      er.node->cache_key,
                      tierName(er.source_tier),
                      tierName(er.target_tier),
                      cascade_with_copy);

    group->evictFromTier(er.node, slot, er.source_tier);

    // Release source blocks back to their pools
    releaseBlocksFromPool(er.component_group_id, er.source_tier, er.blocks_to_release);

    // Set target tier data from CopyEngine result
    setTargetSlot(group, slot, er.node, er.target_tier, er.target_block);

    cascadeEviction(er.node, er.component_group_id, er.source_tier, cascade_with_copy);
    finalizeEviction(er.node);
}

Tier BlockTreeCache::nextLowerTier(Tier tier) const {
    switch (tier) {
        case Tier::DEVICE: return config_.isTierEnabled(Tier::HOST) ? Tier::HOST : Tier::NONE;
        case Tier::HOST:   return config_.isTierEnabled(Tier::DISK) ? Tier::DISK : Tier::NONE;
        default:           return Tier::NONE;
    }
}

void BlockTreeCache::cascadeEviction(TreeNode* node, int source_group_id, Tier tier,
                                     bool cascade_with_copy) {
    auto lower_groups = groupsBelowPriority(source_group_id);
    if (lower_groups.empty())
        return;

    const Tier cascade_target = cascade_with_copy ? nextLowerTier(tier) : Tier::NONE;

    for (int gid : lower_groups) {
        auto gidx = static_cast<size_t>(gid);
        if (gidx >= component_groups_.size() || gidx >= node->group_slots.size())
            continue;

        auto& lower_group = component_groups_[gidx];
        auto& slot        = node->group_slots[gidx];

        // Collect source blocks
        std::vector<BlockIdxType> source_blocks;
        switch (tier) {
            case Tier::DEVICE:
                if (slot.has_device_value()) source_blocks = slot.device_blocks;
                break;
            case Tier::HOST:
                if (slot.has_host_value()) source_blocks = {slot.host_block};
                break;
            case Tier::DISK:
                if (slot.has_disk_value()) source_blocks = {slot.disk_slot};
                break;
            default:
                break;
        }

        if (source_blocks.empty())
            continue;

        if (cascade_target != Tier::NONE) {
            // ---- Cascade with copy: synchronously copy to next tier ----
            BlockIdxType target_block = NULL_BLOCK_IDX;

            // Allocate target block
            if (cascade_target == Tier::HOST) {
                auto pool = lower_group->hostPool();
                if (pool) {
                    auto alloc = pool->malloc();
                    if (alloc.has_value())
                        target_block = alloc.value();
                }
            } else if (cascade_target == Tier::DISK) {
                auto pool = lower_group->diskPool();
                if (pool) {
                    auto slot_opt = pool->malloc();
                    if (slot_opt.has_value())
                        target_block = slot_opt.value();
                }
            }

            if (isNullBlockIdx(target_block)) {
                // Allocation failed — skip this group (don't evict)
                RTP_LLM_LOG_WARNING("BlockTreeCache::cascadeEviction: target alloc failed "
                                    "group[%d] tier %s→%s, skipping",
                                    gid, tierName(tier), tierName(cascade_target));
                continue;
            }

            // Perform copy via unified helper
            bool copy_ok = executeTierCopy(gid, tier, cascade_target, source_blocks, target_block);

            if (!copy_ok) {
                // Copy failed — free target, skip this group (don't evict)
                freeTargetBlock(gid, cascade_target, target_block);

                RTP_LLM_LOG_WARNING("BlockTreeCache::cascadeEviction: copy failed "
                                    "group[%d] tier %s→%s node_key=%ld, skipping",
                                    gid, tierName(tier), tierName(cascade_target),
                                    node->cache_key);
                continue;
            }

            // Copy succeeded: evict source, release source blocks, set target data
            lower_group->evictFromTier(node, slot, tier);
            releaseBlocksFromPool(gid, tier, source_blocks);
            setTargetSlot(lower_group, slot, node, cascade_target, target_block);

            RTP_LLM_LOG_DEBUG("BlockTreeCache::cascadeEviction: copied group[%d] "
                              "tier %s→%s node_key=%ld target_block=%d",
                              gid, tierName(tier), tierName(cascade_target),
                              node->cache_key, target_block);
        } else {
            // ---- Cascade with direct release: no copy ----
            lower_group->evictFromTier(node, slot, tier);
            releaseBlocksFromPool(gid, tier, source_blocks);

            RTP_LLM_LOG_DEBUG("BlockTreeCache::cascadeEviction: released group[%d] "
                              "tier=%s node_key=%ld (direct release)",
                              gid, tierName(tier), node->cache_key);
        }
    }
}

void BlockTreeCache::finalizeEviction(TreeNode* node) {
    if (shouldDeleteNode(node)) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::finalizeEviction: deleting empty node key=%ld", node->cache_key);
        TreeNode* parent = node->parent;
        tree_->removeNode(node);
        tree_->removeEmptyAncestors(parent, reusableGroupIds());
        if (parent && parent != tree_->root() && parent->parent != nullptr) {
            for (auto& g : component_groups_) {
                g->tryAddToDeviceHeap(parent);
            }
        }
    } else {
        if (node->parent && node->parent != tree_->root()) {
            TreeNode* parent = node->parent;
            for (auto& g : component_groups_) {
                g->tryAddToDeviceHeap(parent);
            }
        }
    }
}

bool BlockTreeCache::shouldDeleteNode(const TreeNode* node) const {
    if (node == nullptr || node == tree_->root() || !node->children.empty())
        return false;
    for (const auto& group : component_groups_) {
        auto gidx = static_cast<size_t>(group->component_group_id);
        if (gidx < node->group_slots.size() && !node->group_slots[gidx].is_empty()) {
            return false;
        }
    }
    return true;
}

std::vector<int> BlockTreeCache::allGroupIds() const {
    std::vector<int> ids;
    for (const auto& group : component_groups_) {
        ids.push_back(group->component_group_id);
    }
    return ids;
}

std::vector<int> BlockTreeCache::reusableGroupIds() const {
    std::vector<int> ids;
    for (const auto& group : component_groups_) {
        ids.push_back(group->component_group_id);
    }
    return ids;
}

std::vector<int> BlockTreeCache::groupsBelowPriority(int source_group_id) const {
    CacheGroupType source_type = CacheGroupType::FULL;
    for (const auto& group : component_groups_) {
        if (group->component_group_id == source_group_id) {
            source_type = group->group_type;
            break;
        }
    }
    std::vector<int> result;
    for (const auto& group : component_groups_) {
        bool below = false;
        switch (source_type) {
            case CacheGroupType::FULL:
                below = (group->group_type == CacheGroupType::SWA || group->group_type == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::SWA:
                below = (group->group_type == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::LINEAR:
                below = false;
                break;
        }
        if (below)
            result.push_back(group->component_group_id);
    }
    return result;
}

bool BlockTreeCache::isEvictable(TreeNode* node, int group_id) const {
    if (node == nullptr)
        return false;
    auto gidx = static_cast<size_t>(group_id);
    if (gidx >= node->group_slots.size())
        return false;
    if (!node->group_slots[gidx].has_device_value())
        return false;
    // Note: actual evictability during driveEviction is checked via the
    // is_block_evictable_ callback injected into each ComponentGroup.
    // This public method provides a basic structural check only.
    return true;
}

void BlockTreeCache::setIsBlockEvictable(IsBlockEvictableFn fn) {
    // Forward to all ComponentGroups so driveEviction can check evictability
    for (auto& group : component_groups_) {
        group->setIsBlockEvictable(fn);
    }
}

// TODO: 和match部分要同步fix，当前没有
void BlockTreeCache::releaseMatchedBlocks(const std::vector<BlockIdxType>& block_indices) {
    if (block_indices.empty())
        return;

    // Find FULL group (block_indices are from this group's device_blocks)
    ComponentGroupPtr full_group;
    for (auto& g : component_groups_) {
        if (g->group_type == CacheGroupType::FULL) {
            full_group = g;
            break;
        }
    }
    if (!full_group || full_group->devicePools().empty())
        return;

    const auto& pools = full_group->devicePools();
    size_t num_pools = pools.size();

    // block_indices is laid out as [node0.pool0, node0.pool1, ..., node1.pool0, ...]
    for (size_t i = 0; i < block_indices.size(); ++i) {
        if (isNullBlockIdx(block_indices[i]))
            continue;
        size_t pool_idx = i % num_pools;
        if (pool_idx < pools.size() && pools[pool_idx]) {
            pools[pool_idx]->blockCacheFree(block_indices[i]);
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

void BlockTreeCache::performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx) {
    auto lb_ctx = std::dynamic_pointer_cast<LoadBackAsyncContext>(ctx);
    bool all_ok = true;

    DeviceBufferResolver resolver =
        device_buffer_resolver_ ? device_buffer_resolver_ :
                                  DeviceBufferResolver([](int, BlockIdxType) -> BlockInfo { return BlockInfo{}; });

    for (auto& item : items) {
        auto gid = static_cast<size_t>(item.group_id);
        if (gid >= component_groups_.size() || item.node == nullptr)
            continue;

        // Collect MemoryBlockLayerTagSlot layout (same as performEvictionCopy)
        std::vector<MemoryBlockLayerTagSlot> layer_slots;
        for (int comp_idx : component_groups_[gid]->component_indices) {
            if (comp_idx >= 0 && static_cast<size_t>(comp_idx) < components_.size()) {
                const auto& comp = components_[static_cast<size_t>(comp_idx)];
                for (const auto& lts : comp.memory_block_layer_tag_slots)
                    layer_slots.push_back(lts);
            }
        }
        std::sort(
            layer_slots.begin(),
            layer_slots.end(),
            [](const MemoryBlockLayerTagSlot& a, const MemoryBlockLayerTagSlot& b) { return a.layer_id < b.layer_id; });

        auto& slot    = item.node->group_slots[gid];
        bool  copy_ok = false;

        auto hp = hostPoolForGroup(item.group_id);
        auto dp = diskPoolForGroup(item.group_id);

        if (item.source_tier == Tier::HOST && hp) {
            // Host → GPU (direct H2D)
            copy_ok = copy_engine_->hostToDevice(
                slot.host_block, item.allocated_device_blocks, layer_slots, resolver, *hp);
            if (!copy_ok) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: H2D FAILED "
                                    "group[%d] node_key=%ld",
                                    item.group_id,
                                    item.node->cache_key);
            }
        } else if (item.source_tier == Tier::DISK && hp && dp) {
            // Disk → GPU (cross-layer: Disk → temp Host → GPU, Host not cached)
            BlockIdxType temp_host = NULL_BLOCK_IDX;
            auto         alloc     = hp->malloc();
            if (alloc.has_value())
                temp_host = alloc.value();

            if (!isNullBlockIdx(temp_host)) {
                bool d2h_ok = copy_engine_->diskToHost(slot.disk_slot, temp_host, *hp, *dp);
                if (d2h_ok) {
                    copy_ok = copy_engine_->hostToDevice(
                        temp_host, item.allocated_device_blocks, layer_slots, resolver, *hp);
                }
                hp->free(temp_host);  // Release temp buffer
            }
            if (!copy_ok) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: Disk2D FAILED "
                                    "group[%d] node_key=%ld",
                                    item.group_id,
                                    item.node->cache_key);
            }
        }

        if (!copy_ok) {
            all_ok = false;
            // Rollback: release allocated device blocks via per-pool reference counting
            if (!item.allocated_device_blocks.empty() && gid < component_groups_.size()) {
                component_groups_[gid]->releaseDeviceBlocks(item.allocated_device_blocks);
            }
        }
    }

    // Phase 3: re-acquire lock to update tree state
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : items) {
            auto gid = static_cast<size_t>(item.group_id);
            if (gid >= component_groups_.size() || item.node == nullptr)
                continue;

            auto& group = component_groups_[gid];
            auto& slot  = item.node->group_slots[gid];

            // Check if copy succeeded (device_blocks should be valid)
            if (slot.has_device_value()) {
                // Remove from source heap
                if (item.source_tier == Tier::HOST && group->host_heap) {
                    group->host_heap->invalidate(item.node);
                    slot.in_host_heap = false;
                } else if (item.source_tier == Tier::DISK && group->disk_heap) {
                    group->disk_heap->invalidate(item.node);
                    slot.in_disk_heap = false;
                }
                // Add to device heap if it qualifies as DeviceLeaf
                group->tryAddToDeviceHeap(item.node);
                // Add path lock reference via device pools
                group->referenceDeviceBlocks(item.allocated_device_blocks);
            } else {
                // Copy failed: clear device_blocks from slot
                slot.device_blocks.clear();
            }
        }
    }

    if (lb_ctx) {
        lb_ctx->onTaskComplete(all_ok);
    }
}

void BlockTreeCache::checkWatermark() {
    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        checkTierWatermark(tier);
    }
}

void BlockTreeCache::checkTierWatermark(Tier tier) {
    auto wm = config_.watermarkForTier(tier);
    if (wm.ratio <= 0.0 || !config_.isTierEnabled(tier))
        return;

    for (auto& group : component_groups_) {
        size_t excess = computeGroupExcess(*group, tier, wm.ratio);
        if (excess == 0)
            continue;

        RTP_LLM_LOG_INFO("BlockTreeCache::checkTierWatermark: tier=%s group[%d] "
                         "excess=%zu (ratio=%.2f), evicting",
                         tierName(tier), group->component_group_id, excess, wm.ratio);

        for (size_t i = 0; i < excess; ++i) {
            auto er = group->driveEviction(1, tier);
            if (!er.has_value())
                break;
            submitEviction(*er);
        }
    }
}

size_t BlockTreeCache::computeGroupExcess(const ComponentGroup& group, Tier tier, double ratio) const {
    if (tier == Tier::DEVICE) {
        return group.devicePoolMaxExcess(ratio);
    }
    size_t capacity = (tier == Tier::HOST) ? group.hostPoolCapacity() : group.diskPoolCapacity();
    if (capacity == 0)
        return 0;
    size_t used      = (tier == Tier::HOST) ? group.hostPoolUsed() : group.diskPoolUsed();
    size_t threshold = static_cast<size_t>(capacity * ratio);
    return (used > threshold) ? (used - threshold) : 0;
}

void BlockTreeCache::allocateTargetBlock(EvictionResult& er) {
    if (er.target_tier == Tier::HOST) {
        auto pool = hostPoolForGroup(er.component_group_id);
        if (pool) {
            auto slot       = pool->malloc();
            er.target_block = slot.has_value() ? slot.value() : NULL_BLOCK_IDX;
        }
    } else if (er.target_tier == Tier::DISK) {
        auto pool = diskPoolForGroup(er.component_group_id);
        if (pool) {
            auto slot = pool->malloc();
            if (slot.has_value())
                er.target_block = slot.value();
        }
    }
}

void BlockTreeCache::submitEviction(EvictionResult& er) {
    if (er.target_tier != Tier::NONE && !config_.isTierEnabled(er.target_tier)) {
        er.target_tier = Tier::NONE;
    }

    if (er.target_tier == Tier::NONE) {
        // Don't copy, evict directly
        onEvictionComplete(er, /*cascade_with_copy=*/false);
        return;
    }
    
    allocateTargetBlock(er);
    
    taskStarted();
    auto* work_item = new autil::LambdaWorkItem([this, er]() { performEvictionCopy(er); });
    auto  err       = thread_pool_->pushWorkItem(work_item);
    if (err != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
         // Submit task failed, evict directly
        onEvictionComplete(er, /*cascade_with_copy=*/false);
        taskFinished();
    }
}

}  // namespace rtp_llm
