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
                               int                               eviction_thread_pool_size,
                               std::shared_ptr<StorageBackend>   storage_backend,
                               bool                              enable_device_cache,
                               bool                              enable_memory_cache,
                               bool                              enable_disk_cache,
                               bool                              enable_remote_cache,
                               std::shared_ptr<BroadcastManager> broadcast_manager):
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::move(components)),
    copy_engine_(std::make_shared<CopyEngine>()),
    storage_backend_(std::move(storage_backend)),
    broadcast_manager_(std::move(broadcast_manager)),
    enable_device_cache_(enable_device_cache),
    enable_memory_cache_(enable_memory_cache),
    enable_disk_cache_(enable_disk_cache),
    enable_remote_cache_(enable_remote_cache) {
    // Validate tier dependencies: Disk requires Host (design doc section 2.7)
    if (enable_disk_cache_ && !enable_memory_cache_) {
        throw std::invalid_argument("BlockTreeCache: enable_disk_cache requires enable_memory_cache = true");
    }

    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        static_cast<size_t>(eviction_thread_pool_size), 1000, nullptr, "BlockTreeEvictionPool");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: failed to start eviction thread pool, size=%d", eviction_thread_pool_size);
    }

    RTP_LLM_LOG_INFO("BlockTreeCache: constructed with %zu component groups, %zu components, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     component_groups_.size(),
                     components_.size(),
                     eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     enable_device_cache_ ? "on" : "off",
                     enable_memory_cache_ ? "on" : "off",
                     enable_disk_cache_ ? "on" : "off",
                     enable_remote_cache_ ? "on" : "off");
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
    if (reference_blocks_) {
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
                        reference_blocks_(slot.device_blocks);
                    }
                }
            }
        }
    }

    // Phase 2: load_back — detect and transfer Host/Disk data to GPU
    if (enable_load_back_ && best_match != nullptr) {
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

bool BlockTreeCache::isTierEnabled(Tier tier) const {
    switch (tier) {
        case Tier::DEVICE:
            return enable_device_cache_;
        case Tier::HOST:
            return enable_memory_cache_;
        case Tier::DISK:
            return enable_disk_cache_;
        case Tier::REMOTE:
            return enable_remote_cache_;
        default:
            return false;
    }
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
    if (!isTierEnabled(tier)) {
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

            if (er.target_tier == Tier::NONE || !isTierEnabled(er.target_tier)) {
                // Target tier disabled or direct release: synchronous
                if (!isTierEnabled(er.target_tier) && er.target_tier != Tier::NONE) {
                    RTP_LLM_LOG_DEBUG("BlockTreeCache::evict: target tier %s disabled, "
                                      "downgrading to direct release, group[%d] node_key=%ld",
                                      tierName(er.target_tier),
                                      er.component_group_id,
                                      er.node ? er.node->cache_key : 0);
                }
                onEvictionComplete(er);
            } else {
                // Allocate target block for REUSABLE tier demotion
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
                        if (slot.has_value()) {
                            er.target_block = slot.value();
                        }
                    }
                }

                // Phase 2+3: async data copy + completion callback via autil thread pool
                taskStarted();
                auto* work_item = new autil::LambdaWorkItem([this, er]() { performEvictionCopy(er); });
                auto  err       = thread_pool_->pushWorkItem(work_item);
                if (err != autil::ThreadPool::ERROR_NONE) {
                    RTP_LLM_LOG_WARNING("BlockTreeCache::evict: pushWorkItem failed (err=%d), "
                                        "falling back to synchronous completion",
                                        static_cast<int>(err));
                    work_item->destroy();
                    // Synchronous fallback: evict() already holds mutex_, so we cannot
                    // call performEvictionCopy() (which would try to re-acquire mutex_).
                    // Directly complete the eviction without data copy.
                    onEvictionComplete(er);
                    taskFinished();
                }
            }

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

void BlockTreeCache::performEvictionCopy(EvictionResult er) {
    // Phase 2: perform data movement using CopyEngine + TransferDescriptor (no lock held)
    bool copy_ok = true;

    auto hp = hostPoolForGroup(er.component_group_id);
    auto dp = diskPoolForGroup(er.component_group_id);

    if (copy_engine_ && er.node != nullptr && hp) {
        const auto& desc = er.transfer;

        if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
            // D2H: use source_blocks from TransferDescriptor
            if (!desc.source_blocks.empty() && !isNullBlockIdx(er.target_block)) {
                // Collect layer slots from component descriptors
                auto                                 gid = static_cast<size_t>(er.component_group_id);
                std::vector<MemoryBlockLayerTagSlot> layer_slots;
                if (gid < component_groups_.size()) {
                    for (int comp_idx : component_groups_[gid]->component_indices) {
                        if (comp_idx >= 0 && static_cast<size_t>(comp_idx) < components_.size()) {
                            const auto& comp = components_[static_cast<size_t>(comp_idx)];
                            for (const auto& lts : comp.memory_block_layer_tag_slots) {
                                layer_slots.push_back(lts);
                            }
                        }
                    }
                    std::sort(layer_slots.begin(),
                              layer_slots.end(),
                              [](const MemoryBlockLayerTagSlot& a, const MemoryBlockLayerTagSlot& b) {
                                  return a.layer_id < b.layer_id;
                              });
                }
                DeviceBufferResolver resolver =
                    device_buffer_resolver_ ?
                        device_buffer_resolver_ :
                        DeviceBufferResolver([](int, BlockIdxType) -> BlockInfo { return BlockInfo{}; });
                copy_ok = copy_engine_->deviceToHost(
                    desc.source_blocks[0], er.target_block, layer_slots, resolver, *hp);
                if (!copy_ok) {
                    RTP_LLM_LOG_WARNING("BlockTreeCache::performEvictionCopy: D2H copy FAILED "
                                        "group[%d] node_key=%ld",
                                        er.component_group_id,
                                        er.node->cache_key);
                } else {
                    RTP_LLM_LOG_DEBUG("BlockTreeCache::performEvictionCopy: D2H copy "
                                      "group[%d] node_key=%ld host_block=%d",
                                      er.component_group_id,
                                      er.node->cache_key,
                                      er.target_block);
                }
            }
        } else if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
            if (dp && !desc.source_blocks.empty() && !desc.source_blocks[0].empty()
                && !isNullBlockIdx(desc.source_blocks[0][0]) && !isNullBlockIdx(er.target_block)) {
                copy_ok = copy_engine_->hostToDisk(desc.source_blocks[0][0], er.target_block, *hp, *dp);
                if (!copy_ok) {
                    RTP_LLM_LOG_WARNING("BlockTreeCache::performEvictionCopy: H2Disk copy FAILED "
                                        "group[%d] node_key=%ld",
                                        er.component_group_id,
                                        er.node->cache_key);
                } else {
                    RTP_LLM_LOG_DEBUG("BlockTreeCache::performEvictionCopy: H2Disk copy "
                                      "group[%d] node_key=%ld host=%d disk=%d",
                                      er.component_group_id,
                                      er.node->cache_key,
                                      desc.source_blocks[0][0],
                                      er.target_block);
                }
            }
        } else if (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST) {
            if (dp && !desc.source_blocks.empty() && !desc.source_blocks[0].empty()
                && !isNullBlockIdx(desc.source_blocks[0][0]) && !isNullBlockIdx(er.target_block)) {
                copy_ok = copy_engine_->diskToHost(desc.source_blocks[0][0], er.target_block, *hp, *dp);
                if (!copy_ok) {
                    RTP_LLM_LOG_WARNING("BlockTreeCache::performEvictionCopy: Disk2H copy FAILED "
                                        "group[%d] node_key=%ld",
                                        er.component_group_id,
                                        er.node->cache_key);
                } else {
                    RTP_LLM_LOG_DEBUG("BlockTreeCache::performEvictionCopy: Disk2H copy "
                                      "group[%d] node_key=%ld disk=%d host=%d",
                                      er.component_group_id,
                                      er.node->cache_key,
                                      desc.source_blocks[0][0],
                                      er.target_block);
                }
            }
        }
    }

    // Phase 3: completion callback (re-acquires lock)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (copy_ok) {
            onEvictionComplete(er);
        } else {
            // Rollback: free allocated target block, restore source tier heap
            if (!isNullBlockIdx(er.target_block)) {
                if (er.target_tier == Tier::HOST && hp) {
                    hp->free(er.target_block);
                } else if (er.target_tier == Tier::DISK && dp) {
                    dp->free(er.target_block);
                }
            }
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
    if (copy_ok && enable_remote_cache_ && storage_backend_ && er.node != nullptr) {
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

void BlockTreeCache::onEvictionComplete(const EvictionResult& er) {
    if (er.node == nullptr)
        return;

    auto gid = static_cast<size_t>(er.component_group_id);
    if (gid >= component_groups_.size())
        return;

    auto& group = component_groups_[gid];
    auto& slot  = er.node->group_slots[gid];

    RTP_LLM_LOG_DEBUG("BlockTreeCache::onEvictionComplete: Phase 3 for group[%d] node_key=%ld "
                      "source=%s target=%s",
                      er.component_group_id,
                      er.node->cache_key,
                      tierName(er.source_tier),
                      tierName(er.target_tier));

    group->evictFromTier(er.node, slot, er.source_tier);

    // Release source blocks back to their pools (design doc §2.7 Phase 3)
    if (er.source_tier == Tier::DEVICE && release_blocks_ && !er.blocks_to_release.empty()) {
        release_blocks_(er.blocks_to_release);
    } else if (er.source_tier == Tier::HOST && !er.blocks_to_release.empty()) {
        auto hp = hostPoolForGroup(er.component_group_id);
        if (hp) {
            for (auto block : er.blocks_to_release) {
                if (!isNullBlockIdx(block)) {
                    hp->free(block);
                }
            }
        }
    } else if (er.source_tier == Tier::DISK && !er.blocks_to_release.empty()) {
        auto dp = diskPoolForGroup(er.component_group_id);
        if (dp) {
            for (auto block : er.blocks_to_release) {
                if (!isNullBlockIdx(block)) {
                    dp->free(block);
                }
            }
        }
    }

    // Set target tier data from CopyEngine result
    if (er.target_tier == Tier::HOST && er.source_tier == Tier::DEVICE) {
        if (!isNullBlockIdx(er.target_block)) {
            slot.host_block = er.target_block;
            // Establish cache hold: block becomes tree-visible (refcount 0 -> 1).
            if (auto hp = group->hostPool()) {
                hp->incRef(er.target_block);
            }
            group->tryAddToHostHeap(er.node);
        }
    } else if (er.target_tier == Tier::DISK && er.source_tier == Tier::HOST) {
        if (!isNullBlockIdx(er.target_block)) {
            slot.disk_slot = er.target_block;
            // Establish cache hold: block becomes tree-visible (refcount 0 -> 1).
            if (auto dp = group->diskPool()) {
                dp->incRef(er.target_block);
            }
            group->tryAddToDiskHeap(er.node);
        }
    }

    cascadeEviction(er.node, er.component_group_id, er.source_tier);

    if (shouldDeleteNode(er.node)) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::onEvictionComplete: deleting empty node key=%ld", er.node->cache_key);
        TreeNode* parent = er.node->parent;
        tree_->removeNode(er.node);
        // Ancestor chain cleanup: walk up from parent removing empty nodes.
        tree_->removeEmptyAncestors(parent, reusableGroupIds());
        // After removeEmptyAncestors, verify parent survived.
        // removeNode sets deleted node's parent to nullptr, so if parent
        // was deleted, parent->parent would be nullptr.
        if (parent && parent != tree_->root() && parent->parent != nullptr) {
            for (auto& g : component_groups_) {
                g->tryAddToDeviceHeap(parent);
            }
        }
    } else {
        if (er.node->parent && er.node->parent != tree_->root()) {
            TreeNode* parent = er.node->parent;
            for (auto& g : component_groups_) {
                g->tryAddToDeviceHeap(parent);
            }
        }
    }
}

void BlockTreeCache::cascadeEviction(TreeNode* node, int source_group_id, Tier tier) {
    auto lower_groups = groupsBelowPriority(source_group_id);
    if (lower_groups.empty())
        return;

    RTP_LLM_LOG_DEBUG("BlockTreeCache::cascadeEviction: group[%d] → %zu lower groups "
                      "tier=%s node_key=%ld",
                      source_group_id,
                      lower_groups.size(),
                      tierName(tier),
                      node ? node->cache_key : 0);

    for (int gid : lower_groups) {
        auto gidx = static_cast<size_t>(gid);
        if (gidx >= component_groups_.size() || gidx >= node->group_slots.size())
            continue;

        auto& lower_group = component_groups_[gidx];
        auto& slot        = node->group_slots[gidx];

        // Collect blocks to release before clearing slot markers
        std::vector<BlockIdxType> blocks_to_free;
        switch (tier) {
            case Tier::DEVICE:
                if (slot.has_device_value()) {
                    blocks_to_free = slot.device_blocks;
                }
                break;
            case Tier::HOST:
                if (slot.has_host_value()) {
                    blocks_to_free = {slot.host_block};
                }
                break;
            case Tier::DISK:
                if (slot.has_disk_value()) {
                    blocks_to_free = {slot.disk_slot};
                }
                break;
            default:
                break;
        }

        if (!blocks_to_free.empty()) {
            lower_group->evictFromTier(node, slot, tier);

            // Release blocks back to the lower_group's own pool
            if (tier == Tier::DEVICE && release_blocks_) {
                release_blocks_(blocks_to_free);
            } else if (tier == Tier::HOST && lower_group->hostPool()) {
                for (auto b : blocks_to_free) {
                    if (!isNullBlockIdx(b))
                        lower_group->hostPool()->free(b);
                }
            } else if (tier == Tier::DISK && lower_group->diskPool()) {
                for (auto b : blocks_to_free) {
                    if (!isNullBlockIdx(b))
                        lower_group->diskPool()->free(b);
                }
            }

            RTP_LLM_LOG_DEBUG("BlockTreeCache::cascadeEviction: released %zu blocks from "
                              "lower group[%d] tier=%s node_key=%ld (cascade direct release)",
                              blocks_to_free.size(),
                              gid,
                              tierName(tier),
                              node->cache_key);
        }
    }
}

bool BlockTreeCache::shouldDeleteNode(const TreeNode* node) const {
    if (node == nullptr || node == tree_->root())
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

void BlockTreeCache::releaseMatchedBlocks(const std::vector<BlockIdxType>& block_indices) {
    if (release_blocks_ && !block_indices.empty()) {
        release_blocks_(block_indices);
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
            // Rollback: release allocated device blocks
            if (release_blocks_ && !item.allocated_device_blocks.empty()) {
                release_blocks_(item.allocated_device_blocks);
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
                // Add path lock reference
                if (reference_blocks_) {
                    reference_blocks_(item.allocated_device_blocks);
                }
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
    checkTierWatermark(Tier::DEVICE);
    checkTierWatermark(Tier::HOST);
    checkTierWatermark(Tier::DISK);
}

void BlockTreeCache::checkTierWatermark(Tier tier) {
    // Select watermark config for this tier
    TierWatermark wm;
    switch (tier) {
        case Tier::DEVICE:
            wm = watermark_device_;
            break;
        case Tier::HOST:
            wm = watermark_host_;
            break;
        case Tier::DISK:
            wm = watermark_disk_;
            break;
        default:
            return;
    }
    if (wm.ratio <= 0.0 || !isTierEnabled(tier))
        return;

    if (tier == Tier::DEVICE) {
        // DEVICE has no per-group pool; use external capacity from watermark config
        if (wm.capacity == 0)
            return;
        size_t used = 0;
        // TODO: add device block pool....
        for (const auto& g : component_groups_) {
            auto* heap = g->heapForTier(tier);
            if (heap)
                used += heap->size();
        }
        size_t threshold = static_cast<size_t>(wm.capacity * wm.ratio);
        if (used <= threshold)
            return;

        size_t excess = used - threshold;
        RTP_LLM_LOG_INFO("BlockTreeCache::checkTierWatermark: tier=DEVICE used=%zu > threshold=%zu "
                         "(ratio=%.2f, capacity=%zu), evicting %zu blocks",
                         used, threshold, wm.ratio, wm.capacity, excess);

        for (size_t attempt = 0; attempt < excess; ++attempt) {
            bool found = false;
            for (auto& group : component_groups_) {
                auto er = group->driveEviction(1, tier);
                if (!er.has_value())
                    continue;

                found = true;
                if (er->target_tier != Tier::NONE && !isTierEnabled(er->target_tier)) {
                    er->target_tier = Tier::NONE;
                }

                if (er->target_tier == Tier::NONE) {
                    onEvictionComplete(*er);
                } else {
                    // Allocate target block from per-group pool
                    if (er->target_tier == Tier::HOST) {
                        auto pool = hostPoolForGroup(er->component_group_id);
                        if (pool) {
                            auto slot        = pool->malloc();
                            er->target_block = slot.has_value() ? slot.value() : NULL_BLOCK_IDX;
                        }
                    } else if (er->target_tier == Tier::DISK) {
                        auto pool = diskPoolForGroup(er->component_group_id);
                        if (pool) {
                            auto slot = pool->malloc();
                            if (slot.has_value())
                                er->target_block = slot.value();
                        }
                    }
                    taskStarted();
                    auto  captured_er = *er;
                    auto* work_item =
                        new autil::LambdaWorkItem([this, captured_er]() { performEvictionCopy(captured_er); });
                    auto err = thread_pool_->pushWorkItem(work_item);
                    if (err != autil::ThreadPool::ERROR_NONE) {
                        work_item->destroy();
                        onEvictionComplete(*er);
                        taskFinished();
                    }
                }
                break;
            }
            if (!found)
                break;
        }
        return;
    }

    // HOST / DISK: per-group pool-based watermark
    for (auto& group : component_groups_) {
        size_t capacity = 0, used = 0;
        if (tier == Tier::HOST) {
            capacity = group->hostPoolCapacity();
            used     = group->hostPoolUsed();
        } else {
            capacity = group->diskPoolCapacity();
            used     = group->diskPoolUsed();
        }
        if (capacity == 0)
            continue;

        size_t threshold = static_cast<size_t>(capacity * wm.ratio);
        if (used <= threshold)
            continue;

        size_t excess = used - threshold;
        RTP_LLM_LOG_INFO("BlockTreeCache::checkTierWatermark: tier=%s group[%d] used=%zu > threshold=%zu "
                         "(ratio=%.2f, capacity=%zu), evicting %zu blocks",
                         tierName(tier),
                         group->component_group_id,
                         used,
                         threshold,
                         wm.ratio,
                         capacity,
                         excess);

        for (size_t attempt = 0; attempt < excess; ++attempt) {
            auto er = group->driveEviction(1, tier);
            if (!er.has_value())
                break;

            if (er->target_tier != Tier::NONE && !isTierEnabled(er->target_tier)) {
                er->target_tier = Tier::NONE;
            }

            if (er->target_tier == Tier::NONE) {
                onEvictionComplete(*er);
            } else {
                // Allocate target block from per-group pool
                if (er->target_tier == Tier::HOST) {
                    auto pool = group->hostPool();
                    if (pool) {
                        auto slot        = pool->malloc();
                        er->target_block = slot.has_value() ? slot.value() : NULL_BLOCK_IDX;
                    }
                } else if (er->target_tier == Tier::DISK) {
                    auto pool = group->diskPool();
                    if (pool) {
                        auto slot = pool->malloc();
                        if (slot.has_value())
                            er->target_block = slot.value();
                    }
                }
                taskStarted();
                auto  captured_er = *er;
                auto* work_item =
                    new autil::LambdaWorkItem([this, captured_er]() { performEvictionCopy(captured_er); });
                auto err = thread_pool_->pushWorkItem(work_item);
                if (err != autil::ThreadPool::ERROR_NONE) {
                    work_item->destroy();
                    onEvictionComplete(*er);
                    taskFinished();
                }
            }
        }
    }
}

}  // namespace rtp_llm
