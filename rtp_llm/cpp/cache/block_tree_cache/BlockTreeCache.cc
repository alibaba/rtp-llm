#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>               tree,
                               BlockTreeCacheConfig                     config,
                               std::shared_ptr<StorageBackend>          storage_backend,
                               std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>       task_pool):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    evictor_(
        tree_.get(),
        [this](const TransferDescriptor& descriptor) { return executeTransfer(descriptor); },
        config_.enable_reverse_eviction,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        metrics_reporter_,
        mutex_,
        config_.memory_cache_sync_timeout_ms,
        config_.memory_cache_disk_sync_timeout_ms,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](const std::vector<EvictionReleaseCredit>& credits) {
            reserveInFlightDeviceReleaseCreditsLocked(credits);
        },
        [this](const std::vector<EvictionReleaseCredit>& credits) {
            settleInFlightDeviceReleaseCreditsLocked(credits);
        },
        [this](bool tree_data_mutated, bool check_watermark) {
            if (tree_data_mutated) {
                ++mutation_version_;
            }
            if (check_watermark) {
                checkWatermark();
            }
        },
        [this](CacheKeyType cache_key, size_t group_set_id) {
            if (config_.enable_remote_cache) {
                evictor_.writeRemoteThrough(storage_backend_, cache_key, group_set_id);
            }
        }),
    loader_(
        tree_.get(),
        evictor_,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        metrics_reporter_,
        mutex_,
        config_.memory_cache_disk_sync_timeout_ms,
        config_.memory_cache_sync_timeout_ms,
        config_.enable_device_cache,
        [this](bool tree_data_mutated, bool check_watermark) {
            if (tree_data_mutated) {
                ++mutation_version_;
            }
            if (check_watermark) {
                checkWatermark();
            }
        }) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    evictor_.init(config_.device_eviction_policy, config_.host_eviction_policy, config_.disk_eviction_policy);
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.eviction_thread_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     tree_->groupSets().size(),
                     tree_->reusableGroupCount(),
                     config_.eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_memory_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    initialized_ = true;
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    loader_.shutdown();
    if (!initialized_) {
        RTP_LLM_LOG_INFO("destroyed");
        return;
    }
    RTP_LLM_LOG_INFO("load tickets closed, waiting for pending tasks...");
    waitForPendingTasks();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RTP_LLM_CHECK_WITH_INFO(
            in_flight_device_release_credits_.empty(),
            "BlockTreeCache: in-flight DEVICE release credits remain after pending tasks drained: %zu",
            in_flight_device_release_credits_.size());
    }
    task_pool_->shutdown();
    RTP_LLM_LOG_INFO("destroyed");
}

bool BlockTreeCache::executeTransfer(const TransferDescriptor& descriptor) {
    return transfer_dispatcher_->executePerRank(descriptor);
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType& cache_keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    return loader_.matchLocked(cache_keys);
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);

    BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources);

    if (insert_result.inserted_nodes.empty() && insert_result.adopted_nodes.empty()) {
        return;
    }

    // Stamp and refresh only newly created nodes and exact adopted GroupSet resources.
    evictor_.onInsertCommitted(insert_result);
    ++mutation_version_;
    checkWatermark();
}

int BlockTreeCache::evictForGroup(size_t group_id, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_id);
    if (location == nullptr) {
        return 0;
    }
    const GroupSetPtr& group_set   = tree_->groupSets()[location->group_set_id];
    const auto&        device_pool = group_set->devicePools()[location->member_group_id];

    const size_t initial_free = device_pool->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        auto eviction_desc = evictor_.chooseVictim(location->group_set_id, Tier::DEVICE);
        if (!eviction_desc.has_value()) {
            break;
        }
        eviction_desc->target_tier = Tier::NONE;
        if (!evictor_.submitLocked(*eviction_desc)) {
            break;
        }
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    RTP_LLM_LOG_DEBUG("group_id=%zu group_set[%zu] reclaimed %zu/%zu device blocks",
                      group_id,
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

void BlockTreeCache::releaseMatchedResources(const std::vector<MultiNodeResource>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);
    loader_.releaseMatchedResourcesLocked(resources);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(size_t                                group_id,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_id, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->nodes().size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<BlockTreePoolMetricsSnapshot> BlockTreeCache::poolMetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_reporter_.collectPoolMetricsSnapshots(tree_->groupSets(), evictor_);
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_.collectEvictableMetricsSnapshots(tree_->groupSets(), evictor_);
    }
    metrics_reporter_.reportEvictableBlockCount(snapshots);
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;
    if (limit == 0) {
        return snapshot;
    }

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->nodes().size());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty() && snapshot.keys.size() < limit) {
        const TreeNode* node = pending.back();
        pending.pop_back();
        const bool reusable = std::any_of(node->group_set_resources.begin(),
                                          node->group_set_resources.end(),
                                          [](const GroupSetResource& resource) { return !resource.is_empty(); });
        if (reusable) {
            snapshot.keys.push_back(node->cache_key);
        }
        for (const auto& [cache_key, child] : node->children) {
            (void)cache_key;
            if (child) {
                pending.push_back(child);
            }
        }
    }
    return snapshot;
}

void BlockTreeCache::waitForPendingTasks() {
    task_pool_->waitForIdle();
}

void BlockTreeCache::onBlocksReleased(const std::vector<BlockReleaseReceipt>& receipts) {
    std::lock_guard<std::mutex> lock(mutex_);
    struct DirtyResource {
        TreeNode* node;
        size_t    group_set_id;

        bool operator==(const DirtyResource& other) const {
            return node == other.node && group_set_id == other.group_set_id;
        }
    };
    struct DirtyResourceHash {
        size_t operator()(const DirtyResource& resource) const {
            const size_t node_hash      = std::hash<TreeNode*>{}(resource.node);
            const size_t group_set_hash = std::hash<size_t>{}(resource.group_set_id);
            return node_hash ^ (group_set_hash << 1);
        }
    };

    std::unordered_set<DirtyResource, DirtyResourceHash> dirty_resources;
    dirty_resources.reserve(receipts.size());
    for (const BlockReleaseReceipt& receipt : receipts) {
        if (receipt.new_total_ref_count > 1) {
            continue;
        }
        const ReusableGroupLocation* location = tree_->reusableGroupLocation(receipt.group_id);
        if (location == nullptr) {
            continue;
        }

        const GroupSetPtr& group_set = tree_->groupSets()[location->group_set_id];

        TreeNode* node = group_set->findTreeNodeByDeviceBlock(location->member_group_id, receipt.block_id);
        if (node == nullptr) {
            continue;
        }
        dirty_resources.emplace(DirtyResource{node, location->group_set_id});
    }

    for (const DirtyResource& dirty_resource : dirty_resources) {
        evictor_.refreshCandidate(dirty_resource.node, dirty_resource.group_set_id);
    }
    checkWatermark();
}

bool BlockTreeCache::cancelLoad(const std::shared_ptr<AsyncContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    return loader_.cancelLoadLocked(context);
}

void BlockTreeCache::reserveInFlightDeviceReleaseCreditsLocked(
    const std::vector<EvictionReleaseCredit>& release_credits) {
    for (const EvictionReleaseCredit& credit : release_credits) {
        if (credit.pool != nullptr) {
            ++in_flight_device_release_credits_[credit.pool];
        }
    }
}

void BlockTreeCache::settleInFlightDeviceReleaseCreditsLocked(
    const std::vector<EvictionReleaseCredit>& release_credits) noexcept {
    for (const EvictionReleaseCredit& credit : release_credits) {
        const auto it = in_flight_device_release_credits_.find(credit.pool);
        RTP_LLM_CHECK_WITH_INFO(it != in_flight_device_release_credits_.end() && it->second > 0,
                                "missing in-flight DEVICE release credit while settling pool=%p block=%d",
                                static_cast<void*>(credit.pool.get()),
                                credit.block);
        if (--it->second == 0) {
            in_flight_device_release_credits_.erase(it);
        }
    }
}
void BlockTreeCache::checkWatermark() {
    if (config_.enable_device_cache && config_.device_min_free_blocks > 0) {
        struct PoolDeficit {
            DeviceBlockPoolPtr pool;
            size_t             deficit{0};
            size_t             accepted_credits{0};
        };
        std::vector<PoolDeficit>                     pool_deficits;
        std::unordered_map<DeviceBlockPool*, size_t> pool_indices;
        for (const auto& group_set : tree_->groupSets()) {
            for (const auto& pool : group_set->devicePools()) {
                if (pool_indices.count(pool.get()) != 0) {
                    continue;
                }
                const size_t capacity     = pool->totalBlocksNum();
                const size_t min_free     = std::min(config_.device_min_free_blocks, capacity);
                const size_t free_blocks  = pool->freeBlocksNum();
                const size_t deficit      = free_blocks < min_free ? min_free - free_blocks : 0;
                const auto   in_flight_it = in_flight_device_release_credits_.find(pool);
                const size_t in_flight_credits =
                    in_flight_it == in_flight_device_release_credits_.end() ? 0 : in_flight_it->second;
                pool_indices.emplace(pool.get(), pool_deficits.size());
                pool_deficits.push_back({pool, deficit, in_flight_credits});
            }
        }

        auto has_uncovered_deficit = [&]() {
            return std::any_of(pool_deficits.begin(), pool_deficits.end(), [](const PoolDeficit& state) {
                return state.accepted_credits < state.deficit;
            });
        };
        auto group_set_has_uncovered_deficit = [&](const GroupSetPtr& group_set) {
            for (const auto& pool : group_set->devicePools()) {
                const auto it = pool_indices.find(pool.get());
                if (it != pool_indices.end()) {
                    const auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        return true;
                    }
                }
            }
            return false;
        };

        std::vector<bool> unavailable(tree_->groupSets().size(), false);
        while (has_uncovered_deficit()) {
            bool round_progress = false;
            for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
                const auto& group_set = tree_->groupSets()[group_set_id];
                if (unavailable[group_set_id] || !group_set_has_uncovered_deficit(group_set)) {
                    continue;
                }
                auto eviction_desc = evictor_.chooseVictim(group_set_id, Tier::DEVICE);
                if (!eviction_desc.has_value()) {
                    unavailable[group_set_id] = true;
                    continue;
                }
                std::vector<EvictionReleaseCredit> release_credits;
                if (!evictor_.submitLocked(*eviction_desc, &release_credits)) {
                    unavailable[group_set_id] = true;
                    continue;
                }
                bool credited_uncovered_pool = false;
                for (const EvictionReleaseCredit& credit : release_credits) {
                    const auto it = pool_indices.find(credit.pool.get());
                    if (it == pool_indices.end()) {
                        continue;
                    }
                    auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        ++state.accepted_credits;
                        credited_uncovered_pool = true;
                    }
                }
                if (credited_uncovered_pool) {
                    round_progress = true;
                } else {
                    unavailable[group_set_id] = true;
                }
            }
            if (!round_progress) {
                break;
            }
        }
    }

    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (tier == Tier::DEVICE && config_.device_min_free_blocks > 0) {
            continue;
        }
        auto wm = config_.watermarkForTier(tier);
        if (wm.ratio <= 0.0 || !config_.isTierEnabled(tier))
            continue;

        for (auto& group_set : tree_->groupSets()) {
            auto victims = evictor_.chooseWatermarkVictims(*group_set, tier, wm.ratio);
            for (auto& eviction_desc : victims) {
                evictor_.submitLocked(eviction_desc);
            }
        }
    }
}

}  // namespace rtp_llm
