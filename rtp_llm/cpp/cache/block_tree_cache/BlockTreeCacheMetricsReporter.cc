#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

using DeviceCandidateBlocks = std::unordered_map<const IBlockPool*, std::unordered_set<BlockIdxType>>;

constexpr std::array<Tier, 3>           kMetricTiers      = {Tier::DEVICE, Tier::HOST, Tier::DISK};
constexpr std::array<CacheGroupType, 3> kMetricGroupTypes = {
    CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::LINEAR};

int metricGroupTypeIndex(CacheGroupType group_type) {
    if (group_type == CacheGroupType::FULL) {
        return 0;
    }
    if (group_type == CacheGroupType::SWA) {
        return 1;
    }
    if (group_type == CacheGroupType::LINEAR) {
        return 2;
    }
    return -1;
}

BlockTreePoolMetricsSnapshot makePoolMetricsSnapshot(Tier tier, const IBlockPool& pool, size_t candidate_blocks) {
    BlockTreePoolMetricsSnapshot snapshot;
    snapshot.tier                      = tier;
    snapshot.pool_name                 = pool.poolName();
    snapshot.block_size_bytes          = pool.blockSizeBytes();
    snapshot.total_blocks              = pool.totalBlocksNum();
    snapshot.free_blocks               = pool.freeBlocksNum();
    snapshot.available_blocks          = std::min(snapshot.total_blocks, snapshot.free_blocks + candidate_blocks);
    snapshot.active_tree_cached_blocks = pool.activeTreeCachedBlocksNum();
    snapshot.request_ref_count         = pool.totalRefCount(BlockRefType::REQUEST);
    snapshot.connector_ref_count       = pool.totalRefCount(BlockRefType::CONNECTOR);
    snapshot.block_cache_ref_count     = pool.totalRefCount(BlockRefType::BLOCK_CACHE);
    return snapshot;
}

DeviceCandidateBlocks collectDeviceCandidateBlocks(const std::vector<ComponentGroupPtr>& component_groups,
                                                   const BlockTreeEvictor&               evictor) {
    DeviceCandidateBlocks candidate_blocks;
    for (const ComponentGroupPtr& group : component_groups) {
        const std::vector<TreeNode*> candidate_nodes = evictor.candidateNodes(group->component_group_id, Tier::DEVICE);
        const std::vector<DeviceBlockPoolPtr>& device_pools = group->devicePools();
        for (TreeNode* node : candidate_nodes) {
            if (node == nullptr || static_cast<size_t>(group->component_group_id) >= node->group_slots.size()) {
                continue;
            }
            const std::vector<BlockIdxType> blocks =
                group->getBlocks(node->group_slots[static_cast<size_t>(group->component_group_id)], Tier::DEVICE);
            const size_t block_count = std::min(blocks.size(), device_pools.size());
            for (size_t pool_index = 0; pool_index < block_count; ++pool_index) {
                if (device_pools[pool_index] == nullptr || isNullBlockIdx(blocks[pool_index])) {
                    continue;
                }
                const std::pair<std::unordered_set<BlockIdxType>::iterator, bool> insert_result =
                    candidate_blocks[device_pools[pool_index].get()].insert(blocks[pool_index]);
                if (!insert_result.second) {
                    continue;
                }
            }
        }
    }
    return candidate_blocks;
}

size_t deviceCandidateBlockCount(const DeviceCandidateBlocks& candidate_blocks, const IBlockPool* pool) {
    const DeviceCandidateBlocks::const_iterator candidates_it = candidate_blocks.find(pool);
    return candidates_it == candidate_blocks.end() ? 0 : candidates_it->second.size();
}

}  // namespace

void BlockTreeCacheMetricsReporter::setMetricsReporter(
    const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}

std::vector<BlockTreePoolMetricsSnapshot>
BlockTreeCacheMetricsReporter::collectPoolMetricsSnapshots(const std::vector<ComponentGroupPtr>& component_groups,
                                                           const BlockTreeEvictor&               evictor) const {
    const DeviceCandidateBlocks device_candidate_blocks = collectDeviceCandidateBlocks(component_groups, evictor);
    std::unordered_set<const IBlockPool*>     reported_device_pools;
    std::vector<BlockTreePoolMetricsSnapshot> snapshots;
    for (const ComponentGroupPtr& group : component_groups) {
        const std::vector<DeviceBlockPoolPtr>& device_pools = group->devicePools();
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            if (pool == nullptr) {
                continue;
            }
            const std::pair<std::unordered_set<const IBlockPool*>::iterator, bool> insert_result =
                reported_device_pools.insert(pool.get());
            if (!insert_result.second) {
                continue;
            }
            snapshots.push_back(makePoolMetricsSnapshot(
                Tier::DEVICE, *pool, deviceCandidateBlockCount(device_candidate_blocks, pool.get())));
        }

        const std::shared_ptr<HostBlockPool> host_pool = group->hostPool();
        if (host_pool != nullptr) {
            snapshots.push_back(makePoolMetricsSnapshot(
                Tier::HOST, *host_pool, evictor.candidateCount(group->component_group_id, Tier::HOST)));
        }

        const std::shared_ptr<DiskBlockPool> disk_pool = group->diskPool();
        if (disk_pool != nullptr) {
            snapshots.push_back(makePoolMetricsSnapshot(
                Tier::DISK, *disk_pool, evictor.candidateCount(group->component_group_id, Tier::DISK)));
        }
    }
    return snapshots;
}

std::vector<BlockTreeEvictableMetricsSnapshot>
BlockTreeCacheMetricsReporter::collectEvictableMetricsSnapshots(const std::vector<ComponentGroupPtr>& component_groups,
                                                                const BlockTreeEvictor&               evictor) const {
    std::array<std::array<size_t, kMetricGroupTypes.size()>, kMetricTiers.size()> evictable_counts{};
    for (const ComponentGroupPtr& group : component_groups) {
        if (group == nullptr) {
            continue;
        }
        const int group_type_index = metricGroupTypeIndex(group->group_type);
        if (group_type_index < 0) {
            continue;
        }
        for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
            evictable_counts[tier_index][static_cast<size_t>(group_type_index)] +=
                evictor.candidateCount(group->component_group_id, kMetricTiers[tier_index]);
        }
    }

    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    snapshots.reserve(kMetricTiers.size() * kMetricGroupTypes.size());
    for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
        for (size_t group_type_index = 0; group_type_index < kMetricGroupTypes.size(); ++group_type_index) {
            BlockTreeEvictableMetricsSnapshot snapshot;
            snapshot.tier             = kMetricTiers[tier_index];
            snapshot.group_type       = kMetricGroupTypes[group_type_index];
            snapshot.evictable_blocks = evictable_counts[tier_index][group_type_index];
            snapshots.push_back(snapshot);
        }
    }
    return snapshots;
}

void BlockTreeCacheMetricsReporter::reportEvictableBlockCount(
    const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    for (const BlockTreeEvictableMetricsSnapshot& snapshot : snapshots) {
        RtpLLMCacheEvictionMetricsCollector collector;
        collector.source_tier           = metricTierName(snapshot.tier);
        collector.group_type            = metricCacheGroupTypeName(snapshot.group_type);
        collector.evictable_block_count = static_cast<int64_t>(snapshot.evictable_blocks);
        collector.report_evictable      = true;
        metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionFinished(
    const BlockTreeEvictor::EvictionPlan&  plan,
    const BlockTreeEvictor::CopyResultSet& results,
    const std::vector<ComponentGroupPtr>&  component_groups) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }

    const int64_t finish_time_us = currentTimeUs();
    if (results.primary_success) {
        reportEvictionMove(plan.primary, component_groups, finish_time_us);
    }
    for (size_t move_index = 0; move_index < plan.cascade_moves.size(); ++move_index) {
        if (move_index < results.cascade_success.size() && results.cascade_success[move_index]) {
            reportEvictionMove(plan.cascade_moves[move_index], component_groups, finish_time_us);
        }
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionMove(const EvictionMove&                   eviction_move,
                                                       const std::vector<ComponentGroupPtr>& component_groups,
                                                       int64_t                               finish_time_us) const {
    if (eviction_move.component_group_id < 0) {
        return;
    }
    const size_t group_id = static_cast<size_t>(eviction_move.component_group_id);
    if (group_id >= component_groups.size()) {
        return;
    }
    const ComponentGroupPtr& group = component_groups[group_id];
    if (group == nullptr || group->component_group_id != eviction_move.component_group_id) {
        return;
    }

    RtpLLMCacheEvictionMetricsCollector collector;
    collector.source_tier     = metricTierName(eviction_move.source_tier);
    collector.target_tier     = metricTierName(eviction_move.target_tier);
    collector.group_type      = metricCacheGroupTypeName(group->group_type);
    collector.report_eviction = true;
    if (eviction_move.source_tier_enter_time_us > 0 && finish_time_us >= eviction_move.source_tier_enter_time_us) {
        collector.lifetime_ms     = (finish_time_us - eviction_move.source_tier_enter_time_us) / 1000;
        collector.report_lifetime = true;
    }
    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
}

int BlockTreeCacheMetricsReporter::transferDirectionIndex(Tier source_tier, Tier target_tier) {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return 0;
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return 1;
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return 2;
    }
    if (source_tier == Tier::DISK && target_tier == Tier::DEVICE) {
        return 3;
    }
    return -1;
}

int64_t BlockTreeCacheMetricsReporter::reportTransferStarted(Tier source_tier, Tier target_tier) {
    if (metrics_reporter_ == nullptr) {
        return 0;
    }
    const int direction_index = transferDirectionIndex(source_tier, target_tier);
    if (direction_index < 0) {
        return 0;
    }
    const int64_t begin_time_us = currentTimeUs();
    const int64_t in_flight     = transfer_in_flight_[static_cast<size_t>(direction_index)].fetch_add(1) + 1;
    RtpLLMCacheTransferMetricsCollector collector;
    collector.source_tier        = metricTierName(source_tier);
    collector.target_tier        = metricTierName(target_tier);
    collector.in_flight          = in_flight;
    collector.transfer_completed = false;
    metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
    return begin_time_us;
}

void BlockTreeCacheMetricsReporter::reportTransferFinished(
    Tier source_tier, Tier target_tier, size_t block_count, int64_t begin_time_us, bool success) {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    const int direction_index = transferDirectionIndex(source_tier, target_tier);
    if (direction_index < 0) {
        return;
    }
    const int64_t in_flight = transfer_in_flight_[static_cast<size_t>(direction_index)].fetch_sub(1) - 1;
    RtpLLMCacheTransferMetricsCollector collector;
    collector.source_tier = metricTierName(source_tier);
    collector.target_tier = metricTierName(target_tier);
    collector.block_count = static_cast<int64_t>(block_count);
    collector.latency_us  = currentTimeUs() - begin_time_us;
    collector.in_flight   = in_flight;
    collector.success     = success;
    metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
}

}  // namespace rtp_llm
