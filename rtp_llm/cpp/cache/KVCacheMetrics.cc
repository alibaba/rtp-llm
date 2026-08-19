#include "rtp_llm/cpp/cache/KVCacheMetrics.h"

#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<CachePoolMetricsSnapshot>
mergeCachePoolMetricsSnapshots(const std::vector<KVCachePoolMetricsSnapshot>&   allocator_snapshots,
                               const std::vector<BlockTreePoolMetricsSnapshot>& tree_snapshots) {
    std::unordered_map<std::string, BlockTreePoolMetricsSnapshot> device_tree_snapshots;
    for (const BlockTreePoolMetricsSnapshot& snapshot : tree_snapshots) {
        if (snapshot.tier == Tier::DEVICE) {
            const bool inserted = device_tree_snapshots.emplace(snapshot.pool_name, snapshot).second;
            if (!inserted) {
                RTP_LLM_LOG_WARNING("duplicate device pool name in metrics: %s", snapshot.pool_name.c_str());
            }
        }
    }

    std::vector<CachePoolMetricsSnapshot> report_snapshots;
    report_snapshots.reserve(allocator_snapshots.size() + tree_snapshots.size());

    std::unordered_set<std::string> reported_device_pools;
    for (const KVCachePoolMetricsSnapshot& snapshot : allocator_snapshots) {
        const bool inserted = reported_device_pools.insert(snapshot.pool_name).second;
        if (!inserted) {
            RTP_LLM_LOG_WARNING("duplicate allocator pool name in metrics: %s", snapshot.pool_name.c_str());
            continue;
        }
        CachePoolMetricsSnapshot report_snapshot;
        report_snapshot.tier                      = tierName(Tier::DEVICE);
        report_snapshot.pool_name                 = snapshot.pool_name;
        report_snapshot.block_size_bytes          = snapshot.block_size_bytes;
        report_snapshot.total_blocks              = snapshot.total_blocks;
        report_snapshot.free_blocks               = snapshot.free_blocks;
        report_snapshot.used_blocks               = snapshot.used_blocks;
        report_snapshot.available_blocks          = snapshot.free_blocks;
        report_snapshot.active_tree_cached_blocks = snapshot.active_tree_cached_blocks;
        report_snapshot.reserve_blocks            = snapshot.reserve_blocks;
        report_snapshot.request_ref_blocks        = snapshot.request_ref_blocks;
        report_snapshot.connector_ref_blocks      = snapshot.connector_ref_blocks;
        report_snapshot.block_cache_ref_blocks    = snapshot.block_cache_ref_blocks;
        report_snapshot.eviction_ref_blocks       = snapshot.eviction_ref_blocks;
        report_snapshot.store_ref_blocks          = snapshot.store_ref_blocks;
        report_snapshot.used_ratio                = snapshot.used_ratio;
        const std::unordered_map<std::string, BlockTreePoolMetricsSnapshot>::const_iterator tree_it =
            device_tree_snapshots.find(snapshot.pool_name);
        if (tree_it != device_tree_snapshots.end()) {
            report_snapshot.available_blocks = tree_it->second.available_blocks;
        }
        report_snapshots.push_back(std::move(report_snapshot));
    }

    for (const BlockTreePoolMetricsSnapshot& snapshot : tree_snapshots) {
        if (snapshot.tier == Tier::DEVICE && reported_device_pools.count(snapshot.pool_name) > 0) {
            continue;
        }
        CachePoolMetricsSnapshot report_snapshot;
        report_snapshot.tier                      = tierName(snapshot.tier);
        report_snapshot.pool_name                 = snapshot.pool_name;
        report_snapshot.block_size_bytes          = snapshot.block_size_bytes;
        report_snapshot.total_blocks              = snapshot.total_blocks;
        report_snapshot.free_blocks               = snapshot.free_blocks;
        report_snapshot.used_blocks               = snapshot.used_blocks;
        report_snapshot.available_blocks          = snapshot.available_blocks;
        report_snapshot.request_ref_blocks        = snapshot.request_ref_blocks;
        report_snapshot.connector_ref_blocks      = snapshot.connector_ref_blocks;
        report_snapshot.block_cache_ref_blocks    = snapshot.block_cache_ref_blocks;
        report_snapshot.eviction_ref_blocks       = snapshot.eviction_ref_blocks;
        report_snapshot.store_ref_blocks          = snapshot.store_ref_blocks;
        report_snapshot.active_tree_cached_blocks = snapshot.active_tree_cached_blocks;
        report_snapshot.used_ratio                = snapshot.total_blocks == 0 ?
                                                        0.0f :
                                                        static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.free_blocks)
                                                            / static_cast<double>(snapshot.total_blocks));
        report_snapshots.push_back(std::move(report_snapshot));
    }

    return report_snapshots;
}

}  // namespace rtp_llm
