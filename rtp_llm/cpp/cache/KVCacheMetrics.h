#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"

namespace rtp_llm {

struct CachePoolMetricsSnapshot {
    size_t      total_blocks  = 0;
    size_t      free_blocks   = 0;
    size_t      used_blocks   = 0;
    size_t      active_blocks = 0;
    std::string tier;
    std::string pool_name;
    size_t      block_size_bytes           = 0;
    size_t      available_blocks           = 0;
    size_t      reserve_blocks             = 0;
    size_t      request_ref_blocks         = 0;
    size_t      block_cache_ref_blocks     = 0;
    size_t      load_ref_blocks            = 0;
    size_t      eviction_target_ref_blocks = 0;
    size_t      store_ref_blocks           = 0;
    size_t      frozen_ref_blocks          = 0;
    size_t      unfrozen_blocks            = 0;
    float       used_ratio                 = 0.0f;

    std::optional<int64_t> read_bytes;
    std::optional<int64_t> write_bytes;
    std::optional<double>  read_bandwidth_bytes_per_sec;
    std::optional<double>  write_bandwidth_bytes_per_sec;
};

// Allocator snapshots own device pool metrics. BlockTree pools missing from the allocator are appended,
// which covers Host, Disk, and any tree-only Device pool.
std::vector<CachePoolMetricsSnapshot>
mergeCachePoolMetricsSnapshots(const std::vector<KVCachePoolMetricsSnapshot>&   allocator_snapshots,
                               const std::vector<BlockTreePoolMetricsSnapshot>& tree_snapshots);

}  // namespace rtp_llm
