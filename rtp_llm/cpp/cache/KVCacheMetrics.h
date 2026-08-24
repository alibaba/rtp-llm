#pragma once

#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"

namespace rtp_llm {

struct CachePoolMetricsSnapshot {
    size_t      total_blocks              = 0;
    size_t      free_blocks               = 0;
    size_t      used_blocks               = 0;
    size_t      active_blocks             = 0;
    std::string tier;
    std::string pool_name;
    size_t      block_size_bytes       = 0;
    size_t      available_blocks       = 0;
    size_t      reserve_blocks         = 0;
    size_t      request_ref_blocks     = 0;
    size_t      block_cache_ref_blocks = 0;
    size_t      load_ref_blocks        = 0;
    size_t      eviction_ref_blocks    = 0;
    size_t      store_ref_blocks       = 0;
    float       used_ratio             = 0.0f;
};

// Allocator snapshots own device pool metrics. BlockTree pools missing from the allocator are appended,
// which covers Host, Disk, and any tree-only Device pool.
std::vector<CachePoolMetricsSnapshot>
mergeCachePoolMetricsSnapshots(const std::vector<KVCachePoolMetricsSnapshot>&   allocator_snapshots,
                               const std::vector<BlockTreePoolMetricsSnapshot>& tree_snapshots);

}  // namespace rtp_llm
