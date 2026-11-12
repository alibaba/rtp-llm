#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTier.h"

namespace rtp_llm {

struct CachePoolMetrics {
    Tier        tier{Tier::DEVICE};
    size_t      pool_index{0};
    size_t      total_blocks  = 0;
    size_t      free_blocks   = 0;
    size_t      used_blocks   = 0;
    size_t      active_blocks = 0;
    std::string pool_name{"unnamed"};
    size_t      block_size_bytes            = 0;
    size_t      available_blocks            = 0;
    size_t      reserve_blocks              = 0;
    size_t      request_ref_blocks          = 0;
    size_t      remote_connector_ref_blocks = 0;
    size_t      pd_connector_ref_blocks     = 0;
    size_t      block_cache_ref_blocks      = 0;
    size_t      load_ref_blocks             = 0;
    size_t      eviction_target_ref_blocks  = 0;
    size_t      frozen_ref_blocks           = 0;
    float       used_ratio                  = 0.0f;

    std::optional<int64_t> read_bytes;
    std::optional<int64_t> write_bytes;
    std::optional<double>  read_bandwidth_bytes_per_sec;
    std::optional<double>  write_bandwidth_bytes_per_sec;
};

// Allocator metrics take precedence for device pools. Missing BlockTree pools are appended,
// which covers Host, Disk, and any tree-only Device pool.
std::vector<CachePoolMetrics> mergeCachePoolMetrics(const std::vector<CachePoolMetrics>& allocator_metrics_list,
                                                    const std::vector<CachePoolMetrics>& tree_metrics_list);

}  // namespace rtp_llm
