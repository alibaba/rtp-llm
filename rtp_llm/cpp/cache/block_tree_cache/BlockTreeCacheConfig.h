#pragma once

#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"

namespace rtp_llm {

enum class LastTierEvictionMode {
    ON_DEMAND,
    WATERMARK
};

struct TierWatermark {
    double low_ratio{0.0};
    double high_ratio{0.0};
};

struct EvictionBatchLimits {
    size_t max_pending_bytes;
    size_t max_descriptors_per_batch;
};

// Unified configuration for BlockTreeCache behavior and pool sizing.
struct BlockTreeCacheConfig {
    // ---- Tier enable flags ----
    bool enable_device_cache{true};
    bool enable_host_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};

    // ---- Per-tier watermark ----
    TierWatermark device_watermark;
    TierWatermark host_watermark;
    TierWatermark disk_watermark;

    // ---- Per-tier eviction policy ----
    EvictionPolicy device_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy host_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy disk_eviction_policy{EvictionPolicy::LRU};

    // ---- Shared load/eviction task pool ----
    int    task_pool_size{4};
    size_t business_queue_max_size{10000};

    // ---- Per-rank WorkLoop owners or legacy workers per direction ----
    size_t transfer_worker_count{2};
    size_t transfer_queue_max_size{10000};

    // ---- Transfer deadlines, including queue wait and cross-rank work ----
    int device_host_timeout_ms{10000};
    int disk_to_device_timeout_ms{3000};
    int host_to_disk_timeout_ms{3000};

    // Async: blocks per load/evict pair, split by direction and shared by groups.
    // Legacy: rank-wide byte budget measured in maximum-stride blocks.
    size_t               device_disk_staging_block_count{128};
    size_t               max_descriptors_per_transfer_batch{32};
    size_t               device_evict_max_pending_bytes{0};
    size_t               host_evict_max_pending_bytes{0};
    LastTierEvictionMode last_tier_evict_mode{LastTierEvictionMode::ON_DEMAND};

    // ---- FULL prefix invariant scanner (diagnostic only) ----
    // The factory zeroes this on ranks that do not own a mutable BlockTree.
    int64_t full_prefix_scan_interval_ms{30000};  // 30 seconds; 0 = disabled, no thread

    // ---- Query helpers ----
    bool isTierEnabled(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return enable_device_cache;
            case Tier::HOST:
                return enable_host_cache;
            case Tier::DISK:
                return enable_disk_cache;
            case Tier::REMOTE:
                return enable_remote_cache;
            default:
                return false;
        }
    }

    Tier evictionTarget(Tier source) const {
        if (source == Tier::DEVICE && enable_host_cache) {
            return Tier::HOST;
        }
        return source == Tier::HOST && enable_disk_cache ? Tier::DISK : Tier::NONE;
    }

    bool usesWatermark(Tier tier) const {
        return tier < Tier::REMOTE && isTierEnabled(tier)
               && (evictionTarget(tier) != Tier::NONE || last_tier_evict_mode == LastTierEvictionMode::WATERMARK);
    }

    TierWatermark watermarkForTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return device_watermark;
            case Tier::HOST:
                return host_watermark;
            case Tier::DISK:
                return disk_watermark;
            default:
                return {};
        }
    }
};

}  // namespace rtp_llm
