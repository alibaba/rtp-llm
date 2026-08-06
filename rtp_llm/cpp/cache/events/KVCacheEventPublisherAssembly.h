#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm {

// Pure helpers that turn KVCacheConfig values into publisher wiring decisions.
// They are kept free of cache/GPU dependencies so the gating and derivation
// rules used by KVCacheManager::initCacheEventPublisher stay unit-testable.

enum class KVCacheEventPublisherGate {
    ENABLED,
    DISABLED_INACTIVE,           // warmup, empty type or explicit "none"
    DISABLED_UNKNOWN_TYPE,       // unrecognized type; deserves a warning
    DISABLED_PIPELINE_PARALLEL,  // pp_size > 1 cannot elect a unique owner
    DISABLED_CP_SHARDED,         // CP sharded KV cache publishes per-rank keys
                                 // whose token granularity differs from the
                                 // external logical block size; unsupported
    DISABLED_NON_OWNER_RANK,     // only tp_rank == 0 publishes events
    DISABLED_NO_REUSE_GROUP,     // no cache group participates in prefix reuse
};

inline KVCacheEventPublisherGate evaluateKVCacheEventPublisherGate(const std::string& type,
                                                                   bool               warmup,
                                                                   int64_t            tp_rank,
                                                                   int64_t            pp_size,
                                                                   bool               cp_sharded,
                                                                   bool               has_reuse_group) noexcept {
    if (warmup || type.empty() || type == "none") {
        return KVCacheEventPublisherGate::DISABLED_INACTIVE;
    }
    if (type != "log" && type != "kvcm") {
        return KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE;
    }
    if (pp_size != 1) {
        return KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL;
    }
    if (cp_sharded) {
        return KVCacheEventPublisherGate::DISABLED_CP_SHARDED;
    }
    if (!isKVCacheEventPublisherOwner(tp_rank, pp_size)) {
        return KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK;
    }
    if (!has_reuse_group) {
        return KVCacheEventPublisherGate::DISABLED_NO_REUSE_GROUP;
    }
    return KVCacheEventPublisherGate::ENABLED;
}

// Raw, unclamped values as they appear in KVCacheConfig.
struct KVCacheEventPublisherRawSettings {
    std::string type;
    std::string manager_endpoint;
    int64_t     queue_capacity        = 0;
    int64_t     report_batch_size     = 0;
    int         flush_interval_ms     = 0;
    int         heartbeat_interval_ms = 0;
    int         request_timeout_ms    = 0;
    int         snapshot_timeout_ms   = 0;
    int         retry_interval_ms     = 0;
    int         snapshot_interval_ms  = 0;
    int64_t     log_max_keys          = 0;
};

inline KVCacheEventPublisherConfig deriveKVCacheEventPublisherConfig(const KVCacheEventPublisherRawSettings& raw) {
    KVCacheEventPublisherConfig config;
    config.type                  = raw.type;
    config.manager_endpoint      = raw.manager_endpoint;
    config.queue_capacity        = static_cast<size_t>(std::max<int64_t>(raw.queue_capacity, 1));
    config.report_batch_size     = static_cast<size_t>(std::max<int64_t>(raw.report_batch_size, 1));
    config.flush_interval_ms     = std::max(raw.flush_interval_ms, 1);
    config.heartbeat_interval_ms = std::max(raw.heartbeat_interval_ms, 1);
    config.request_timeout_ms    = std::max(raw.request_timeout_ms, 1);
    config.snapshot_timeout_ms   = std::max(raw.snapshot_timeout_ms, 1);
    config.retry_interval_ms     = std::max(raw.retry_interval_ms, 1);
    config.snapshot_interval_ms  = std::max(raw.snapshot_interval_ms, 1);
    config.log_max_keys_per_batch = static_cast<size_t>(std::max<int64_t>(raw.log_max_keys, 0));
    return config;
}

inline std::string resolveKVCacheEventInstanceGroup(const std::string& event_instance_group,
                                                    const std::string& reco_instance_group) {
    return event_instance_group.empty() ? reco_instance_group : event_instance_group;
}

// The published location is one logical DP-replica endpoint, so the aggregate
// spec accounts for the same block's shards on every TP rank even though only
// tp_rank=0 owns event publication.
inline int64_t aggregateKVCacheEventSpecSizeBytes(const std::vector<int64_t>& group_block_size_bytes,
                                                  int64_t                     tp_size) {
    int64_t total = 0;
    for (const auto group_size : group_block_size_bytes) {
        total += group_size;
    }
    return total * std::max<int64_t>(tp_size, 1);
}

}  // namespace rtp_llm
