#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

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
    DISABLED_NON_DEVICE_GROUP,   // the HBM-only protocol cannot represent a
                                 // dense reusable group placed on host memory
};

inline KVCacheEventPublisherGate evaluateKVCacheEventPublisherGate(const std::string& type,
                                                                   bool               warmup,
                                                                   int64_t            tp_rank,
                                                                   int64_t            pp_size,
                                                                   bool               cp_sharded,
                                                                   bool               has_reuse_group,
                                                                   bool all_reuse_groups_on_device = true) noexcept {
    if (warmup || isInactiveKVCacheEventPublisherType(type)) {
        return KVCacheEventPublisherGate::DISABLED_INACTIVE;
    }
    // Only the owner rank diagnoses a requested publisher. Returning early on
    // other TP ranks avoids multiplying the same topology/config warning.
    if (tp_rank != 0) {
        return KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK;
    }
    if (!isSupportedKVCacheEventPublisherType(type)) {
        return KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE;
    }
    if (pp_size != 1) {
        return KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL;
    }
    if (cp_sharded) {
        return KVCacheEventPublisherGate::DISABLED_CP_SHARDED;
    }
    if (!has_reuse_group) {
        return KVCacheEventPublisherGate::DISABLED_NO_REUSE_GROUP;
    }
    if (!all_reuse_groups_on_device) {
        return KVCacheEventPublisherGate::DISABLED_NON_DEVICE_GROUP;
    }
    return KVCacheEventPublisherGate::ENABLED;
}

// KVCM takes an authoritative startup snapshot. Installing it first creates a
// lock-based handoff: mutations before the snapshot lock are represented by
// the snapshot, and later mutations are queued as deltas. Publishers without
// a startup snapshot must start before installation so no transition can be
// accepted by SharedBlockCache while the publisher is still NOT_RUNNING.
inline bool installKVCacheEventPublisherBeforeStart(const std::string& type) noexcept {
    return type == kKVCacheEventPublisherKVCM;
}

inline bool shouldReportKVCacheEventMetrics(int64_t tp_rank) noexcept {
    return tp_rank == 0;
}

inline int32_t checkedKVCacheEventInt32(int64_t value, const char* field_name) {
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("KV cache event ") + field_name + " exceeds int32 range");
    }
    return static_cast<int32_t>(value);
}

// Initialization failures are terminal for the attempted publisher instance.
// Preserve cumulative counters and the lifetime high-water mark, but report
// zero current depth after the manager detaches the now-inactive queue. Never
// expose a transient or DISABLED state for this failed lifecycle.
inline PublisherStatus publisherInitializationFailureStatus(const KVCacheEventPublisherPtr& publisher) noexcept {
    auto status       = publisher ? publisher->status() : PublisherStatus{};
    status.state      = PublisherState::CIRCUIT_OPEN;
    status.queue_size = 0;
    return status;
}

inline PublisherStatus publisherGateFailureStatus() noexcept {
    PublisherStatus status;
    status.state = PublisherState::GATED;
    return status;
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
    int64_t     snapshot_max_keys     = 0;
    int64_t     snapshot_max_bytes    = 0;
};

inline KVCacheEventPublisherRawSettings makeKVCacheEventPublisherRawSettings(const KVCacheConfig& source) {
    KVCacheEventPublisherRawSettings raw;
    raw.type                  = source.kv_cache_event_publisher_type;
    raw.manager_endpoint      = source.kv_cache_event_manager_endpoint;
    raw.queue_capacity        = source.kv_cache_event_queue_capacity;
    raw.report_batch_size     = source.kv_cache_event_report_batch_size;
    raw.flush_interval_ms     = source.kv_cache_event_flush_interval_ms;
    raw.heartbeat_interval_ms = source.kv_cache_event_heartbeat_interval_ms;
    raw.request_timeout_ms    = source.kv_cache_event_request_timeout_ms;
    raw.snapshot_timeout_ms   = source.kv_cache_event_snapshot_timeout_ms;
    raw.retry_interval_ms     = source.kv_cache_event_retry_interval_ms;
    raw.snapshot_interval_ms  = source.kv_cache_event_snapshot_interval_ms;
    raw.log_max_keys          = source.kv_cache_event_log_max_keys;
    raw.snapshot_max_keys     = source.kv_cache_event_snapshot_max_keys;
    raw.snapshot_max_bytes    = source.kv_cache_event_snapshot_max_bytes;
    return raw;
}

inline KVCacheEventPublisherConfig deriveKVCacheEventPublisherConfig(const KVCacheEventPublisherRawSettings& raw) {
    const auto require_positive = [](auto value, const char* field_name) {
        if (value <= 0) {
            throw std::runtime_error(std::string("KV cache event ") + field_name + " must be positive");
        }
    };
    require_positive(raw.queue_capacity, "queue capacity");
    require_positive(raw.report_batch_size, "report batch size");
    require_positive(raw.flush_interval_ms, "flush interval");
    require_positive(raw.heartbeat_interval_ms, "heartbeat interval");
    require_positive(raw.request_timeout_ms, "request timeout");
    require_positive(raw.snapshot_timeout_ms, "snapshot timeout");
    require_positive(raw.retry_interval_ms, "retry interval");
    require_positive(raw.snapshot_interval_ms, "snapshot interval");
    require_positive(raw.snapshot_max_keys, "snapshot key count");
    require_positive(raw.snapshot_max_bytes, "snapshot byte size");
    if (raw.log_max_keys < 0) {
        throw std::runtime_error("KV cache event log key count must be non-negative");
    }
    if (raw.queue_capacity > static_cast<int64_t>(kKVCacheEventMaxQueueCapacity)) {
        throw std::runtime_error("KV cache event queue capacity exceeds resource safety limit");
    }
    if (raw.report_batch_size > static_cast<int64_t>(kKVCacheEventMaxReportBatchSize)) {
        throw std::runtime_error("KV cache event report batch size exceeds resource safety limit");
    }
    if (raw.snapshot_max_keys > static_cast<int64_t>(kKVCacheEventMaxSnapshotKeys)) {
        throw std::runtime_error("KV cache event snapshot key count exceeds resource safety limit");
    }
    if (raw.snapshot_max_bytes > static_cast<int64_t>(kKVCacheEventMaxSnapshotBytes)) {
        throw std::runtime_error("KV cache event snapshot byte size exceeds resource safety limit");
    }
    KVCacheEventPublisherConfig config;
    config.type                   = raw.type;
    config.manager_endpoint       = raw.manager_endpoint;
    config.queue_capacity         = static_cast<size_t>(raw.queue_capacity);
    config.report_batch_size      = static_cast<size_t>(raw.report_batch_size);
    config.flush_interval_ms      = raw.flush_interval_ms;
    config.heartbeat_interval_ms  = raw.heartbeat_interval_ms;
    config.request_timeout_ms     = raw.request_timeout_ms;
    config.snapshot_timeout_ms    = raw.snapshot_timeout_ms;
    config.retry_interval_ms      = raw.retry_interval_ms;
    config.snapshot_interval_ms   = raw.snapshot_interval_ms;
    config.log_max_keys_per_batch = static_cast<size_t>(raw.log_max_keys);
    config.snapshot_max_keys      = static_cast<size_t>(raw.snapshot_max_keys);
    config.snapshot_max_bytes     = static_cast<size_t>(raw.snapshot_max_bytes);
    return config;
}

inline std::string resolveKVCacheEventInstanceGroup(const std::string& event_instance_group,
                                                    const std::string& reco_instance_group) {
    return event_instance_group.empty() ? reco_instance_group : event_instance_group;
}

struct KVCacheEventPublisherContextSettings {
    std::string event_instance_group;
    std::string reco_instance_group;
    std::string instance_id;
    std::string host_ip_port;
    std::string model_name;
    std::string dtype;
    int32_t     block_size_tokens = 0;
    int64_t     spec_size_bytes   = 0;
    int32_t     tp_size           = 1;
    int32_t     dp_size           = 1;
    int32_t     pp_size           = 1;
    int32_t     dp_rank           = 0;
    bool        use_mla           = false;
};

inline KVCacheEventPublisherContextSettings makeKVCacheEventPublisherContextSettings(const KVCacheConfig& source) {
    KVCacheEventPublisherContextSettings settings;
    settings.event_instance_group = source.kv_cache_event_instance_group;
    settings.reco_instance_group  = source.reco_instance_group;
    settings.instance_id          = source.kv_cache_event_instance_id;
    settings.host_ip_port         = source.kv_cache_event_host_ip_port;
    return settings;
}

inline KVCacheEventPublisherContext
makeKVCacheEventPublisherContext(const KVCacheEventPublisherContextSettings& settings) {
    KVCacheEventPublisherContext context;
    context.instance_group =
        resolveKVCacheEventInstanceGroup(settings.event_instance_group, settings.reco_instance_group);
    context.instance_id  = settings.instance_id;
    context.host_ip_port = settings.host_ip_port;
    context.model_name   = settings.model_name;
    context.dtype        = settings.dtype;
    context.spec_name    = "rtp_llm_hbm_" + std::to_string(settings.block_size_tokens);
    context.location_uri =
        "rtp-llm://" + settings.host_ip_port + "/hbm?size=" + std::to_string(settings.spec_size_bytes);
    context.block_size_tokens = settings.block_size_tokens;
    context.spec_size_bytes   = settings.spec_size_bytes;
    context.tp_size           = settings.tp_size;
    context.dp_size           = settings.dp_size;
    context.pp_size           = settings.pp_size;
    context.dp_rank           = settings.dp_rank;
    context.use_mla           = settings.use_mla;
    return context;
}

// The published location is one logical DP-replica endpoint, so the aggregate
// spec accounts for the same block's shards on every TP rank even though only
// tp_rank=0 owns event publication.
inline std::optional<int64_t> aggregateKVCacheEventSpecSizeBytes(const std::vector<int64_t>& group_block_size_bytes,
                                                                 int64_t                     tp_size) {
    if (tp_size <= 0) {
        return std::nullopt;
    }
    int64_t total = 0;
    for (const auto group_size : group_block_size_bytes) {
        if (group_size <= 0 || total > std::numeric_limits<int64_t>::max() - group_size) {
            return std::nullopt;
        }
        total += group_size;
    }
    if (total <= 0 || tp_size > std::numeric_limits<int64_t>::max() / total) {
        return std::nullopt;
    }
    return total * tp_size;
}

}  // namespace rtp_llm
