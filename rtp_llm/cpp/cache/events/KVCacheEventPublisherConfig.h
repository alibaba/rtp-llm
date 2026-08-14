#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtp_llm {

inline constexpr const char* kKVCacheEventPublisherNone = "none";
inline constexpr const char* kKVCacheEventPublisherLog  = "log";
inline constexpr const char* kKVCacheEventPublisherKVCM = "kvcm";

// Resource ceilings are part of the exporter isolation boundary. They cap
// eager resident memory (queue and logical mirror) separately from transient
// request construction and response parsing, so a bad direct C++
// configuration or an untrusted peer cannot turn this optional subsystem into
// an unbounded allocation. The public defaults stay
// comfortably below the queue/batch ceilings; snapshot defaults deliberately
// equal their ceilings because KVCM snapshots are authoritative and cannot be
// paginated.
inline constexpr size_t kKVCacheEventMaxQueueCapacity   = size_t{1} << 20;
inline constexpr size_t kKVCacheEventMaxReportBatchSize = size_t{1} << 14;
inline constexpr size_t kKVCacheEventMaxSnapshotKeys    = 1000000;
inline constexpr size_t kKVCacheEventMaxSnapshotBytes   = size_t{256} * 1024 * 1024;
inline constexpr size_t kKVCacheEventMaxReportBytes     = size_t{16} * 1024 * 1024;
inline constexpr size_t kKVCacheEventMaxResponseBytes   = size_t{1} * 1024 * 1024;

inline bool isSupportedKVCacheEventPublisherType(const std::string& type) noexcept {
    return type == kKVCacheEventPublisherLog || type == kKVCacheEventPublisherKVCM;
}

inline bool isInactiveKVCacheEventPublisherType(const std::string& type) noexcept {
    return type.empty() || type == kKVCacheEventPublisherNone;
}

struct KVCacheEventPublisherConfig {
    std::string type = kKVCacheEventPublisherNone;

    std::string manager_endpoint;

    size_t queue_capacity         = 100000;
    size_t report_batch_size      = 1000;
    int    flush_interval_ms      = 20;
    int    heartbeat_interval_ms  = 1000;
    int    request_timeout_ms     = 1500;
    int    snapshot_timeout_ms    = 30000;
    int    retry_interval_ms      = 500;
    int    snapshot_interval_ms   = 300000;
    size_t log_max_keys_per_batch = 8;
    size_t snapshot_max_keys      = 1000000;
    size_t snapshot_max_bytes     = 256 * 1024 * 1024;
    // Internal hard limit for registration, control, and mutation JSON. It is
    // intentionally not deployment-configurable; tests may lower it to
    // exercise fail-closed serialization without allocating a huge string.
    size_t report_max_bytes = kKVCacheEventMaxReportBytes;
};

struct KVCacheEventPublisherContext {
    std::string instance_group;
    std::string instance_id;
    std::string host_ip_port;
    std::string model_name;
    std::string dtype;
    std::string spec_name;
    std::string location_uri;

    int32_t block_size_tokens = 0;
    int64_t spec_size_bytes   = 0;
    int32_t tp_size           = 1;
    int32_t dp_size           = 1;
    int32_t pp_size           = 1;
    int32_t dp_rank           = 0;
    bool    use_mla           = false;
};

}  // namespace rtp_llm
