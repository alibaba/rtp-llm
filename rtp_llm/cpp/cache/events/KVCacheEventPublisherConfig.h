#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtp_llm {

struct KVCacheEventPublisherConfig {
    std::string type = "none";

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

// Pipeline-parallel ranks do not expose a stable pp_rank today, so there is no
// safe way to select exactly one cache-event owner across PP stages. Keep the
// feature disabled for pp_size > 1 until that identity is available.
inline bool isKVCacheEventPublisherOwner(int64_t tp_rank, int64_t pp_size) noexcept {
    return tp_rank == 0 && pp_size == 1;
}

}  // namespace rtp_llm
