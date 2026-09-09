#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

// Configuration shared by RDMA providers and their transport adapters.
struct RdmaConfig {
    // Empty bind IP enables automatic detection.
    std::string bind_ip;
    int         port               = 0;
    int         connect_timeout_ms = 250;
    // Per-read limit. A caller may further cap it by its request deadline.
    int64_t read_timeout_ms = 3000;
    // Parallel RC QPs used to stripe one descriptor read.
    uint32_t qp_count = 8;
    // Reclaim exported slots that were not released.
    int64_t slot_gc_timeout_ms = 60 * 1000;
    // Larger outputs are split across slots.
    int64_t max_slot_bytes = 1024L * 1024 * 1024;
    // Aggregate payload accepted from one multimodal receipt. Matches the total RDMA memory pool.
    int64_t max_receipt_bytes = 8L * 1024 * 1024 * 1024;
};

}  // namespace rtp_llm
