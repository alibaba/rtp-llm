#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {

// Keep these client-side admission limits aligned with KvMetaManager::Limits.
// They reject an incompatible deployment before any metadata or data-plane I/O.
inline constexpr std::size_t   kMMKvcmMaxBatchItems         = 64;
inline constexpr std::size_t   kMMKvcmMaxKeyBytes           = 512;
inline constexpr std::size_t   kMMKvcmMaxInstanceIdBytes    = 512;
inline constexpr std::size_t   kMMKvcmMaxInstanceGroupBytes = 512;
inline constexpr std::size_t   kMMKvcmMaxUserDataBytes      = 64 * 1024;
inline constexpr std::uint64_t kMMKvcmMaxObjectBytes        = 1ULL * 1024 * 1024 * 1024;
inline constexpr std::uint64_t kMMKvcmMaxBatchBytes         = 4ULL * 1024 * 1024 * 1024;
inline constexpr std::int32_t  kMMKvcmMaxWriteTimeoutSecs   = 1800;
// The shared gRPC control client can retain at most 1024 pending release
// handles.  Keep one valid KVCM receipt within that bound so it cannot be
// rejected as oversized by an otherwise empty async-release queue.
inline constexpr std::size_t kMMKvcmMaxObjectsPerReceipt = 1024;
inline constexpr std::size_t kMMKvcmMaxLogicalValues     = 16384;
inline constexpr std::size_t kMMKvcmMaxTensorDimensions  = 16;

// Configuration for the isolated KVMeta exact-size object path used by EPD
// multimodal embeddings. This is independent of the KV-cache connector.
struct MMKvcmConfig {
    std::vector<std::string> addresses;
    std::string              instance_id;
    std::string              instance_group;
    std::string              user_data;
    std::string              transfer_client_config;
    uint32_t                 call_timeout_ms       = 3000;
    int32_t                  write_timeout_seconds = 30;
    // Default request budget is 120s; retain objects for one extra minute so
    // fallback GC cannot race an otherwise valid multimodal request.
    int64_t object_gc_timeout_ms = 180 * 1000;
    int64_t max_object_bytes     = 1024LL * 1024 * 1024;
    int64_t max_receipt_bytes    = 8LL * 1024 * 1024 * 1024;
};

}  // namespace rtp_llm
