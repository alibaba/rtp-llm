#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Lightweight block descriptor for cache-store / RPC use cases.
// Upper layers may convert (device, scalar_type) to rtp_llm::MemoryType/DataType and
// build Buffer views as needed.
// This header is intentionally kept dependency-free so low-level transfer modules
// (e.g. rtp_llm::transfer::tcp) can include it without pulling in engine_base/stream.
struct BlockInfo {
    // Legacy CUDA/host distinction, retained for existing aggregate initializers.
    // New producers also set device_type from the underlying tensor.
    bool    is_cuda      = false;
    int32_t device_index = 0;

    int32_t scalar_type = 0;  // c10::ScalarType

    void*  addr       = nullptr;
    size_t size_bytes = 0;

    // c10::DeviceType of the backing tensor. Keep this at the end so existing
    // aggregate initializers remain source compatible. -1 denotes a legacy
    // descriptor whose device kind must be resolved from is_cuda.
    int32_t device_type = -1;
};

struct BlockInfoPair {
    BlockInfo kv;
    BlockInfo kv_scale;
};

struct KVPartitionBytes {
    size_t k_off = 0;
    size_t k_sz  = 0;
    size_t v_off = 0;
    size_t v_sz  = 0;
};

}  // namespace rtp_llm
