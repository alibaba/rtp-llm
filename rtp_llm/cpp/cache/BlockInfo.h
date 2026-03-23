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
    // Torch device of the backing storage (CPU/CUDA), taken from the underlying tensor.
    // Kept as raw values to avoid torch->rtp conversions inside cache.
    bool    is_cuda      = false;
    int32_t device_index = 0;

    int32_t scalar_type = 0;  // c10::ScalarType

    void*  addr       = nullptr;
    size_t size_bytes = 0;
};

struct BlockInfoPair {
    BlockInfo kv;
    BlockInfo kv_scale;
};

}  // namespace rtp_llm
