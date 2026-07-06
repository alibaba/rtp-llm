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
    // is_cuda: true when the backing storage is on an accelerator device
    // (CUDA or XPU), false when on host/CPU memory. Despite the name, this is
    // NOT CUDA-specific -- XPU blocks also set it to true. Taken from the
    // underlying tensor and kept as raw values to avoid torch->rtp conversions
    // inside cache.
    // TODO(xpu) [MUST-DO next iteration]: rename is_cuda -> is_device_memory
    // (or is_accelerator) and update all read/write sites. Deferred here to
    // keep this PR low-risk; the name is misleading for non-CUDA accelerators.
    bool    is_cuda = false;
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
