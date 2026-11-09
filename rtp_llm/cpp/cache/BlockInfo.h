#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Each 128-element FP8 row has one FP32 scale or four replicated UE8M0
// exponent bytes. Arbitrary FP32 scales require payload requantization for MX.
// Unwritten rows have unspecified contents and must not be validated as tokens.
enum class IndexerScaleFormat : uint32_t {
    NONE  = 0,
    FP32  = 1,
    UE8M0 = 2
};

struct IndexerCacheLayout {
    IndexerScaleFormat format      = IndexerScaleFormat::NONE;
    uint32_t           page_tokens = 0;
    uint32_t           page_stride = 0;

    template<class Wire>
    void toWire(Wire* wire) const {
        if (format != IndexerScaleFormat::NONE) {
            wire->set_indexer_scale_format(static_cast<uint32_t>(format));
            wire->set_indexer_page_tokens(page_tokens);
            wire->set_indexer_page_stride(page_stride);
        }
    }
    template<class Wire>
    static IndexerCacheLayout fromWire(const Wire& wire) {
        return {static_cast<IndexerScaleFormat>(wire.indexer_scale_format()),
                wire.indexer_page_tokens(),
                wire.indexer_page_stride()};
    }
};

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

    void*              addr       = nullptr;
    size_t             size_bytes = 0;
    IndexerCacheLayout indexer_layout;
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
