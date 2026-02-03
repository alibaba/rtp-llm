#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

constexpr int32_t NULL_BLOCK_IDX = -1;

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

struct BlockAddrInfo {
    void* kv_addr       = nullptr;
    void* kv_scale_addr = nullptr;
};

// Lightweight block descriptor for cache-store / RPC use cases.
// Upper layers may convert (device, scalar_type) to rtp_llm::MemoryType/DataType and build Buffer views as needed.
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

struct KVCacheInfo {
    size_t                    available_kv_cache = 0;
    size_t                    total_kv_cache     = 0;
    size_t                    block_size         = 0;
    std::vector<CacheKeyType> cached_keys;
    int64_t                   version = -1;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;
};

struct MallocInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id          = 0;
    bool                    verbose             = true;  // for failed log
    bool                    enable_device_cache = true;

    std::vector<std::vector<int>> mm_intervals;  // for mm multimodal
};

struct MallocResult {
    bool success;
    int  reuse_len;

    int64_t match_cost_time_us = 0;
};

struct FreeInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;

    int64_t request_id = 0;
};

struct InsertInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident;
};

}  // namespace rtp_llm