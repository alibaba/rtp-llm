#pragma once

#include <vector>
#include <cstdint>

#include <map>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

struct CacheLayerLayout {
    std::vector<int>       layer_to_groups;
    std::vector<BufferPtr> layers_to_buffer_ptrs;
};

struct MatchResult {
    size_t                            reuse_length;
    std::vector<std::vector<int64_t>> cached_keys;
    std::vector<std::vector<int>>     block_indices;
};

struct MallocInfo {
    MallocInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}

    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr complete_token_ids;
};

struct MallocResult {
    bool    success;
    int     reuse_len;
};

// fallback 
struct FreeInfo { 
    FreeInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}
    BatchKVCacheResourcePtr batch_kv_cache_resource;
};

struct FreeResult {
    bool success;
};

struct InsertInfo {
    InsertInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}
    BatchKVCacheResourcePtr batch_kv_cache_resource;
};

struct InsertResult {
    bool success;
};

typedef size_t  CacheKeyType;
typedef int32_t BlockIdxType;

int32_t NULL_BLOCK_IDX = -1;

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

}  // namespace rtp_llm