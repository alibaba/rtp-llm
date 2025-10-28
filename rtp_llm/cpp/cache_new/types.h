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

// is_reuse_cache = true，基于输入的 cache_keys 做 block_cache 匹配，再走 block_pool 分配；
// is_reuse_cache = false, 直接走 block_pool 分配

struct MallocInfo {
    MallocInfo(void* stream): stream(stream) {}

    void* stream;
};

struct MallocResult {
    bool        success;
    MatchResult match_result;
};

// fallback
struct FreeInfo {
    FreeInfo(void* stream): stream(stream) {}
    void* stream;
};

struct FreeResult {
    bool success;
};

struct InsertInfo {
    InsertInfo(void* stream): stream(stream) {}
    void* stream;
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