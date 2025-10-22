#pragma once

#include <vector>
#include <cstdint>

#include <map>

namespace rtp_llm {


struct CacheLayerLayout {
    vector<int> layer_to_groups;
    vector<BufferPtr> layers_to_buffer_ptrs;
}

struct MatchResult {
    size_t                              reuse_length;
    std::vector<std::vector<int64_t>>   cached_keys;
    std::vector<std::vector<int>>       block_indices;   
};

// is_reuse_cache = true，基于输入的 cache_keys 做 block_cache 匹配，再走 block_pool 分配；
// is_reuse_cache = false, 直接走 block_pool 分配

struct MallocInfo {
    MallocInfo(GenerateStreamPtr stream):
        stream(stream) {}

    GenerateStreamPtr stream;
};


struct MallocResult {
    bool success;
    MatchResult match_result;
};

// fallback 
struct FreeInfo { 
    FreeInfo(GenerateStreamPtr stream):
        stream(stream) {}
    GenerateStreamPtr stream;
};


struct FreeResult {
    bool success;
};

struct InsertInfo {
    InsertInfo(GenerateStreamPtr stream):
        stream(stream) {}
    GenerateStreamPtr stream;
};


struct InsertResult {
    bool success;
};

}