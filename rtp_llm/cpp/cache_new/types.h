#pragma once

#include <vector>
#include <cstdint>

#include <map>

namespace rtp_llm {


struct CacheLayerLayout {
    vector<int> layer_to_groups;
    vector<BufferPtr> layers_to_buffer_ptrs;
}


// KVCacheResource share_ptr 析构自动 kv cache manager free block_indices
struct GroupedKVCacheResource {
    std::vector<int32_t> block_indices,
};

struct MatchResult {
    size_t                          reuse_length;
    std::vector<std::vector<int>>   block_indices;   
};

// is_reuse_cache = true，基于输入的 cache_keys 做 block_cache 匹配，再走 block_pool 分配；
// is_reuse_cache = false, 直接走 block_pool 分配

struct MallocInfo {
    MallocInfo(GenerateStreamPtr stream, bool is_reuse_cache = false):
        stream(stream), is_reuse_cache(is_reuse_cache) {}

    GenerateStreamPtr stream;
    bool is_reuse_cache;
};


struct MallocResult {
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources;
    MatchResult match_result;
};

struct FreeInfo { 
    FreeInfo(GenerateStreamPtr                                           stream,
             const std::vector<std::shared_ptr<GroupedKVCacheResource>>& grouped_kv_cache_resources):
        stream(stream),
        grouped_kv_cache_resources(grouped_kv_cache_resources) {}
    GenerateStreamPtr stream;
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources; // share_ptr 析构自动 free block_indices
};


struct FreeResult {
    bool success;
};

struct InsertInfo {
    InsertInfo(GenerateStreamPtr                                           stream,
               const std::vector<std::shared_ptr<GroupedKVCacheResource>>& grouped_kv_cache_resources):
        stream(stream),
        grouped_kv_cache_resources(grouped_kv_cache_resources) {}
    int64_t                                              request_id;
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources;
};


struct InsertResult {
    bool success;
};

}