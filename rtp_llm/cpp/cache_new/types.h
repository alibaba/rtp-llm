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
    const std::vector<int64_t>& cache_keys,
    const std::vector<int32_t>& block_indices,
};

struct MatchResult {
    size_t                          reuse_length;
    std::vector<std::vector<int>>   block_indices;   
};

struct MallocResult {
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources;
    MatchResult match_result;
};

// initKVBlock 的场景： 
// enable_reuse_cache = true，基于输入的 cache_keys 做 block_cache 匹配，再走 block_pool 分配；
// incrKVBlock 的场景： enable_reuse_cache = false, 直接走 block_pool 分配

struct MallocInfo {
    MallocInfo(int64_t                                  request_id,
               const std::vector<int32_t>&              token_ids,
               const std::shared_ptr<std::vector<int64_t>>&              cache_keys,
               const std::vector<std::vector<int32_t>>& mm_bounds    = {},
               bool                                     need_loss    = false,
               bool                                     verbose      = false):
        request_id(request_id),
        token_ids(token_ids),
        cache_keys(cache_keys),
        mm_bounds(mm_bounds),
        need_loss(need_loss),
        verbose(verbose) {}

    int64_t                                 request_id;
    const std::vector<int32_t>&             token_ids;
    const std::vector<int64_t>&             cache_keys;
    const std::vector<std::vector<int32_t>> mm_bounds = {};
    bool                                    need_loss = false;
    bool                                    verbose   = false;
    bool                                    enable_reuse_cache = true;
};

struct FreeInfo { 
    FreeInfo(int64_t                                                     request_id,
             const std::vector<std::shared_ptr<GroupedKVCacheResource>>& grouped_kv_cache_resources):
        request_id(request_id),
        grouped_kv_cache_resources(grouped_kv_cache_resources) {}
    
    int64_t                                              request_id; 
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources; // share_ptr 析构自动 free block_indices
};


struct FreeResult {};

struct InsertInfo {
    InsertInfo(int64_t                                                     request_id,
               const std::vector<std::shared_ptr<GroupedKVCacheResource>>& grouped_kv_cache_resources):
        request_id(request_id),
        grouped_kv_cache_resources(grouped_kv_cache_resources) {}
    
    int64_t                                              request_id;
    std::vector<std::shared_ptr<GroupedKVCacheResource>> grouped_kv_cache_resources;
};


struct InsertResult {};

}