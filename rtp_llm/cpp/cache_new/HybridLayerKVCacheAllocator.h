#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

// HybridLayerKVCacheAllocator is used for model with different kinds of attentions
// and each kind of attentions share the same block_pool.

class HybridLayerKVCacheAllocator: public KVCacheAllocator {
public:
    HybridLayerKVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);
    bool init() {
        // 1. build a hybrid pool for all kv_cache_groups_
        // 2. build kv_cache_groups_ by CacheConfig and the hybrid pool
    };

    MallocResult malloc(const MallocInfo& malloc_info) {
        // if no blocks allocated in stream {
        //     return mallocWithCache(malloc_info);
        // } else {
        //     return mallocSimple(malloc_info);
        // }
    }

    FreeResult free(const FreeInfo& free_info) {
        // only consider the scenario of full fallback.
    }
    InsertResult insertIntoCache(const InsertInfo& insert_info){
        // insert blocks in stream that have been cached in block_cache into block_cache   
    };

    CacheLayerLayout layerCacheBase() const {
        
    };

private:
    MallocResult mallocWithCache(const MallocInfo& malloc_info) {
        // MallocResult malloc_result;
        // int full_reuse_len = INT_MAX;
        // std::vector<MatchResult> match_results;
        // auto cache_keys = malloc_info.stream->kvCache().cache_keys;

        // for (auto& kv_cache_group : kv_cache_groups_) {   
        //     auto match_result = kv_cache_group->match(cache_keys);
        //     match_results.push_back(match_result);
            
        //     if (kv_cache_group->type() == KVCacheType::FULL && match_result.reuse_length < full_reuse_len) {
        //         full_reuse_len = match_result.reuse_length;
        //     }
        // }

        // int reuse_len = 0;
        // for (int i = full_reuse_len - 1; i >= 0; i--) {
        //     if (cache_keys[i] in all match_result.cached_keys) {
        //         reuse_len = i+1;
        //         break;
        //     }
        // }
        
        // for (auto& match_result : match_results) {
        //     // update stream's BatchKVCacheResource
        // }

        // for (auto& kv_cache_group : kv_cache_groups_) {
        //     auto block_indices = kv_cache_group->alloc(cache_keys, reuse_len);
        //     // update stream's BatchKVCacheResource
        // }

        // return malloc_result;
    }

    MallocResult mallocSimple(const MallocInfo& malloc_info) {
        // for(auto& kv_cache_group : kv_cache_groups_) {
        //     // cache_keys = cache_keys that are not allocated blocks;
        //     auto block_indices = kv_cache_group->alloc(cache_keys, 0);
        //     // update stream's BatchKVCacheResource
        //     if (kv_cache_group->type() == KVCacheType::LINEAR) {
        //         // insert previous blocks into block_cache and free it
        //     }
        // }
    }
};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocatorPtr>;

}  // namespace rtp_llm
