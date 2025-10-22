#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {


// SingleTypedKVCacheAllocator is used for model with full attentions only

class SingleTypeKVCacheAllocator : public KVCacheAllocator {
public: 
    SingleTypedKVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);
    bool init() {
        // 1. build a pool by CacheConfig
        // 2. build a kv_cache_group by the pool
    };

    MallocResult malloc(const MallocInfo& malloc_info) {
        // if no blocks allocated in stream {
        //     return mallocWithCache(malloc_info);
        // } else {
        //     return mallocSimple(malloc_info);
        // }
    }

    FreeResult free(const FreeInfo& free_info) {
       // 
    }

    InsertResult insertIntoCache(const InsertInfo& insert_info) {
        // 
    }

private:
    MallocResult mallocWithCache(const MallocInfo& malloc_info) {
        // 1. match cache_keys with block_cache
        // 2. alloc blocks from block_pool
    }

    MallocResult mallocSimple(const MallocInfo& malloc_info) {
        // 1. alloc blocks from block_pool
    }
};

using SingleTypeKVCacheAllocator = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
