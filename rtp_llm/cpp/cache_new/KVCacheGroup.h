#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroupSpec.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCache.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    // add reused blocks' reference count
    std::vector<int> alloc(vector<int64_t> cache_keys, int reuse_len = 0)
    MatchResult match(vector<int64_t> cache_keys) const;
    void free(vector<int> block_indices);
    void insertIntoCache(vector<int64_t> cache_keys, vector<int> block_indices);
    
    std::map<int, BufferPtr> blockBuffer(int block_id, int64_t cache_key);
    
    KVCacheType type() const;
private:
    // evict first if block_pool's blocks are not enough when alloc 
    bool evict(int need_evict_len);

    vector<int> layer_ids_;
    KVCacheSpec group_spec_;
    BlockCachePtr block_cache_;
    BlockPoolPtr block_pool_;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm

