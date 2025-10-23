#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroupSpec.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCache.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    std::vector<int> alloc(vector<int64_t> cache_keys, int reuse_len = 0)
    // void reuse_len(vector<int64_t> cache_keys) const;
    MatchResult match(vector<int64_t> cache_keys) const;
    void free(vector<int> block_indices);
    void insertIntoCache(vector<int64_t> cache_keys, vector<int> block_indices);

    int get_block_stride() const;

private:
    // global_layer_id
    vector<int> layer_ids_;
    KVCacheSpec group_spec_;
    BlockCachePtr block_cache_;
    BlockPoolPtr block_pool_;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm

