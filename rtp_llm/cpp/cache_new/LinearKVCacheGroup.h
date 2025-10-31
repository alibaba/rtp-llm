#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class LinearKVCacheGroup: public KVCacheGroup {
public:
    void                              malloc(CacheKeysType& cache_keys, int reuse_len = 0);
    MatchResult                       match(CacheKeysType& cache_keys);
    void                              free(BlockIndicesType& block_indices);
    void                              insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices);
    std::map<BlockIdxType, BufferPtr> blockBuffer(BlockIdxType block_id, CacheKeyType cache_key);
    KVCacheType                       type() const;
    void                              removeSkippedBlocks(BlockIndicesType& block_indices);

private:
    vector<int>   layer_ids_;
    KVCacheSpec   group_spec_;
    BlockCachePtr block_cache_;
    BlockPoolPtr  block_pool_;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
