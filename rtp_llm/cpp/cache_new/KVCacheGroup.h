#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroupSpec.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    void                              malloc(CacheKeysType& cache_keys, int reuse_len = 0)                        = 0;
    MatchResult                       match(CacheKeysType& cache_keys)                                            = 0;
    void                              free(BlockIndicesType& block_indices)                                       = 0;
    void                              insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices) = 0;
    std::map<BlockIdxType, BufferPtr> blockBuffer(BlockIdxType block_id, CacheKeyType cache_key)                  = 0;
    KVCacheType                       type() const                                                                = 0;
    void                              removeSkippedBlocks(BlockIndicesType& block_indices)                        = 0;

private:
    std::vector<int> layer_ids_;
    KVCacheSpec      group_spec_;
    BlockCacheV1Ptr  block_cache_;
    BlockPoolPtr     block_pool_;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
