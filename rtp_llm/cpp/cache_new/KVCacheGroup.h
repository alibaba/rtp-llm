#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/KVCacheSpec.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

// first call:
// match, alloc, insert to cache

// other call:
// insert to cache, removeSkippedBlocks, alloc

// query finish:
// free

class KVCacheGroup {
public:
    virtual MatchResult match(CacheKeysType& cache_keys)                                                 = 0;
    virtual void        alloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int token_len) = 0;
    virtual void        insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices)      = 0;

    virtual void removeSkippedBlocks(BlockIndicesType& block_indices) = 0;
    virtual void free(BlockIndicesType& block_indices)                = 0;

    // add loss, is_resident for insertIntoCache
    virtual KVCacheType                       type() const                                               = 0;
    virtual std::map<BlockIdxType, BufferPtr> blockBuffer(BlockIdxType block_id, CacheKeyType cache_key) = 0;

private:
    std::vector<int> layer_ids_;
    KVCacheSpec      group_spec_;
    BlockCacheV1Ptr  block_cache_;
    BlockPoolPtr     block_pool_;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
