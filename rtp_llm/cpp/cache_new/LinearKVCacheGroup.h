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
    MatchResult match(CacheKeysType& cache_keys) override;
    void        alloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) override;
    void        insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices, bool is_resident) override;

    void removeSkippedBlocks(BlockIndicesType& block_indices) override;
    void free(const BlockIndicesType& block_indices) override;

private:
    int needBlocksNum(int seq_len, int current_blocks) const override;

private:
    int chunk_size;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
