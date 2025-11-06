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
    LinearKVCacheGroup(const LayerIdsType& layer_ids, std::shared_ptr<KVCacheSpec> group_spec, BlockPoolPtr block_pool):
        KVCacheGroup(layer_ids, group_spec, block_pool) {}

    MatchResult match(const CacheKeysType& cache_keys) override;
    bool        malloc(const CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;

    void removeSkippedBlocks(BlockIndicesType& block_indices) override;
    void free(const BlockIndicesType& block_indices) override;

private:
    int needBlocksNum(int seq_len, int current_blocks) const override;

private:
    int save_point;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
