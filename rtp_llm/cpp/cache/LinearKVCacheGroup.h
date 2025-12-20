#pragma once

#include <memory>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class LinearKVCacheGroup: public KVCacheGroup {
public:
    LinearKVCacheGroup(const LayerIdsType&          layer_ids,
                       std::shared_ptr<KVCacheSpec> kvcache_spec,
                       BlockPoolPtr                 block_pool,
                       int                          group_id):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id) {}

    MatchResult match(const CacheKeysType& cache_keys) override;
    bool        malloc(BlockIndicesType& block_indices, int seq_len) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;

    void removeSkippedBlocks(BlockIndicesType& block_indices) override;
    void free(const BlockIndicesType& block_indices) override;
    void reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) override;

private:
    int needBlocksNum(int seq_len, int current_blocks) const override;

private:
    int  chunk_blocks;
    int  save_point;
    bool reuse_cache;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
