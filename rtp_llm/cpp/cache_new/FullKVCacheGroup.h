#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(const LayerIdsType&          layer_ids,
                     std::shared_ptr<KVCacheSpec> kvcache_spec,
                     BlockPoolPtr                 block_pool,
                     int                          group_id):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id) {}

    bool        malloc(const CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) override;
    MatchResult match(const CacheKeysType& cache_keys) override;
    void        free(const BlockIndicesType& block_indices) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;
    void removeSkippedBlocks(BlockIndicesType& block_indices) override;
    int  needBlocksNum(int seq_len, int current_blocks) const override;
    void reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) override;

private:
};

}  // namespace rtp_llm
