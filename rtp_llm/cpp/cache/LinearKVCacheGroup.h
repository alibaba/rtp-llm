#pragma once

#include <memory>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class LinearKVCacheGroup: public KVCacheGroup {
public:
    LinearKVCacheGroup(const LayerIdsType&          layer_ids,
                       std::shared_ptr<KVCacheSpec> kvcache_spec,
                       BlockPoolPtr                 block_pool,
                       int                          group_id,
                       int                          linear_step = 0,
                       RoleType                     role_type   = RoleType::PDFUSION):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id), role_type_(role_type), linear_step_(linear_step) {}

    MatchResult match(const CacheKeysType& cache_keys) override;
    // Match a single cache key (used by Hybrid allocator to do right-to-left joint matching).
    MatchResult matchSingleKey(CacheKeyType cache_key) const;
    bool        malloc(BlockIndicesType& block_indices, int seq_len, bool enable_reuse_cache = false) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;

    void removeSkippedBlocks(BlockIndicesType& block_indices, bool enable_reuse_cache = false) override;
    void free(const BlockIndicesType& block_indices) override;
    void reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) override;
    int  needBlocksNum(int seq_len, int current_blocks) const override;

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;

private:
    RoleType role_type_{RoleType::PDFUSION};
    // NOTE: linear attention cache can be sparsified; current implementation is conservative:
    // - always keep the last 2 blocks (decode edge case: read block i, write block i+1)
    // - other blocks can be freed (set to NULL_BLOCK_IDX)
    int linear_step_ = 0;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
