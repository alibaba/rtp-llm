#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on seq_len % seq_size_per_block.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   int64_t                 initial_hash = 0);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on current seq_len.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash = 0);

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

// Cumulative key at an exact token endpoint, including image identity after each base-block
// segment. For a canonical CP tail, pass the base block size rather than the CP reuse unit.
int64_t cacheKeyAtTokenEnd(CompleteTokenIds& complete_token_ids,
                           int               batch_id,
                           int               seq_size_per_block,
                           int               token_end,
                           int64_t           initial_hash = 0);

// Reuse the preceding complete base key and hash only the final partial segment.
int64_t cacheKeyAtTokenEnd(CompleteTokenIds&    complete_token_ids,
                           int                  batch_id,
                           int                  seq_size_per_block,
                           int                  token_end,
                           const CacheKeysType& base_keys,
                           int64_t              initial_hash = 0);

// Verify a cached segment even when its suffix extends beyond the request's
// common prefix. Preserve the request's image identity and base-block hashing.
int64_t cacheKeyForSourceSegment(const CompleteTokenIds& complete_token_ids,
                                 const std::vector<int>& source_tokens,
                                 int                     segment_begin,
                                 int                     seq_size_per_block,
                                 int64_t                 initial_hash);

// Select the segment's seed from complete base keys, or the configured initial seed.
int64_t cacheKeyForSourceSegment(const CompleteTokenIds& complete_token_ids,
                                 const std::vector<int>& source_tokens,
                                 int                     segment_begin,
                                 int                     seq_size_per_block,
                                 const CacheKeysType&    base_keys,
                                 int64_t                 initial_hash);

}  // namespace rtp_llm
