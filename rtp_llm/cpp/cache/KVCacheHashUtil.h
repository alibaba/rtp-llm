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

// Subsequent fill: replace the previous partial key, append newly completed base blocks,
// and include the current partial key for runtime/P->D consumers. Cache publication
// uses last_block_aligned to exclude a partial key from the complete-key protocol.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash = 0);

// Drop the last partial block in cache_keys before complete-only publication.
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

// Cumulative key at an exact token endpoint, including image identity after each base-block
// segment. For a canonical CP tail, pass the base block size rather than the CP reuse unit.
int64_t cacheKeyAtTokenEnd(CompleteTokenIds&       complete_token_ids,
                           int                     batch_id,
                           int                     seq_size_per_block,
                           int                     token_end,
                           int64_t                 initial_hash = 0);

// Verify a cached segment even when its suffix extends beyond the request's
// common prefix. Preserve the request's image identity and base-block hashing.
int64_t cacheKeyForSourceSegment(const CompleteTokenIds& complete_token_ids,
                                 const std::vector<int>& source_tokens,
                                 int                     segment_begin,
                                 int                     seq_size_per_block,
                                 int64_t                 initial_hash);

}  // namespace rtp_llm
