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

// Subsequent fill: extend rolling keys for completed blocks. The default still
// drops the tail partial key.
// Callers must preserve the already-keyed prefix (not beam replacement).
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash = 0,
                     bool                    include_partial_block = false);

// Build a one-row remote key snapshot without changing the request's local keys.
CacheKeysType buildDecodePublicationKeys(CompleteTokenIds& tokens, int seq_size_per_block,
                                         size_t computed_end, int64_t initial_hash,
                                         const BatchKVCacheResource* append_only_source);

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
