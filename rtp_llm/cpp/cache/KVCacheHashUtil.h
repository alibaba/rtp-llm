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
// drops the tail partial key; EOS may include it for original local insertion.
// Callers must preserve the already-keyed prefix (not beam replacement).
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash = 0,
                     bool                    include_partial_block = false);

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
