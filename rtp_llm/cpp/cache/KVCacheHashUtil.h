#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// MTP keys include one lookahead token per block. The final key is provisional
// until that token is known, even for an exactly aligned input. Keep it for PD
// page indexing, but mark last_block_aligned=false so it cannot be published.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   bool                    mtp_lookahead = false);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// With MTP, only blocks with a known lookahead are retained. All returned keys
// are publishable, so last_block_aligned is true even if tokens have a tail.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     bool                    mtp_lookahead = false);

// Drop an incomplete/provisional last key before publishing reusable blocks.
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
