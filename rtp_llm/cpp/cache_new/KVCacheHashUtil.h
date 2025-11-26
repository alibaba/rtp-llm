#pragma once

#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on seq_len % seq_size_per_block.
void initCacheKeys(BatchKVCacheResource& batch_kv_cache_resource,
                   CompleteTokenIds&     complete_token_ids,
                   int                   seq_size_per_block);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on current seq_len.
void updateCacheKeys(BatchKVCacheResource& batch_kv_cache_resource,
                     CompleteTokenIds&     complete_token_ids,
                     int                   seq_size_per_block);

}  // namespace rtp_llm
