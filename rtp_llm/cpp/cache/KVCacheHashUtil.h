#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on seq_len % seq_size_per_block.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids);
inline void
initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids, int) {
    initCacheKeys(std::move(batch_kv_cache_resource), std::move(complete_token_ids));
}

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on current seq_len.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids);
inline void
updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids, int) {
    updateCacheKeys(std::move(batch_kv_cache_resource), std::move(complete_token_ids));
}

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
