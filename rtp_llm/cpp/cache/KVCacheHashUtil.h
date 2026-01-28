#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates KVCacheResource::lastBlockAligned for each batch based on seq_len % seq_size_per_block.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// Also updates KVCacheResource::lastBlockAligned for each batch based on current seq_len.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block);

// Drop the last block in cache_keys for batches that are not last_block_aligned
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

// Drop the last block in cache_keys for a single KVCacheResource if it is not last_block_aligned
void dropLastPartialBlock(KVCacheResource& resource);

}  // namespace rtp_llm
