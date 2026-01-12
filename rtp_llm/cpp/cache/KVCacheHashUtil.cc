#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    // Initial fill: compute cache_keys for all blocks, including the final partial block.
    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->clearCacheKeys(i);

        int64_t rolling_hash = 0;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            const int pos       = index * seq_size_per_block;
            const int block_len = std::min(seq_size_per_block, seq_len - pos);
            rolling_hash        = rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->last_block_aligned = (seq_len % seq_size_per_block == 0);
    RTP_LLM_LOG_INFO(
        "initCacheKeys finished, batch_kv_cache_resource: %p, last_block_aligned: %d, cache_keys size: %ld",
        batch_kv_cache_resource.get(),
        batch_kv_cache_resource->last_block_aligned,
        batch_kv_cache_resource->cacheKeys().size());
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        const auto& keys         = batch_kv_cache_resource->cacheKeys(i);
        const int   total_blocks = seq_len / seq_size_per_block;  // floor, only full blocks

        // If last_block_aligned was false previously, the last cache key corresponds to a partial block.
        // Drop it before we append new full-block cache keys.
        if (!batch_kv_cache_resource->last_block_aligned && !keys.empty()) {
            batch_kv_cache_resource->popBackCacheKey(i);
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? 0 : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int pos = index * seq_size_per_block;
            hash          = rtp_llm::hashInt64Array(hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    // After incremental update we guarantee all existing keys are for full blocks.
    batch_kv_cache_resource->last_block_aligned = true;
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->last_block_aligned) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->last_block_aligned = true;
}

}  // namespace rtp_llm
