#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    const int  batch_size         = batch_kv_cache_resource->batchSize();
    const int  seq_len            = complete_token_ids->seqLength();
    const bool last_block_aligned = (seq_len % seq_size_per_block == 0);

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
        batch_kv_cache_resource->setLastBlockAligned(i, last_block_aligned);
    }
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
        if (!batch_kv_cache_resource->lastBlockAligned(i) && !keys.empty()) {
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

        // After incremental update we guarantee all existing keys are for full blocks.
        batch_kv_cache_resource->setLastBlockAligned(i, true);
    }
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    for (int i = 0; i < batch_size; ++i) {
        if (!batch_kv_cache_resource->lastBlockAligned(i)) {
            batch_kv_cache_resource->popBackCacheKey(i);
            batch_kv_cache_resource->setLastBlockAligned(i, true);
        }
    }
}

void dropLastPartialBlock(KVCacheResource& resource) {
    if (resource.lastBlockAligned()) {
        return;
    }
    auto& keys = resource.cacheKeys();
    if (!keys.empty()) {
        keys.pop_back();
    }
    resource.setLastBlockAligned(true);
}

}  // namespace rtp_llm
