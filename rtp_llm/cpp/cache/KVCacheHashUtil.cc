#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

CacheKeysType calculateCacheKeys(const int* token_ids, size_t token_count, int seq_size_per_block) {
    CacheKeysType cache_keys;
    if (!token_ids || seq_size_per_block <= 0) {
        return cache_keys;
    }

    const size_t block_size  = seq_size_per_block;
    const size_t block_count = token_count / block_size + (token_count % block_size != 0);

    int64_t rolling_hash = 0;
    for (size_t block_index = 0; block_index < block_count; ++block_index) {
        const size_t pos       = block_index * block_size;
        const size_t block_len = std::min(block_size, token_count - pos);
        rolling_hash          = hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
        cache_keys.push_back(rolling_hash);
    }
    return cache_keys;
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        auto cache_keys = calculateCacheKeys(complete_token_ids->data(i), seq_len, seq_size_per_block);
        batch_kv_cache_resource->cacheResource(i).setCacheKeys(std::move(cache_keys));
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
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
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
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
    batch_kv_cache_resource->setLastBlockAligned(true);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
