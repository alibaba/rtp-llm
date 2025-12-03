#include "rtp_llm/cpp/cache_new/KVCacheHashUtil.h"

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
        auto& keys = batch_kv_cache_resource->batch_resource[i].cache_keys;
        keys.clear();

        int64_t rolling_hash = 0;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            const int pos       = index * seq_size_per_block;
            const int block_len = std::min(seq_size_per_block, seq_len - pos);
            rolling_hash        = rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
            keys.push_back(rolling_hash);
        }
    }

    batch_kv_cache_resource->last_block_aligned = (seq_len % seq_size_per_block == 0);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        auto&     keys         = batch_kv_cache_resource->batch_resource[i].cache_keys;
        const int total_blocks = seq_len / seq_size_per_block;  // floor, only full blocks

        // If last_block_aligned was false previously, the last cache key corresponds to a partial block.
        // Drop it before we append new full-block cache keys.
        if (!batch_kv_cache_resource->last_block_aligned && !keys.empty()) {
            keys.pop_back();
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? 0 : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int pos = index * seq_size_per_block;
            hash          = rtp_llm::hashInt64Array(hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block);
            keys.push_back(hash);
        }
    }

    // After incremental update we guarantee all existing keys are for full blocks.
    batch_kv_cache_resource->last_block_aligned = true;
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->last_block_aligned) {
        return;
    }
    for (auto& resource : batch_kv_cache_resource->batch_resource) {
        resource.cache_keys.pop_back();
    }
    batch_kv_cache_resource->last_block_aligned = true;
}

}  // namespace rtp_llm
