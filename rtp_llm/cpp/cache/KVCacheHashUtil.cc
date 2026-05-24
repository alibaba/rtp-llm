#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>
#include <cstdint>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

namespace {

// Apply salt fields on top of the unsalted Jenkins roll (M03 §3.1).
// A default-initialised (all-zero) salt makes this a no-op, preserving
// today's legacy hash bytes exactly.
inline int64_t applySalt(int64_t h, const CacheKeySalt& s) {
    uint64_t u = static_cast<uint64_t>(h);
    u ^= s.model_id;
    u ^= static_cast<uint64_t>(s.dtype_id) << 8;
    u ^= static_cast<uint64_t>(s.lora_id) << 16;
    u ^= static_cast<uint64_t>(s.K_state) << 24;
    u ^= static_cast<uint64_t>(s.Tlog) << 32;
    return static_cast<int64_t>(u);
}

// Rolling hash with salt: Jenkins-64 over the token chunk, then XOR-fold
// the salt fields. This is the single source of truth for cache_key bytes.
inline int64_t rollHash(int64_t prev, int32_t* begin, int32_t* end, const CacheKeySalt& s) {
    int64_t h = rtp_llm::hashInt64Array(prev, begin, end);
    return applySalt(h, s);
}

}  // namespace

uint32_t nonzeroFieldBitmap(const CacheKeySalt& salt) {
    uint32_t bitmap = 0;
    if (salt.model_id != 0) bitmap |= 1u << 0;
    if (salt.dtype_id != 0) bitmap |= 1u << 1;
    if (salt.lora_id  != 0) bitmap |= 1u << 2;
    if (salt.K_state  != 0) bitmap |= 1u << 3;
    if (salt.Tlog     != 0) bitmap |= 1u << 4;
    return bitmap;
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   const CacheKeySalt&     salt) {
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
            rolling_hash        = rollHash(rolling_hash, token_ids + pos, token_ids + pos + block_len, salt);
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    // Legacy unsalted entry: zero-initialised salt -> XOR with 0 -> identity.
    static const CacheKeySalt kZeroSalt{};
    initCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, kZeroSalt);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     const CacheKeySalt&     salt) {
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
            hash          = rollHash(hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block, salt);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    // After incremental update we guarantee all existing keys are for full blocks.
    batch_kv_cache_resource->setLastBlockAligned(true);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block) {
    // Legacy unsalted entry: zero-initialised salt -> XOR with 0 -> identity.
    static const CacheKeySalt kZeroSalt{};
    updateCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, kZeroSalt);
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
