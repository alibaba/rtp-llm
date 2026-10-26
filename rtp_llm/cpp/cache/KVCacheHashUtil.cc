#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

namespace {
int64_t appendCacheKeySegment(int64_t hash, const CompleteTokenIds& ids, const int32_t* tokens, int pos, int len) {
    hash = hashInt64Array(hash, tokens + pos, tokens + pos + len);
    const auto identity = ids.imageCacheIdentity(pos, len);
    if (!identity.empty()) {
        hash = hashInt64Array(hash, identity.data(), identity.data() + identity.size());
    }
    return hash;
}
}  // namespace

int64_t cacheKeyForSourceSegment(const CompleteTokenIds& complete_token_ids,
                                 const std::vector<int>& source_tokens,
                                 int                     segment_begin,
                                 int                     seq_size_per_block,
                                 int64_t                 initial_hash) {
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0 && segment_begin >= 0 && segment_begin % seq_size_per_block == 0,
                            "cached segment must begin at a base-block boundary");
    int64_t hash = initial_hash;
    for (size_t offset = 0; offset < source_tokens.size(); offset += seq_size_per_block) {
        const size_t count = std::min(static_cast<size_t>(seq_size_per_block), source_tokens.size() - offset);
        hash               = hashInt64Array(hash, source_tokens.data() + offset, source_tokens.data() + offset + count);
        const auto identity =
            complete_token_ids.imageCacheIdentity(segment_begin + static_cast<int>(offset), static_cast<int>(count));
        if (!identity.empty()) {
            hash = hashInt64Array(hash, identity.data(), identity.data() + identity.size());
        }
    }
    return hash;
}

int64_t cacheKeyAtTokenEnd(CompleteTokenIds&       complete_token_ids,
                           int                     batch_id,
                           int                     seq_size_per_block,
                           int                     token_end,
                           int64_t                 initial_hash) {
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0 && token_end >= 0 && token_end <= complete_token_ids.seqLength(),
                            "invalid cache endpoint=%d block_size=%d seq_length=%d",
                            token_end,
                            seq_size_per_block,
                            complete_token_ids.seqLength());
    RTP_LLM_CHECK_WITH_INFO(batch_id >= 0 && batch_id < complete_token_ids.batchSize(),
                            "invalid cache batch id=%d", batch_id);
    const int32_t* tokens = complete_token_ids.data(batch_id);
    int64_t        hash   = initial_hash;
    for (int pos = 0; pos < token_end; pos += seq_size_per_block) {
        hash = appendCacheKeySegment(
            hash, complete_token_ids, tokens, pos, std::min(seq_size_per_block, token_end - pos));
    }
    return hash;
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   int64_t                 initial_hash) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    // Initial fill: compute cache_keys for all blocks, including the final partial block.
    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->clearCacheKeys(i);

        int64_t rolling_hash = initial_hash;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            const int pos       = index * seq_size_per_block;
            const int block_len = std::min(seq_size_per_block, seq_len - pos);
            rolling_hash = appendCacheKeySegment(rolling_hash, *complete_token_ids, token_ids, pos, block_len);
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        const auto& keys         = batch_kv_cache_resource->cacheKeys(i);
        const int   total_blocks = seq_len / seq_size_per_block;

        // If last_block_aligned was false previously, the last cache key corresponds to a partial block.
        // Drop it before we append new full-block cache keys.
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
            batch_kv_cache_resource->popBackCacheKey(i);
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? initial_hash : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int pos = index * seq_size_per_block;
            hash = appendCacheKeySegment(hash, *complete_token_ids, token_ids, pos, seq_size_per_block);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
        const int tail_length = seq_len % seq_size_per_block;
        if (tail_length != 0) {
            hash = appendCacheKeySegment(
                hash, *complete_token_ids, token_ids, total_blocks * seq_size_per_block, tail_length);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
