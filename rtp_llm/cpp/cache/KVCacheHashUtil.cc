#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   bool                    mtp_lookahead) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    // Initial fill: compute cache_keys for all blocks, including the final partial block.
    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->clearCacheKeys(i);

        int64_t rolling_hash = mtp_lookahead ? kMtpCacheKeySeed : 0;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            // The previous MTP key already includes this block's first token.
            const int begin = index * seq_size_per_block + (mtp_lookahead && index > 0 ? 1 : 0);
            const int end   = std::min((index + 1) * seq_size_per_block + (mtp_lookahead ? 1 : 0), seq_len);
            rolling_hash    = rtp_llm::hashInt64Array(rolling_hash, token_ids + begin, token_ids + end);
            if (mtp_lookahead && (index + 1) * seq_size_per_block >= seq_len) {
                // In particular, L=B+1 leaves no unhashed tokens for the tail.
                // It must not alias the preceding stable key in PD maps or the
                // prefix tree. This key is never published for prefix reuse.
                int32_t provisional_domain[] = {0x4d545050, seq_len};  // MTPP
                rolling_hash = rtp_llm::hashInt64Array(rolling_hash, provisional_domain, provisional_domain + 2);
            }
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len == 0 || (!mtp_lookahead && seq_len % seq_size_per_block == 0));
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     bool                    mtp_lookahead) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        const auto& keys         = batch_kv_cache_resource->cacheKeys(i);
        const int   total_blocks = std::max(0, seq_len - (mtp_lookahead ? 1 : 0)) / seq_size_per_block;

        // Remove the partial/provisional key before extending the stable chain.
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
            batch_kv_cache_resource->popBackCacheKey(i);
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? (mtp_lookahead ? kMtpCacheKeySeed : 0) : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int begin = index * seq_size_per_block + (mtp_lookahead && index > 0 ? 1 : 0);
            const int end   = (index + 1) * seq_size_per_block + (mtp_lookahead ? 1 : 0);
            hash            = rtp_llm::hashInt64Array(hash, token_ids + begin, token_ids + end);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    // Every retained key has all dependencies needed for prefix reuse.
    batch_kv_cache_resource->setLastBlockAligned(true);
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
