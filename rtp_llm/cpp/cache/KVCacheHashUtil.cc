#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace {

void appendCacheKeys(CacheKeysType& keys, CompleteTokenIds& token_ids, int row,
                     int seq_size_per_block, int seq_len, int total_blocks, int64_t initial_hash) {
    const auto* tokens = token_ids.data(row);
    auto hash = keys.empty() ? initial_hash : keys.back();
    for (int index = static_cast<int>(keys.size()); index < total_blocks; ++index) {
        const int pos = index * seq_size_per_block;
        const int block_len = std::min(seq_size_per_block, seq_len - pos);
        hash = hashInt64Array(hash, tokens + pos, tokens + pos + block_len);
        const auto identity = token_ids.multimodalCacheIdentity(pos, block_len);
        if (!identity.empty()) {
            hash = hashInt64Array(hash, identity.data(), identity.data() + identity.size());
        }
        keys.push_back(hash);
    }
}

}  // namespace

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

        appendCacheKeys(batch_kv_cache_resource->cacheResource(i).cacheKeys(), *complete_token_ids, i,
                        seq_size_per_block, seq_len, desired_blocks, initial_hash);
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
                     int64_t                 initial_hash,
                     bool                    include_partial_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    const int full_blocks = seq_len / seq_size_per_block;
    const int total_blocks = full_blocks + (include_partial_block && seq_len % seq_size_per_block != 0);
    if (include_partial_block) {
        for (int i = 0; i < batch_size; ++i) {
            const auto& keys = batch_kv_cache_resource->cacheKeys(i);
            const size_t full_keys = keys.size() - (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty());
            if (full_keys > static_cast<size_t>(full_blocks)) {
                // A shortened history invalidates the old key suffix.
                initCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, initial_hash);
                return;
            }
        }
    }
    for (int i = 0; i < batch_size; ++i) {
        auto& keys = batch_kv_cache_resource->cacheResource(i).cacheKeys();
        // The old partial key cannot seed a new complete block.
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
            keys.pop_back();
        }
        appendCacheKeys(keys, *complete_token_ids, i, seq_size_per_block, seq_len, total_blocks, initial_hash);
    }

    // Default callers retain the original full-block-only contract.
    batch_kv_cache_resource->setLastBlockAligned(!include_partial_block || seq_len % seq_size_per_block == 0);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
}

CacheKeysType buildDecodePublicationKeys(CompleteTokenIds& tokens, int seq_size_per_block,
                                         size_t computed_end, int64_t initial_hash,
                                         const BatchKVCacheResource* append_only_source) {
    const int seq_len = tokens.seqLength();
    const int blocks = static_cast<int>(std::min(computed_end, static_cast<size_t>(seq_len))) / seq_size_per_block;
    CacheKeysType keys;
    keys.reserve(blocks);
    if (append_only_source && append_only_source->cacheKeysInitialized()) {
        const auto& source = append_only_source->cacheKeys(0);
        const auto complete = source.size() - (!append_only_source->lastBlockAligned() && !source.empty());
        if (complete <= static_cast<size_t>(seq_len / seq_size_per_block)) {
            keys.assign(source.begin(), source.begin() + std::min(complete, static_cast<size_t>(blocks)));
        }
    }
    appendCacheKeys(keys, tokens, 0, seq_size_per_block, seq_len, blocks, initial_hash);
    return keys;
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
