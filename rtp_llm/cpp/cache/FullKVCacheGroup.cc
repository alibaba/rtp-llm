#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int FullKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    return std::max((seq_len + reserve_step + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

NeedBlocksInfo FullKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)reuse_blocks_len;
    (void)reuse_enabled;
    NeedBlocksInfo info;
    const int      common_slots = needBlocksNum(common_seq_len, /*current_blocks=*/0);
    const int      total_slots  = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step);
    info.common_blocks          = std::max(common_slots, 0);
    info.extra_blocks           = std::max(total_slots - common_slots, 0);
    return info;
}

bool FullKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    int need_blocks_num = needBlocksNum(seq_len, static_cast<int>(block_ids.blocksNum()), reserve_step);
    if (need_blocks_num == 0) {
        return true;
    }
    auto free_blocks_num = freeBlocksNum();
    if (free_blocks_num < need_blocks_num) {
        if (!ensureFreeBlocks(need_blocks_num)) {
            RTP_LLM_LOG_WARNING(
                "Insufficient free blocks for common part: need %d, have %zu", need_blocks_num, free_blocks_num);
            return false;
        }
    }

    auto result = block_pool_->malloc(need_blocks_num);
    if (result.empty()) {
        return false;
    }
    block_ids.add(result);
    return true;
}

MatchResult FullKVCacheGroup::match(const CacheKeysType& cache_keys) {
    MatchResult final_result;

    for (const auto& cache_key : cache_keys) {
        auto result = block_cache_->match(cache_key, group_id_);
        if (isNullBlockIdx(result.matched_index)) {
            break;
        }
        final_result.reuse_blocks++;
        final_result.block_indices.push_back(result.matched_index);
    }

    final_result.reuse_length = final_result.reuse_blocks * seqSizePerBlock();

    return final_result;
}

void FullKVCacheGroup::free(const BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }

    block_pool_->requestFree(block_indices);
    RTP_LLM_LOG_DEBUG("Freed %zu blocks", block_indices.size());
}

void FullKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    block_pool_->requestReference(new_block_indices);
}

void FullKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                       const BlockIndicesType& block_indices,
                                       bool                    is_resident) {
    if (cache_keys.empty()) {
        return;
    }

    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_ERROR(
            "Cache keys size (%zu) doesn't match block indices size (%zu)", cache_keys.size(), block_indices.size());
        return;
    }

    const int last_index = cache_keys.size() - 1;
    for (int i = last_index; i >= 0; --i) {
        BlockCache::CacheItem item;
        item.cache_key   = cache_keys[i];
        item.group_id    = group_id_;
        item.block_index = block_indices[i];
        item.is_resident = is_resident;
        if (block_cache_->put(item)) {
            block_pool_->blockCacheReference(block_indices[i]);
        }
    }

    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}

void FullKVCacheGroup::removeSkippedBlocks(BlockIds& /*block_ids*/, bool /*enable_reuse_cache*/, int /*reserve_step*/) {
}

void FullKVCacheGroup::insertPartialTailForBatch(const InsertInfo& insert_info,
                                                 int               batch_id,
                                                 bool              is_linear_attention) {
    auto res = insert_info.batch_kv_cache_resource;
    if (!res || !insert_info.complete_token_ids) {
        return;
    }
    if (res->lastBlockAligned()) {
        return;
    }
    const auto& keys = res->cacheKeys(batch_id);
    const auto& blk  = res->blocks(batch_id, group_id_);
    if (keys.empty() || blk.empty()) {
        return;
    }
    const int last = static_cast<int>(keys.size()) - 1;
    if (isNullBlockIdx(blk[static_cast<size_t>(last)])) {
        return;
    }

    const CacheKeyType parent_key = last > 0 ? keys[static_cast<size_t>(last - 1)] : static_cast<CacheKeyType>(0);
    const int          seq_len    = insert_info.complete_token_ids->seqLength();
    const int          B          = seq_size_per_block_;
    const int          L          = seq_len - last * B;
    if (L <= 0 || L > B) {
        return;
    }

    const int32_t* tok = insert_info.complete_token_ids->data(batch_id);
    const int      off = last * B;

    BlockCache::CacheItem item;
    item.cache_key        = keys[static_cast<size_t>(last)];
    item.parent_block_key = parent_key;
    item.group_id         = group_id_;
    item.block_index      = blk[static_cast<size_t>(last)];
    item.is_resident      = insert_info.is_resident;
    item.valid_token_len  = L;
    item.prefix_tokens.assign(tok + off, tok + off + L);
    item.is_linear_group = is_linear_attention;

    if (block_cache_->put(item)) {
        block_pool_->blockCacheReference(blk[static_cast<size_t>(last)]);
    }
}

}  // namespace rtp_llm
