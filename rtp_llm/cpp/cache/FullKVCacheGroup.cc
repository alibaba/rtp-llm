#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
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

bool FullKVCacheGroup::malloc(BlockIndicesType& block_indices, int seq_len, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    int need_blocks_num = needBlocksNum(seq_len, static_cast<int>(block_indices.size()), reserve_step);
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
    block_indices.insert(block_indices.end(), result.begin(), result.end());

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

void FullKVCacheGroup::reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) {
    block_indices.insert(block_indices.end(), new_block_indices.begin(), new_block_indices.end());
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

void FullKVCacheGroup::removeSkippedBlocks(BlockIndicesType& block_indices, bool enable_reuse_cache, int reserve_step) {
}

}  // namespace rtp_llm
