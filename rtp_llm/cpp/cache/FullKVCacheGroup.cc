#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int FullKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    const int block_size = seqSizePerBlock();
    return std::max((seq_len + reserve_step + block_size - 1) / block_size - current_blocks, 0);
}

int FullKVCacheGroup::estimatePeakNeedBlocks(int                     seq_len,
                                             const BlockIndicesType& current_block_indices,
                                             int                     remaining_tokens,
                                             int                     reserve_step,
                                             bool                    enable_reuse_cache) const {
    (void)enable_reuse_cache;
    const int current_blocks = static_cast<int>(current_block_indices.size());
    return std::max(
        (seq_len + remaining_tokens + reserve_step + seqSizePerBlock() - 1) / seqSizePerBlock() - current_blocks, 0);
}

int FullKVCacheGroup::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                         int  common_seq_len,
                                                         int  remaining_tokens,
                                                         int  reserve_step,
                                                         bool enable_reuse_cache,
                                                         int  target_batch_size) const {
    (void)enable_reuse_cache;
    const int batch_size    = std::max(target_batch_size, 1);
    const int common_blocks = (common_seq_len + seqSizePerBlock() - 1) / seqSizePerBlock();
    const int peak_blocks   = (seq_len + remaining_tokens + reserve_step + seqSizePerBlock() - 1) / seqSizePerBlock();
    return common_blocks + batch_size * std::max(peak_blocks - common_blocks, 0);
}

NeedBlocksInfo FullKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    NeedBlocksInfo info;
    const int      common_slots        = needBlocksNum(common_seq_len, /*current_blocks=*/0);
    const int      total_slots         = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step);
    const int      reused_common_slots = reuse_enabled ? std::min(std::max(reuse_blocks_len, 0), common_slots) : 0;
    info.common_blocks                 = std::max(common_slots - reused_common_slots, 0);
    info.extra_blocks                  = std::max(total_slots - common_slots, 0);
    return info;
}

bool FullKVCacheGroup::malloc(BlockIds&            block_ids,
                              int                  seq_len,
                              bool                 enable_reuse_cache,
                              int                  reserve_step,
                              std::vector<size_t>* backfilled_positions) {
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
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

MatchResult FullKVCacheGroup::matchPrefix(const CacheKeysType& cache_keys) const {
    MatchResult final_result;

    if (!shared_cache_) {
        return final_result;
    }

    for (const auto& cache_key : cache_keys) {
        auto block_idx = shared_cache_->matchGroup(cache_key, tag());
        if (isNullBlockIdx(block_idx)) {
            break;
        }
        final_result.reuse_blocks++;
        final_result.block_indices.push_back(block_idx);
    }

    final_result.reuse_length = final_result.reuse_blocks * seqSizePerBlock();

    return final_result;
}

void FullKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                       const BlockIndicesType& block_indices,
                                       bool                    is_resident) {
    KVCacheGroup::insertIntoCache(cache_keys, block_indices, is_resident);
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

void FullKVCacheGroup::removeSkippedBlocks(BlockIds& /*block_ids*/, bool /*enable_reuse_cache*/, int /*reserve_step*/) {
}

}  // namespace rtp_llm
