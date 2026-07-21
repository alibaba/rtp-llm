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
    const int total_slots = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step);

    std::vector<size_t> positions_to_backfill;
    const auto&         existing_blocks = block_ids.blocks();
    const size_t        existing_scan   = std::min(existing_blocks.size(), static_cast<size_t>(total_slots));
    for (size_t i = 0; i < existing_scan; ++i) {
        if (isNullBlockIdx(existing_blocks[i])) {
            positions_to_backfill.push_back(i);
        }
    }

    const int new_slots       = std::max(total_slots - static_cast<int>(block_ids.blocksNum()), 0);
    const int need_blocks_num = static_cast<int>(positions_to_backfill.size()) + new_slots;
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

    auto result = block_pool_->malloc(static_cast<size_t>(need_blocks_num));
    if (!result.has_value() || result->size() != static_cast<size_t>(need_blocks_num)) {
        return false;
    }
    block_pool_->incRef(*result, BlockRefType::REQUEST);

    size_t allocated_index = 0;
    for (const size_t position : positions_to_backfill) {
        block_ids.setAt(position, (*result)[allocated_index++]);
    }
    if (backfilled_positions != nullptr) {
        *backfilled_positions = positions_to_backfill;
    }

    BlockIndicesType new_blocks;
    new_blocks.reserve(static_cast<size_t>(new_slots));
    for (int i = 0; i < new_slots; ++i) {
        new_blocks.push_back((*result)[allocated_index++]);
    }
    if (!new_blocks.empty()) {
        block_ids.add(new_blocks);
    }
    RTP_LLM_CHECK_WITH_INFO(allocated_index == result->size(),
                            "full kv allocation accounting mismatch, used=%zu allocated=%zu",
                            allocated_index,
                            result->size());
    return true;
}

MatchResult FullKVCacheGroup::matchPrefix(const CacheKeysType& cache_keys) const {
    (void)cache_keys;
    return {};
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

    BlockIndicesType valid_blocks;
    valid_blocks.reserve(block_indices.size());
    for (const BlockIdxType block : block_indices) {
        if (!isNullBlockIdx(block)) {
            valid_blocks.push_back(block);
        }
    }
    if (!valid_blocks.empty()) {
        block_pool_->decRef(valid_blocks, BlockRefType::REQUEST);
    }
    RTP_LLM_LOG_DEBUG("Freed %zu blocks", valid_blocks.size());
}

void FullKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    BlockIndicesType valid_blocks;
    valid_blocks.reserve(new_block_indices.size());
    for (const BlockIdxType block : new_block_indices) {
        if (!isNullBlockIdx(block)) {
            valid_blocks.push_back(block);
        }
    }
    if (!valid_blocks.empty()) {
        block_pool_->incRef(valid_blocks, BlockRefType::REQUEST);
    }
}

void FullKVCacheGroup::removeSkippedBlocks(BlockIds& /*block_ids*/, bool /*enable_reuse_cache*/, int /*reserve_step*/) {
}

}  // namespace rtp_llm
