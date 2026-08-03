#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isActiveTailBlock(int block_idx, int seq_slots, int active_tail_blocks) {
    if (seq_slots <= 0 || block_idx >= seq_slots) {
        return false;
    }
    return block_idx >= std::max(seq_slots - active_tail_blocks, 0);
}

bool shouldAllocateBlock(
    int block_idx, int seq_slots, int reserve_step, int step, bool enable_reuse_cache, int active_tail_blocks) {
    const bool is_reserve = reserve_step > 0 && block_idx >= seq_slots;
    const bool step_hit   = ((block_idx + 1) % step) == 0;
    return is_reserve || isActiveTailBlock(block_idx, seq_slots, active_tail_blocks)
           || (enable_reuse_cache && step_hit);
}

}  // namespace

bool SWAKVCacheGroup::shouldCheckSWATailBlockIds() const {
    return policy().validate_tail_blocks;
}

bool SWAKVCacheGroup::effectiveReuseCacheForAllocation(bool enable_reuse_cache) const {
    return enable_reuse_cache && policy().enable_prefix_reuse;
}

int SWAKVCacheGroup::activeTailBlockCount() const {
    return static_cast<int>(std::max(1u, policy().active_tail_blocks));
}

void SWAKVCacheGroup::checkSWATailBlockIds(const BlockIds& block_ids, const char* caller) const {
    if (!shouldCheckSWATailBlockIds()) {
        return;
    }

    const auto& blocks = block_ids.blocks();
    if (blocks.empty()) {
        return;
    }

    const size_t block_num = blocks.size();
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(blocks[block_num - 1]),
                            "%s invalid SWA block ids: tail block is NULL, block_num=%zu",
                            caller,
                            block_num);
    if (activeTailBlockCount() >= 2 && block_num >= 2) {
        RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(blocks[block_num - 2]),
                                "%s invalid SWA block ids: tail-1 block is NULL, block_num=%zu",
                                caller,
                                block_num);
    }
}

void SWAKVCacheGroup::filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const {
    out.clear();
    out.reserve(in.size());
    for (auto b : in) {
        if (!isNullBlockIdx(b)) {
            out.push_back(b);
        }
    }
}

int SWAKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    const int block_size = seqSizePerBlock();
    return std::max((seq_len + reserve_step + block_size - 1) / block_size - current_blocks, 0);
}

// Conservative upper bound: sliding-window peak usage never exceeds full-attention usage.
int SWAKVCacheGroup::estimatePeakNeedBlocks(int                     seq_len,
                                            const BlockIndicesType& current_block_indices,
                                            int                     remaining_tokens,
                                            int                     reserve_step,
                                            bool                    enable_reuse_cache) const {
    (void)enable_reuse_cache;
    int allocated_blocks = 0;
    for (const auto block_index : current_block_indices) {
        allocated_blocks += !isNullBlockIdx(block_index);
    }
    return std::max(needBlocksNum(seq_len + remaining_tokens, 0, reserve_step) - allocated_blocks, 0);
}

int SWAKVCacheGroup::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                        int  common_seq_len,
                                                        int  remaining_tokens,
                                                        int  reserve_step,
                                                        bool enable_reuse_cache,
                                                        int  target_batch_size) const {
    (void)enable_reuse_cache;
    const int batch_size    = std::max(target_batch_size, 1);
    const int common_blocks = needBlocksNum(common_seq_len, 0);
    const int peak_blocks   = needBlocksNum(seq_len + remaining_tokens, 0, reserve_step);
    return common_blocks + batch_size * std::max(peak_blocks - common_blocks, 0);
}

NeedBlocksInfo SWAKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)common_seq_len;
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(reuse_enabled);
    const int  active_tail_blocks      = activeTailBlockCount();

    NeedBlocksInfo info;

    const int seq_slots   = needBlocksNum(seq_len, 0);
    const int total_slots = needBlocksNum(seq_len, 0, reserve_step);

    info.common_blocks = 0;
    for (int i = reuse_blocks_len; i < seq_slots; ++i) {
        if (shouldAllocateBlock(i, seq_slots, /*reserve_step=*/0, step, effective_reuse_enabled, active_tail_blocks)) {
            ++info.extra_blocks;
        }
    }
    info.extra_blocks += std::max(total_slots - std::max(seq_slots, reuse_blocks_len), 0);

    info.extra_blocks = std::max(info.extra_blocks, 0);
    return info;
}

MatchResult SWAKVCacheGroup::matchSingleKey(CacheKeyType cache_key) const {
    MatchResult result;
    if (!shared_cache_) {
        return result;
    }
    auto block_idx = shared_cache_->matchGroup(cache_key, tag());
    if (!isNullBlockIdx(block_idx)) {
        result.block_indices = {block_idx};
    }
    return result;
}

bool SWAKVCacheGroup::malloc(BlockIds&            block_ids,
                             int                  seq_len,
                             bool                 enable_reuse_cache,
                             int                  reserve_step,
                             std::vector<size_t>* backfilled_positions) {
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(enable_reuse_cache);
    const int  active_tail_blocks      = activeTailBlockCount();
    const int  current_blocks_len      = static_cast<int>(block_ids.blocksNum());
    const int  seq_slots               = needBlocksNum(seq_len, 0, 0);
    const int  new_blocks_len          = needBlocksNum(seq_len, current_blocks_len, reserve_step);

    if (new_blocks_len == 0) {
        checkSWATailBlockIds(block_ids, "SWAKVCacheGroup::malloc");
        return true;
    }

    int need_alloc_blocks = 0;
    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        if (shouldAllocateBlock(i, seq_slots, reserve_step, step, effective_reuse_enabled, active_tail_blocks)) {
            need_alloc_blocks++;
        }
    }

    if (need_alloc_blocks > 0) {
        const auto free_blocks_num = freeBlocksNum();
        if (free_blocks_num < static_cast<size_t>(need_alloc_blocks)) {
            if (!ensureFreeBlocks(need_alloc_blocks)) {
                RTP_LLM_LOG_WARNING("Insufficient free blocks for SWAKVCacheGroup: need %d, have %zu",
                                    need_alloc_blocks,
                                    free_blocks_num);
                return false;
            }
        }
    }

    BlockIndicesType allocated_blocks;
    if (need_alloc_blocks > 0) {
        allocated_blocks = block_pool_->malloc(need_alloc_blocks);
        if (allocated_blocks.size() != static_cast<size_t>(need_alloc_blocks)) {
            if (!allocated_blocks.empty()) {
                block_pool_->requestFree(allocated_blocks);
            }
            return false;
        }
    }

    BlockIndicesType new_ids;
    new_ids.reserve(static_cast<size_t>(new_blocks_len));
    size_t allocated_idx = 0;
    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        const bool should_alloc =
            shouldAllocateBlock(i, seq_slots, reserve_step, step, effective_reuse_enabled, active_tail_blocks);
        if (should_alloc) {
            new_ids.push_back(allocated_blocks[allocated_idx++]);
        } else {
            new_ids.push_back(NULL_BLOCK_IDX);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(allocated_idx == allocated_blocks.size(),
                            "swa kv allocation accounting mismatch, used=%zu allocated=%zu",
                            allocated_idx,
                            allocated_blocks.size());
    block_ids.add(new_ids);
    checkSWATailBlockIds(block_ids, "SWAKVCacheGroup::malloc");
    return true;
}

void SWAKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        checkSWATailBlockIds(block_ids, "SWAKVCacheGroup::removeSkippedBlocks");
        return;
    }
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(enable_reuse_cache);
    const int  active_tail_blocks      = activeTailBlockCount();
    const int  block_size              = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - active_tail_blocks - 1 - reserve_step; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[i])) {
            continue;
        }
        if (effective_reuse_enabled && ((i + 1) % step) == 0) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->requestFree(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
    checkSWATailBlockIds(block_ids, "SWAKVCacheGroup::removeSkippedBlocks");
}

void SWAKVCacheGroup::free(const BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }
    BlockIndicesType valid;
    filterValidBlocks(block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestFree(valid);
    }
}

void SWAKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    BlockIndicesType valid;
    filterValidBlocks(new_block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestReference(valid);
    }
}

}  // namespace rtp_llm
