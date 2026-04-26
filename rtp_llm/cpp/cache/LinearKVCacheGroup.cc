#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

#include <algorithm>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void LinearKVCacheGroup::filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const {
    out.clear();
    out.reserve(in.size());
    for (auto b : in) {
        if (!isNullBlockIdx(b)) {
            out.push_back(b);
        }
    }
}

int LinearKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    int extra_blocks = reserve_step ? reserve_step - 1 : 0;
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ + extra_blocks - current_blocks, 0);
}

bool LinearKVCacheGroup::shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const {
    if (pos < 0) {
        return false;
    }

    const int  step        = std::max(1, linear_step_);
    const int  seq_slots   = needBlocksNum(seq_len, 0, 0);
    const int  total_slots = needBlocksNum(seq_len, 0, reserve_step);
    const bool is_seq_tail = (seq_slots > 0) && (pos >= std::max(0, seq_slots - 2)) && (pos < seq_slots);
    const bool is_reserve  = (reserve_step > 0) && (pos >= seq_slots) && (pos < total_slots);
    const bool step_hit    = (((pos + 1) % step) == 0);
    return is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
}

NeedBlocksInfo LinearKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    NeedBlocksInfo info;

    // common_slots: blocks for common_seq_len (no reserve)
    const int common_slots = needBlocksNum(common_seq_len, 0);
    // total_slots includes reserve_step - 1 extra linear slots when reserve_step is non-zero.
    const int total_slots = needBlocksNum(seq_len, 0, reserve_step);

    auto common_required = [&](int pos) { return shouldMaterializeBlock(pos, common_seq_len, 0, reuse_enabled); };
    auto final_required  = [&](int pos) { return shouldMaterializeBlock(pos, seq_len, reserve_step, reuse_enabled); };

    for (int pos = 0; pos < common_slots; ++pos) {
        if (common_required(pos)) {
            info.common_blocks++;
        }
    }
    for (int pos = 0; pos < total_slots; ++pos) {
        if (final_required(pos) && !(pos < common_slots && common_required(pos))) {
            info.extra_blocks++;
        }
    }

    // Linear reuse materializes only one prefix block: the matched tail at
    // reuse_blocks_len - 1. Do not count that block as newly allocated.
    const int reused_tail_pos = (reuse_enabled && reuse_blocks_len > 0) ? reuse_blocks_len - 1 : -1;
    if (reused_tail_pos >= 0) {
        if (reused_tail_pos < common_slots && common_required(reused_tail_pos)) {
            info.common_blocks--;
        } else if (reused_tail_pos < total_slots && final_required(reused_tail_pos)) {
            info.extra_blocks--;
        }
    }

    info.common_blocks = std::max(info.common_blocks, 0);
    info.extra_blocks  = std::max(info.extra_blocks, 0);
    return info;
}

MatchResult LinearKVCacheGroup::matchSingleKey(CacheKeyType cache_key) const {
    MatchResult result;
    auto        matched = block_cache_->match(cache_key, group_id_);
    if (!isNullBlockIdx(matched.matched_index)) {
        result.block_indices = {matched.matched_index};
    }
    return result;
}

MatchResult LinearKVCacheGroup::match(const CacheKeysType& cache_keys) {
    (void)cache_keys;
    RTP_LLM_CHECK_WITH_INFO(false, "SWA should not call match, use matchSingleKey instead");
    return {};
}

bool LinearKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    const int step               = std::max(1, linear_step_);
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int seq_slots          = needBlocksNum(seq_len, 0, 0);
    const int total_slots        = needBlocksNum(seq_len, 0, reserve_step);
    const int new_blocks_len     = std::max(total_slots - current_blocks_len, 0);

    auto should_materialize = [&](int pos) {
        // Materialize tail and tail-1: causal_conv1d_update may read
        // (seq_len - 2) / SBP when seq_len crosses a block boundary.
        // Leaving tail-1 NULL can hit IMA on long prompts.
        const bool is_seq_tail = (seq_slots > 0) && (pos >= std::max(0, seq_slots - 2)) && (pos < seq_slots);
        const bool is_reserve  = (reserve_step > 0) && (pos >= seq_slots) && (pos < total_slots);
        const bool step_hit    = (((pos + 1) % step) == 0);
        return is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
    };

    std::vector<size_t> positions_to_backfill;
    const auto&         existing_blocks = block_ids.blocks();
    const int           existing_scan   = std::min(current_blocks_len, total_slots);
    for (int i = 0; i < existing_scan; ++i) {
        if (should_materialize(i) && isNullBlockIdx(existing_blocks[static_cast<size_t>(i)])) {
            positions_to_backfill.push_back(static_cast<size_t>(i));
        }
    }

    int need_alloc_blocks = 0;
    need_alloc_blocks += static_cast<int>(positions_to_backfill.size());
    for (int i = current_blocks_len; i < total_slots; i++) {
        if (should_materialize(i)) {
            need_alloc_blocks++;
        }
    }

    if (need_alloc_blocks > 0) {
        const auto free_blocks_num = freeBlocksNum();
        if (free_blocks_num < static_cast<size_t>(need_alloc_blocks)) {
            if (!ensureFreeBlocks(need_alloc_blocks)) {
                RTP_LLM_LOG_WARNING("Insufficient free blocks for LinearKVCacheGroup: need %d, have %zu",
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
            return false;
        }
    }

    size_t allocated_idx = 0;
    for (size_t pos : positions_to_backfill) {
        block_ids.setAt(pos, allocated_blocks[allocated_idx++]);
    }

    BlockIndicesType new_ids;
    new_ids.reserve(static_cast<size_t>(new_blocks_len));
    for (int i = current_blocks_len; i < total_slots; i++) {
        if (should_materialize(i)) {
            new_ids.push_back(allocated_blocks[allocated_idx++]);
        } else {
            new_ids.push_back(NULL_BLOCK_IDX);
        }
    }
    if (!new_ids.empty()) {
        block_ids.add(new_ids);
    }
    RTP_LLM_CHECK_WITH_INFO(allocated_idx == allocated_blocks.size(),
                            "linear kv allocation accounting mismatch, used=%zu allocated=%zu",
                            allocated_idx,
                            allocated_blocks.size());
    return true;
}

void LinearKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                         const BlockIndicesType& block_indices,
                                         bool                    is_resident) {
    if (cache_keys.empty() || block_indices.empty()) {
        return;
    }
    const size_t n = std::min(cache_keys.size(), block_indices.size());
    for (size_t i = 0; i < n; ++i) {
        const auto b = block_indices[i];
        if (isNullBlockIdx(b)) {
            continue;
        }
        BlockCache::CacheItem item;
        item.cache_key   = cache_keys[i];
        item.group_id    = group_id_;
        item.block_index = b;
        item.is_resident = is_resident;
        if (block_cache_->put(item)) {
            block_pool_->blockCacheReference(b);
        }
    }
}

void LinearKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    const auto& block_indices = block_ids.blocks();  // const view for reading current state
    if (block_indices.empty()) {
        return;
    }
    const int step       = std::max(1, linear_step_);
    const int block_size = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    // keep last 2 and every reserve_step
    for (int i = block_size - 3 - reserve_step; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[i])) {
            continue;
        }
        if (enable_reuse_cache && ((i + 1) % step) == 0) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->requestFree(blocks_to_free);
        block_ids.remove(pos_to_remove);  // null-out by position, updates kernel slots incrementally
    }
}

void LinearKVCacheGroup::free(const BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }
    BlockIndicesType valid;
    filterValidBlocks(block_indices, valid);
    if (valid.empty()) {
        return;
    }
    block_pool_->requestFree(valid);
}

void LinearKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    BlockIndicesType valid;
    filterValidBlocks(new_block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestReference(valid);
    }
}

}  // namespace rtp_llm
