#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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
    return std::max((seq_len + reserve_step + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

NeedBlocksInfo SWAKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)common_seq_len;
    const int step = std::max(1, linear_step_);

    auto count_linear_sparse_range = [&](int begin, int end) -> int {
        if (end <= begin) {
            return 0;
        }
        if (!reuse_enabled) {
            return 1;
        }
        const int eligible = (end + 1) / step - (begin + 1) / step;
        const int tail     = ((end + 1) % step == 0) ? 0 : 1;
        return eligible + tail;
    };

    NeedBlocksInfo info;

    const int seq_slots   = needBlocksNum(seq_len, 0);
    const int total_slots = needBlocksNum(seq_len, 0, reserve_step);

    info.common_blocks = 0;
    info.extra_blocks  = count_linear_sparse_range(reuse_blocks_len, seq_slots);
    info.extra_blocks += std::max(total_slots - seq_slots, 0);

    info.extra_blocks = std::max(info.extra_blocks, 0);
    return info;
}

MatchResult SWAKVCacheGroup::matchSingleKey(CacheKeyType cache_key) const {
    MatchResult result;
    if (!shared_cache_) {
        return result;
    }
    auto block_idx = shared_cache_->matchGroup(cache_key, group_id_);
    if (!isNullBlockIdx(block_idx)) {
        result.block_indices = {block_idx};
    }
    return result;
}

MatchResult SWAKVCacheGroup::match(const CacheKeysType& cache_keys) {
    (void)cache_keys;
    RTP_LLM_CHECK_WITH_INFO(false, "SWA should not call match, use matchSingleKey instead");
    return {};
}

bool SWAKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    const int step               = std::max(1, linear_step_);
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int seq_slots          = needBlocksNum(seq_len, 0, 0);
    const int new_blocks_len     = needBlocksNum(seq_len, current_blocks_len, reserve_step);

    if (new_blocks_len == 0) {
        return true;
    }

    int need_alloc_blocks = 0;

    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        const bool is_seq_tail  = (seq_slots > 0) && (i == seq_slots - 1);
        const bool is_reserve   = (reserve_step > 0) && (i >= seq_slots);
        const bool step_hit     = (((i + 1) % step) == 0);
        const bool should_alloc = is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
        if (should_alloc) {
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

    BlockIndicesType new_ids;
    new_ids.reserve(static_cast<size_t>(new_blocks_len));
    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        const bool is_seq_tail  = (seq_slots > 0) && (i == seq_slots - 1);
        const bool is_reserve   = (reserve_step > 0) && (i >= seq_slots);
        const bool step_hit     = (((i + 1) % step) == 0);
        const bool should_alloc = is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
        if (should_alloc) {
            auto result = block_pool_->malloc(1);
            if (result.empty()) {
                return false;
            }
            new_ids.push_back(result[0]);
        } else {
            new_ids.push_back(NULL_BLOCK_IDX);
        }
    }
    block_ids.add(new_ids);
    return true;
}

void SWAKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                      const BlockIndicesType& block_indices,
                                      bool                    is_resident) {
    if (cache_keys.empty() || block_indices.empty() || !shared_cache_) {
        return;
    }
    const size_t n = std::min(cache_keys.size(), block_indices.size());
    for (size_t i = 0; i < n; ++i) {
        const auto b = block_indices[i];
        if (isNullBlockIdx(b)) {
            continue;
        }
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_id_ + 1), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id_)] = b;
        shared_cache_->put(cache_keys[i], group_slots, is_resident);
    }
}

void SWAKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        return;
    }
    const int step       = std::max(1, linear_step_);
    const int block_size = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - 3 - reserve_step; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[i])) {
            break;
        }
        if (enable_reuse_cache && ((i + 1) % step) == 0) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->requestFree(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
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
