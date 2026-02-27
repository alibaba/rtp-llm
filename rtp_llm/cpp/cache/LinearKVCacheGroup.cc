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
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ + reserve_step - current_blocks, 0);
}

NeedBlocksInfo LinearKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    const int reuse_begin = reuse_blocks_len;
    const int step        = std::max(1, linear_step_);

    // calculate the number of blocks in the range (begin, end]
    auto count_linear_sparse_range = [&](int begin, int end) -> int {
        if (end <= begin) {
            return 0;
        }
        if (!reuse_enabled) {
            // keeps only the tail block
            return 1;
        }
        const int eligible = (end + 1) / step - (begin + 1) / step;
        const int tail     = ((end + 1) % step == 0) ? 0 : 1;
        return eligible + tail;
    };

    NeedBlocksInfo info;

    // common_slots: blocks for common_seq_len (no reserve)
    const int common_slots = needBlocksNum(common_seq_len, 0);
    // seq_slots: blocks for seq_len (no reserve)
    const int seq_slots = needBlocksNum(seq_len, 0);
    // total_slots = seq_slots + reserve_step
    const int total_slots = needBlocksNum(seq_len, 0, reserve_step);

    info.common_blocks = count_linear_sparse_range(reuse_begin, common_slots);
    info.extra_blocks  = count_linear_sparse_range(common_slots, seq_slots);
    info.extra_blocks += std::max(total_slots - seq_slots, 0);  // for reserve_step

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

MatchResult LinearKVCacheGroup::match(const CacheKeysType&                 cache_keys,
                                      const std::vector<std::vector<int>>& mm_intervals) {
    return {};
}

bool LinearKVCacheGroup::malloc(BlockIndicesType& block_indices,
                                int               seq_len,
                                bool              enable_reuse_cache,
                                int               reserve_step) {
    const int step               = std::max(1, linear_step_);
    const int current_blocks_len = static_cast<int>(block_indices.size());
    const int seq_slots          = needBlocksNum(seq_len, 0, 0);
    const int new_blocks_len     = needBlocksNum(seq_len, current_blocks_len, reserve_step);

    if (new_blocks_len == 0) {
        return true;
    }

    // LinearKVCacheGroup::malloc is responsible for:
    // 1. allocating blocks for the current sequence length;
    // 2. free unused blocks to reduce kvcache block usage;

    // Two policies to follow:
    // 1. Linear Steps: keep N * linear_step blocks if cache reuse enabled;
    // 2. Allocate Tail Blocks: allocate the last partial block when initialization and keep last 2 block during
    // decoding;

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
                RTP_LLM_LOG_WARNING("Insufficient free blocks for LinearKVCacheGroup: need %d, have %zu",
                                    need_alloc_blocks,
                                    free_blocks_num);
                return false;
            }
        }
    }

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
            block_indices.push_back(result[0]);
        } else {
            block_indices.push_back(NULL_BLOCK_IDX);
        }
    }
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

void LinearKVCacheGroup::removeSkippedBlocks(BlockIndicesType& block_indices,
                                             bool              enable_reuse_cache,
                                             int               reserve_step) {
    if (block_indices.empty()) {
        return;
    }
    const int step       = std::max(1, linear_step_);
    const int block_size = block_indices.size();
    // keep last 2 and every reserve_step
    for (int i = block_size - 3 - reserve_step; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[i])) {
            break;
        }
        if (enable_reuse_cache && ((i + 1) % step) == 0) {
            continue;
        }
        block_pool_->requestFree(block_indices[i]);
        block_indices[i] = NULL_BLOCK_IDX;
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

void LinearKVCacheGroup::reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) {
    block_indices.insert(block_indices.end(), new_block_indices.begin(), new_block_indices.end());
    BlockIndicesType valid;
    filterValidBlocks(new_block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestReference(valid);
    }
}

}  // namespace rtp_llm
