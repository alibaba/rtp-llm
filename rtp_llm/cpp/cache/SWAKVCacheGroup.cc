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

int SWAKVCacheGroup::countTailAllocations(int begin, int end, int total_slots) const {
    if (end <= begin || total_slots <= 0) {
        return 0;
    }
    const int tail_begin  = std::max(0, total_slots - 2);
    const int alloc_begin = std::max(begin, tail_begin);
    return std::max(end - alloc_begin, 0);
}

// Estimate per-sequence block need for a new SWA request (capacity check before malloc).
// SWA only physically allocates the last 2 slots ("tail"); earlier slots are NULL_BLOCK_IDX.
//   1. tail:    2 blocks if seq fills a full block, else 1
//   2. reserve: extra slots from reserve_step for future decode headroom
//   3. reuse:   reused blocks overlapping the tail reduce new allocations
// SWA sequences don't share blocks across a batch, so common_blocks is always 0.
NeedBlocksInfo SWAKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)common_seq_len;
    NeedBlocksInfo info;
    if (seq_len <= 0) {
        return info;
    }

    const int total_slots   = needBlocksNum(seq_len, /*current_blocks=*/0);
    const int tail_blocks   = std::min(total_slots, 2);
    const int reserve_extra = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step) - total_slots;

    int need = tail_blocks + reserve_extra;
    if (reuse_enabled) {
        const int alloc_begin   = std::max(0, total_slots - 2);
        const int reuse_overlap = std::max(reuse_blocks_len - alloc_begin, 0);
        need -= reuse_overlap;
    }

    info.extra_blocks = std::max(need, 0);
    return info;
}

MatchResult SWAKVCacheGroup::matchSingleKey(CacheKeyType cache_key) const {
    MatchResult result;
    auto        matched = block_cache_->match(cache_key, group_id_);
    if (!isNullBlockIdx(matched.matched_index)) {
        result.block_indices = {matched.matched_index};
    }
    return result;
}

MatchResult SWAKVCacheGroup::match(const CacheKeysType& cache_keys) {
    (void)cache_keys;
    return {};
}

bool SWAKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int new_blocks_len     = needBlocksNum(seq_len, current_blocks_len, reserve_step);
    if (new_blocks_len == 0) {
        return true;
    }

    const int total_slots       = current_blocks_len + new_blocks_len;
    const int need_alloc_blocks = countTailAllocations(current_blocks_len, total_slots, total_slots);
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
            return false;
        }
    }

    BlockIndicesType new_ids;
    new_ids.reserve(static_cast<size_t>(new_blocks_len));
    size_t    allocated_pos = 0;
    const int tail_begin    = std::max(0, total_slots - 2);
    for (int i = current_blocks_len; i < total_slots; ++i) {
        if (i >= tail_begin) {
            new_ids.push_back(allocated_blocks[allocated_pos++]);
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

void SWAKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    (void)reserve_step;
    const auto& block_indices = block_ids.blocks();
    if (block_indices.size() <= 2) {
        return;
    }

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    const size_t        keep_begin = block_indices.size() - 2;
    for (size_t i = 0; i < keep_begin; ++i) {
        if (isNullBlockIdx(block_indices[i])) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(i);
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
