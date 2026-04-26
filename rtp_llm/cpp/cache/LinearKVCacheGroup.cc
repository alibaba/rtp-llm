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

int LinearKVCacheGroup::logicalSlots(int seq_len, int reserve_step) const {
    const int extra = reserve_step ? reserve_step - 1 : 0;
    return (seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ + extra;
}

int LinearKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    if (fixed_cap_ > 0) {
        const int target = std::min(logicalSlots(seq_len, reserve_step), fixed_cap_);
        return std::max(target - current_blocks, 0);
    }
    int extra_blocks = reserve_step ? reserve_step - 1 : 0;
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ + extra_blocks - current_blocks, 0);
}

bool LinearKVCacheGroup::shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const {
    if (pos < 0) {
        return false;
    }
    if (fixed_cap_ > 0) {
        return pos < std::min(logicalSlots(seq_len, reserve_step), fixed_cap_);
    }

    const int  step        = std::max(1, linear_step_);
    const int  seq_slots   = logicalSlots(seq_len, 0);
    const int  total_slots = logicalSlots(seq_len, reserve_step);
    const bool is_seq_tail = (seq_slots > 0) && (pos >= std::max(0, seq_slots - 2)) && (pos < seq_slots);
    const bool is_reserve  = (reserve_step > 0) && (pos >= seq_slots) && (pos < total_slots);
    const bool step_hit    = (((pos + 1) % step) == 0);
    return is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
}

NeedBlocksInfo LinearKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    NeedBlocksInfo info;
    if (fixed_cap_ > 0) {
        // Ring buffer: at any moment we hold at most `fixed_cap_` blocks.
        // common_blocks / extra_blocks here are informational — used by the
        // caller to estimate "need new blocks beyond the already-referenced
        // prefix".  Cap both at `fixed_cap_` so callers do not pre-reserve
        // pool blocks we will never hold simultaneously.
        const int common_slots = std::min(logicalSlots(common_seq_len, 0), fixed_cap_);
        const int seq_slots    = std::min(logicalSlots(seq_len, 0), fixed_cap_);
        const int total_slots  = std::min(logicalSlots(seq_len, reserve_step), fixed_cap_);

        info.common_blocks = std::max(common_slots - std::min(reuse_blocks_len, fixed_cap_), 0);
        info.extra_blocks =
            std::max(total_slots - std::max(common_slots, seq_slots), 0) + std::max(seq_slots - common_slots, 0);
        return info;
    }

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
    MatchResult result;
    const int   m = static_cast<int>(cache_keys.size());
    if (m == 0) {
        return result;
    }

    // Ring-buffer Linear groups do not own prefix-reuse matching. Their
    // retained blocks represent a rolling tail/state window, so prefix reuse
    // must be decided by the allocator or by a dedicated SWA group.
    if (fixed_cap_ > 0) {
        return result;
    }

    // Legacy (fixed_cap_ == 0): keep stash-era behavior that scans only the
    // last two positions and returns a prefix-length block_indices with
    // NULL placeholders at non-hit slots.
    const int scan_window = 2;
    const int start       = m - 1;
    const int end         = std::max(0, m - scan_window);

    for (int i = start; i >= end; --i) {
        auto single = matchSingleKey(cache_keys[static_cast<size_t>(i)]);
        if (single.block_indices.empty()) {
            continue;
        }
        result.reuse_blocks = static_cast<size_t>(i + 1);
        result.block_indices.assign(static_cast<size_t>(i + 1), NULL_BLOCK_IDX);
        result.block_indices[static_cast<size_t>(i)] = single.block_indices[0];
        if (i - 1 >= 0) {
            auto prev = matchSingleKey(cache_keys[static_cast<size_t>(i - 1)]);
            if (!prev.block_indices.empty()) {
                result.block_indices[static_cast<size_t>(i - 1)] = prev.block_indices[0];
            }
        }
        result.reuse_length = result.reuse_blocks * static_cast<size_t>(seq_size_per_block_);
        return result;
    }
    return result;
}

bool LinearKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    if (fixed_cap_ > 0) {
        // Ring-buffer malloc — maintain invariant block_ids.blocksNum() <= fixed_cap_
        // with all entries valid (no NULL padding).
        const int new_slots = logicalSlots(seq_len, reserve_step);
        const int target    = std::min(new_slots, fixed_cap_);
        int       current   = static_cast<int>(block_ids.blocksNum());

        // Recycle detection: BlockIds pointer reused for a new request (starts empty).
        auto seen_it = last_seq_slots_.find(&block_ids);
        int  prev    = (seen_it != last_seq_slots_.end()) ? seen_it->second : 0;
        if (current == 0 && prev > 0) {
            prev = 0;
        }

        // Phase 1: fill up to `target` (first-fill before the ring is full).
        if (current < target) {
            const int fill = target - current;
            if (freeBlocksNum() < static_cast<size_t>(fill) && !ensureFreeBlocks(fill)) {
                RTP_LLM_LOG_WARNING(
                    "LinearKVCacheGroup ring fill: insufficient free blocks, need %d have %zu", fill, freeBlocksNum());
                return false;
            }
            auto new_ids = block_pool_->malloc(static_cast<size_t>(fill));
            if (static_cast<int>(new_ids.size()) != fill) {
                if (!new_ids.empty()) {
                    block_pool_->requestFree(new_ids);
                }
                return false;
            }
            block_ids.add(new_ids);
            current = static_cast<int>(block_ids.blocksNum());
        }

        // Phase 2: rotation — how many block boundaries have been crossed
        // since the last call at the cap?
        if (current == fixed_cap_) {
            const int effective_prev = std::max(prev, current);
            const int rotations      = std::max(0, new_slots - effective_prev);
            for (int r = 0; r < rotations; ++r) {
                if (freeBlocksNum() < 1 && !ensureFreeBlocks(1)) {
                    RTP_LLM_LOG_WARNING("LinearKVCacheGroup ring rotate: no free block available");
                    return false;
                }
                auto new_id = block_pool_->malloc(1);
                if (new_id.empty()) {
                    return false;
                }
                // Free the oldest and shift: [A, B] -> [B, ?] -> [B, new]
                const auto oldest = block_ids.blocks()[0];
                if (!isNullBlockIdx(oldest)) {
                    block_pool_->requestFree({oldest});
                    // Drop block_to_key_ mapping for the evicted block so its
                    // stale cache entry cleanup on next insert doesn't fail.
                    block_to_key_.erase(oldest);
                }
                block_ids.swap(0, 1);
                block_ids.popBack();
                block_ids.add({new_id[0]});
            }
        }

        last_seq_slots_[&block_ids] = new_slots;
        return true;
    }

    // ---- Legacy path (fixed_cap_ == 0) ----
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
        // If this block was previously inserted with a different key, remove
        // the stale entry first. This happens when fixed-allocation pools
        // (SWA/State) reuse the same physical block for updated content.
        auto it = block_to_key_.find(b);
        if (it != block_to_key_.end() && it->second != cache_keys[i]) {
            block_cache_->remove(it->second, group_id_);
            block_to_key_.erase(it);
        }
        BlockCache::CacheItem item;
        item.cache_key   = cache_keys[i];
        item.group_id    = group_id_;
        item.block_index = b;
        item.is_resident = is_resident;
        if (block_cache_->put(item)) {
            block_pool_->blockCacheReference(b);
            block_to_key_[b] = cache_keys[i];
        }
    }
}

void LinearKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    if (fixed_cap_ > 0) {
        // Ring-buffer mode: malloc already maintains the invariant; nothing to skip.
        return;
    }
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

void LinearKVCacheGroup::recordReuse(const BlockIds& block_ids, int reuse_slots) {
    if (fixed_cap_ <= 0) {
        return;
    }
    last_seq_slots_[&block_ids] = std::max(reuse_slots, 0);
}

}  // namespace rtp_llm
