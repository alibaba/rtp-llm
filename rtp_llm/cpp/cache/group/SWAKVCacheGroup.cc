#include "rtp_llm/cpp/cache/group/SWAKVCacheGroup.h"

#include <algorithm>
#include <cstdlib>
#include <string>

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

bool dsv4TrapInvalidKVAccessEnabled() {
    const char* value = std::getenv("DSV4_TRAP_INVALID_KV_ACCESS");
    if (value == nullptr) {
        return false;
    }
    const std::string flag(value);
    return !flag.empty() && flag != "0" && flag != "false" && flag != "FALSE" && flag != "off" && flag != "OFF";
}

}  // namespace

bool SWAKVCacheGroup::shouldCheckSWATailBlockIds() const {
    if (!dsv4TrapInvalidKVAccessEnabled()) {
        return false;
    }
    return policy_.validate_tail_blocks;
}

bool SWAKVCacheGroup::effectiveReuseCacheForAllocation(bool enable_reuse_cache) const {
    return enable_reuse_cache && policy_.reuse_policy == CacheReusePolicy::REUSABLE;
}

int SWAKVCacheGroup::activeTailBlockCount() const {
    return std::max(1, policy_.active_tail_blocks);
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
    return std::max((seq_len + reserve_step + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
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
    auto block_idx = shared_cache_->matchGroup(cache_key, group_id_);
    if (!isNullBlockIdx(block_idx)) {
        result.block_indices = {block_idx};
    }
    return result;
}

bool SWAKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
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
        auto allocated_opt = block_pool_->malloc(static_cast<size_t>(need_alloc_blocks));
        // malloc(n) is atomic: it returns exactly n blocks or nullopt.
        if (!allocated_opt.has_value() || allocated_opt->size() != static_cast<size_t>(need_alloc_blocks)) {
            return false;
        }
        allocated_blocks = std::move(*allocated_opt);
        // malloc() only reserves capacity at refCount 0; take the request holder ref to hold the blocks.
        block_pool_->incRef(allocated_blocks);
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
            break;
        }
        if (effective_reuse_enabled && ((i + 1) % step) == 0) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->releaseRef(blocks_to_free);
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
        block_pool_->releaseRef(valid);
    }
}

void SWAKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    BlockIndicesType valid;
    filterValidBlocks(new_block_indices, valid);
    if (!valid.empty()) {
        block_pool_->incRef(valid);
    }
}

}  // namespace rtp_llm
