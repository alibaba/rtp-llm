#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

int estimatePeakFromSlotCosts(std::vector<int> block_costs,
                              int              final_slots,
                              int              new_slot_cost,
                              int              reserve_step,
                              int              retained_tail_blocks,
                              bool             enable_reuse_cache,
                              int              linear_step) {
    int physical_blocks = std::accumulate(block_costs.begin(), block_costs.end(), 0);
    int peak_blocks     = physical_blocks;

    while (static_cast<int>(block_costs.size()) < final_slots) {
        block_costs.push_back(new_slot_cost);
        physical_blocks += new_slot_cost;
        peak_blocks = std::max(peak_blocks, physical_blocks);

        for (int slot = static_cast<int>(block_costs.size()) - retained_tail_blocks - 1 - reserve_step; slot >= 0;
             --slot) {
            auto& cost = block_costs[static_cast<size_t>(slot)];
            if (cost == 0) {
                continue;
            }
            if (enable_reuse_cache && (slot + 1) % linear_step == 0) {
                continue;
            }
            physical_blocks -= cost;
            cost = 0;
        }
    }
    return peak_blocks;
}

}  // namespace

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
    int       extra_blocks = reserve_step ? reserve_step - 1 : 0;
    const int block_size   = seqSizePerBlock();
    return std::max((seq_len + block_size - 1) / block_size + extra_blocks - current_blocks, 0);
}

int LinearKVCacheGroup::materializedTailBlockCount() const {
    return std::max(1, static_cast<int>(activeTailBlocks()));
}

int LinearKVCacheGroup::retainedTailBlockCount() const {
    return std::max(2, materializedTailBlockCount());
}

bool LinearKVCacheGroup::shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const {
    if (pos < 0) {
        return false;
    }

    const int  step                     = std::max(1, linear_step_);
    const int  materialized_tail_blocks = materializedTailBlockCount();
    const int  seq_slots                = needBlocksNum(seq_len, 0, 0);
    const int  total_slots              = needBlocksNum(seq_len, 0, reserve_step);
    const bool is_seq_tail =
        (seq_slots > 0) && (pos >= std::max(0, seq_slots - materialized_tail_blocks)) && (pos < seq_slots);
    const bool is_reserve = (reserve_step > 0) && (pos >= seq_slots) && (pos < total_slots);
    const bool step_hit   = (((pos + 1) % step) == 0);
    return is_reserve || (enable_reuse_cache ? (step_hit || is_seq_tail) : is_seq_tail);
}

int LinearKVCacheGroup::estimatePeakNeedBlocks(int                     seq_len,
                                               const BlockIndicesType& current_block_indices,
                                               int                     remaining_tokens,
                                               int                     reserve_step,
                                               bool                    enable_reuse_cache) const {
    const int step              = std::max(1, linear_step_);
    const int current_seq_slots = (seq_len + seqSizePerBlock() - 1) / seqSizePerBlock();
    const int final_seq_slots   = (seq_len + remaining_tokens + seqSizePerBlock() - 1) / seqSizePerBlock();
    const int extra_blocks      = reserve_step ? reserve_step - 1 : 0;
    const int total_slots       = final_seq_slots + extra_blocks;

    std::vector<int> block_costs;
    block_costs.reserve(std::max(total_slots, static_cast<int>(current_block_indices.size())));

    int current_physical_blocks = 0;
    if (!current_block_indices.empty()) {
        for (const auto block_index : current_block_indices) {
            const bool allocated = !isNullBlockIdx(block_index);
            block_costs.push_back(allocated ? 1 : 0);
            current_physical_blocks += allocated;
        }
    } else {
        const int initial_slots = current_seq_slots + extra_blocks;
        for (int i = 0; i < initial_slots; ++i) {
            block_costs.push_back(shouldMaterializeBlock(i, seq_len, reserve_step, enable_reuse_cache) ? 1 : 0);
        }
    }

    const int peak_blocks = estimatePeakFromSlotCosts(
        std::move(block_costs), total_slots, 1, reserve_step, retainedTailBlockCount(), enable_reuse_cache, step);
    return std::max(peak_blocks - current_physical_blocks, 0);
}

int LinearKVCacheGroup::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                           int  common_seq_len,
                                                           int  remaining_tokens,
                                                           int  reserve_step,
                                                           bool enable_reuse_cache,
                                                           int  target_batch_size) const {
    const int  step       = std::max(1, linear_step_);
    const int  batch_size = std::max(target_batch_size, 1);
    const auto seq_slots  = [&](int length) { return (length + seqSizePerBlock() - 1) / seqSizePerBlock(); };

    const int        common_slots   = seq_slots(common_seq_len);
    const int        initial_slots  = seq_slots(seq_len);
    const int        reserve_blocks = reserve_step > 0 ? reserve_step - 1 : 0;
    const int        initial_total  = initial_slots + reserve_blocks;
    const int        final_total    = seq_slots(seq_len + remaining_tokens) + reserve_blocks;
    std::vector<int> block_costs;
    block_costs.reserve(static_cast<size_t>(std::max(initial_total, final_total)));

    for (int slot = 0; slot < common_slots; ++slot) {
        block_costs.push_back(shouldMaterializeBlock(slot, common_seq_len, 0, enable_reuse_cache) ? 1 : 0);
    }

    // initMalloc's incrMalloc phase appends the whole prompt suffix without sparse cleanup.
    for (int slot = common_slots; slot < initial_total; ++slot) {
        block_costs.push_back(shouldMaterializeBlock(slot, seq_len, reserve_step, enable_reuse_cache) ? batch_size : 0);
    }

    return estimatePeakFromSlotCosts(std::move(block_costs),
                                     final_total,
                                     batch_size,
                                     reserve_step,
                                     retainedTailBlockCount(),
                                     enable_reuse_cache,
                                     step);
}

NeedBlocksInfo LinearKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    NeedBlocksInfo info;

    const int common_slots = needBlocksNum(common_seq_len, 0);
    const int total_slots  = needBlocksNum(seq_len, 0, reserve_step);

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
    if (!shared_cache_) {
        return result;
    }
    auto block_idx = shared_cache_->matchGroup(cache_key, tag());
    if (!isNullBlockIdx(block_idx)) {
        result.block_indices = {block_idx};
    }
    return result;
}

bool LinearKVCacheGroup::malloc(BlockIds&            block_ids,
                                int                  seq_len,
                                bool                 enable_reuse_cache,
                                int                  reserve_step,
                                std::vector<size_t>* backfilled_positions) {
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int total_slots        = needBlocksNum(seq_len, 0, reserve_step);
    const int new_blocks_len     = std::max(total_slots - current_blocks_len, 0);

    auto should_materialize = [&](int pos) {
        return shouldMaterializeBlock(pos, seq_len, reserve_step, enable_reuse_cache);
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
            if (!allocated_blocks.empty()) {
                block_pool_->requestFree(allocated_blocks);
            }
            return false;
        }
    }

    size_t allocated_idx = 0;
    for (size_t pos : positions_to_backfill) {
        block_ids.setAt(pos, allocated_blocks[allocated_idx++]);
    }
    if (backfilled_positions != nullptr) {
        *backfilled_positions = positions_to_backfill;
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

void LinearKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        return;
    }
    const int step                 = std::max(1, linear_step_);
    const int retained_tail_blocks = retainedTailBlockCount();
    const int block_size           = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - retained_tail_blocks - 1 - reserve_step; i >= 0; i--) {
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
        block_ids.remove(pos_to_remove);
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
