#include "rtp_llm/cpp/cache/SWACacheManager.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isActiveTailBlock(int block_idx, int seq_slots, int active_tail_blocks) {
    if (seq_slots <= 0 || block_idx >= seq_slots) {
        return false;
    }
    return block_idx >= std::max(seq_slots - active_tail_blocks, 0);
}

bool shouldAllocateBlock(int block_idx, int seq_slots, int reserve_step, int active_tail_blocks) {
    const bool is_reserve = reserve_step > 0 && block_idx >= seq_slots;
    return is_reserve || isActiveTailBlock(block_idx, seq_slots, active_tail_blocks);
}

int estimatePeakFromSlotCosts(std::vector<int> block_costs,
                              int              first_decode_slots,
                              int              first_decode_tail_start,
                              int              first_decode_peak_slots,
                              int              final_slots,
                              int              max_decode_growth,
                              int              new_slot_cost,
                              int              retained_tail_blocks) {
    int physical_blocks = std::accumulate(block_costs.begin(), block_costs.end(), 0);
    int peak_blocks     = physical_blocks;
    for (int slot = first_decode_tail_start; slot < std::min(first_decode_slots, static_cast<int>(block_costs.size()));
         ++slot) {
        if (block_costs[slot] == 0) {
            block_costs[slot] = new_slot_cost;
            physical_blocks += new_slot_cost;
        }
    }
    size_t next_to_release = 0;
    auto   advance         = [&](int slots) {
        while (static_cast<int>(block_costs.size()) < slots) {
            block_costs.push_back(new_slot_cost);
            physical_blocks += new_slot_cost;
        }
        peak_blocks = std::max(peak_blocks, physical_blocks);
        const size_t tail_start =
            block_costs.size() - std::min(static_cast<size_t>(retained_tail_blocks), block_costs.size());
        while (next_to_release < tail_start) {
            physical_blocks -= block_costs[next_to_release];
            block_costs[next_to_release++] = 0;
        }
    };
    peak_blocks = std::max(
        peak_blocks,
        physical_blocks + std::max(first_decode_peak_slots - static_cast<int>(block_costs.size()), 0) * new_slot_cost);
    advance(first_decode_slots);
    while (static_cast<int>(block_costs.size()) < final_slots) {
        // A speculative decode may append several slots before any old blocks are released.
        const int growth = std::min(max_decode_growth, final_slots - static_cast<int>(block_costs.size()));
        peak_blocks      = std::max(peak_blocks, physical_blocks + growth * new_slot_cost);
        advance(static_cast<int>(block_costs.size()) + 1);
    }
    return peak_blocks;
}

}  // namespace

bool SWACacheManager::shouldCheckSWATailBlockIds() const {
    return policy().validate_tail_blocks;
}

int SWACacheManager::activeTailBlockCount() const {
    return static_cast<int>(std::max(1u, policy().active_tail_blocks));
}

void SWACacheManager::checkSWATailBlockIds(const BlockIds& block_ids, const char* caller) const {
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

void SWACacheManager::checkHistoryBlockIds(
    const BlockIds& block_ids, HistorySlots history, const char* caller, bool check_tail) const {
    if (!shouldCheckSWATailBlockIds()) {
        return;
    }
    const auto& blocks = block_ids.blocks();
    RTP_LLM_CHECK_WITH_INFO(history.begin <= history.end && history.end <= blocks.size(),
                            "%s invalid SWA history slots [%zu,%zu), block_num=%zu",
                            caller,
                            history.begin,
                            history.end,
                            blocks.size());
    for (size_t slot = history.begin; slot < history.end; ++slot) {
        RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(blocks[slot]),
                                "%s missing required SWA history at slot=%zu, block_num=%zu",
                                caller,
                                slot,
                                blocks.size());
    }
    if (check_tail && !blocks.empty()) {
        RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(blocks.back()),
                                "%s invalid SWA block ids: tail block is NULL, block_num=%zu",
                                caller,
                                blocks.size());
    }
}

void SWACacheManager::filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const {
    out.clear();
    out.reserve(in.size());
    for (auto b : in) {
        if (!isNullBlockIdx(b)) {
            out.push_back(b);
        }
    }
}

int SWACacheManager::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    const int block_size = seqSizePerBlock();
    return std::max((seq_len + reserve_step + block_size - 1) / block_size - current_blocks, 0);
}

int SWACacheManager::estimatePeakNeedBlocks(int                     seq_len,
                                            const BlockIndicesType& current_block_indices,
                                            int                     remaining_tokens,
                                            int                     reserve_step,
                                            bool                    enable_reuse_cache) const {
    const int        active_tail_blocks  = activeTailBlockCount();
    const int        prefill_tail_blocks = prefillTailBlocks(tail_blocks_, enable_reuse_cache);
    std::vector<int> block_costs;
    int              allocated_blocks = 0;
    if (current_block_indices.empty()) {
        const int seq_slots   = needBlocksNum(seq_len, 0);
        const int total_slots = seq_slots;
        block_costs.reserve(total_slots);
        for (int slot = 0; slot < total_slots; ++slot) {
            block_costs.push_back(isActiveTailBlock(slot, seq_slots, prefill_tail_blocks) ? 1 : 0);
        }
    } else {
        block_costs.reserve(current_block_indices.size());
        for (const auto block_index : current_block_indices) {
            const int cost = !isNullBlockIdx(block_index);
            block_costs.push_back(cost);
            allocated_blocks += cost;
        }
    }
    if (remaining_tokens <= 0) {
        return std::accumulate(block_costs.begin(), block_costs.end(), 0) - allocated_blocks;
    }
    const int decode_tokens = std::max(1, reserve_step);
    const int peak_blocks =
        estimatePeakFromSlotCosts(std::move(block_costs),
                                  needBlocksNum(seq_len + 1, 0, reserve_step),
                                  std::max(needBlocksNum(seq_len + 1, 0) - active_tail_blocks, 0),
                                  needBlocksNum(seq_len + std::min(remaining_tokens, decode_tokens), 0, reserve_step),
                                  needBlocksNum(seq_len + remaining_tokens, 0, reserve_step),
                                  needBlocksNum(decode_tokens, 0),
                                  1,
                                  active_tail_blocks + reserve_step);
    return std::max(peak_blocks - allocated_blocks, 0);
}

int SWACacheManager::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                        int  common_seq_len,
                                                        int  remaining_tokens,
                                                        int  reserve_step,
                                                        bool enable_reuse_cache,
                                                        int  target_batch_size) const {
    const int        prefill_tail_blocks = prefillTailBlocks(tail_blocks_, enable_reuse_cache);
    const int        batch_size          = std::max(target_batch_size, 1);
    const int        active_tail_blocks  = activeTailBlockCount();
    const int        common_slots        = needBlocksNum(std::clamp(common_seq_len, 0, seq_len), 0);
    const int        seq_slots           = needBlocksNum(seq_len, 0);
    const int        total_slots         = seq_slots;
    std::vector<int> block_costs;
    block_costs.reserve(total_slots);
    for (int slot = 0; slot < common_slots; ++slot) {
        block_costs.push_back(isActiveTailBlock(slot, common_slots, prefill_tail_blocks) ? 1 : 0);
    }
    for (int slot = common_slots; slot < total_slots; ++slot) {
        block_costs.push_back(isActiveTailBlock(slot, seq_slots, prefill_tail_blocks) ? batch_size : 0);
    }
    if (remaining_tokens <= 0) {
        return std::accumulate(block_costs.begin(), block_costs.end(), 0);
    }
    const int decode_tokens = std::max(1, reserve_step);
    return estimatePeakFromSlotCosts(
        std::move(block_costs),
        needBlocksNum(seq_len + 1, 0, reserve_step),
        std::max(needBlocksNum(seq_len + 1, 0) - active_tail_blocks, 0),
        needBlocksNum(seq_len + std::min(remaining_tokens, decode_tokens), 0, reserve_step),
        needBlocksNum(seq_len + remaining_tokens, 0, reserve_step),
        needBlocksNum(decode_tokens, 0),
        batch_size,
        active_tail_blocks + reserve_step);
}

NeedBlocksInfo SWACacheManager::getNeedBlocks(int                      common_seq_len,
                                              int                      seq_len,
                                              int                      reserve_step,
                                              int                      reuse_blocks_len,
                                              bool                     reuse_enabled,
                                              const RequiredPositions& required_positions) const {
    (void)reserve_step;
    const int      tail_blocks  = prefillTailBlocks(tail_blocks_, reuse_enabled);
    const int      reuse_slots  = reuse_enabled ? reuse_blocks_len : 0;
    const int      common_slots = needBlocksNum(std::clamp(common_seq_len, 0, seq_len), 0);
    const int      total_slots  = needBlocksNum(seq_len, 0);
    NeedBlocksInfo info;
    for (int pos = 0; pos < common_slots; ++pos) {
        if ((pos >= reuse_slots && isActiveTailBlock(pos, common_slots, tail_blocks))
            || required_positions.count(static_cast<size_t>(pos)) != 0) {
            ++info.common_blocks;
        }
    }
    for (int pos = common_slots; pos < total_slots; ++pos) {
        if ((pos >= reuse_slots && isActiveTailBlock(pos, total_slots, tail_blocks))
            || required_positions.count(static_cast<size_t>(pos)) != 0) {
            ++info.extra_blocks;
        }
    }
    return info;
}

bool SWACacheManager::malloc(BlockIds&                block_ids,
                             int                      seq_len,
                             bool                     enable_reuse_cache,
                             int                      reserve_step,
                             std::vector<size_t>*     backfilled_positions,
                             const RequiredPositions& required_positions) {
    (void)reserve_step;
    return mallocImpl(block_ids,
                      seq_len,
                      0,
                      prefillTailBlocks(tail_blocks_, enable_reuse_cache),
                      true,
                      backfilled_positions,
                      required_positions);
}

bool SWACacheManager::mallocForDecode(BlockIds&            block_ids,
                                      int                  seq_len,
                                      bool                 enable_reuse_cache,
                                      int                  reserve_step,
                                      std::vector<size_t>* backfilled_positions) {
    (void)enable_reuse_cache;
    return mallocImpl(block_ids, seq_len, reserve_step, activeTailBlockCount(), false, backfilled_positions, {});
}

bool SWACacheManager::mallocWithHistory(BlockIds&                block_ids,
                                         int                      seq_len,
                                         bool                     enable_reuse_cache,
                                         int                      reserve_step,
                                         std::vector<size_t>*     backfilled_positions,
                                         const RequiredPositions& required_positions,
                                         HistorySlots             history) {
    (void)reserve_step;
    return mallocImpl(block_ids,
                      seq_len,
                      0,
                      prefillTailBlocks(tail_blocks_, enable_reuse_cache),
                      true,
                      backfilled_positions,
                      required_positions,
                      history);
}

bool SWACacheManager::mallocImpl(BlockIds&                   block_ids,
                                 int                         seq_len,
                                 int                         reserve_step,
                                 int                         tail_blocks,
                                 bool                        is_prefill,
                                 std::vector<size_t>*        backfilled_positions,
                                 const RequiredPositions&    required_positions,
                                 std::optional<HistorySlots> history) {
    if (history) {
        // A recovered history slot must already be bound to a cached block.
        // Reject a missing slot before allocating any fresh continuation block.
        checkHistoryBlockIds(block_ids, *history, "SWACacheManager::mallocWithHistory", false);
    }
    const auto check_blocks = [&]() {
        if (history) {
            checkHistoryBlockIds(block_ids, *history, "SWACacheManager::mallocWithHistory");
        } else {
            checkSWATailBlockIds(block_ids, "SWACacheManager::malloc");
        }
    };
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int seq_slots          = needBlocksNum(seq_len, 0, 0);
    const int new_blocks_len     = needBlocksNum(seq_len, current_blocks_len, reserve_step);

    if (is_prefill && required_positions.empty() && new_blocks_len == 0) {
        check_blocks();
        return true;
    }

    auto is_required = [&](int pos) {
        return required_positions.find(static_cast<size_t>(pos)) != required_positions.end();
    };
    auto should_allocate = [&](int pos) {
        return shouldAllocateBlock(pos, seq_slots, reserve_step, tail_blocks) || is_required(pos);
    };

    std::vector<size_t> positions_to_backfill;
    if (!is_prefill || !required_positions.empty()) {
        const auto& existing_blocks = block_ids.blocks();
        const int   existing_scan   = std::min(current_blocks_len, needBlocksNum(seq_len, 0, reserve_step));
        for (int i = 0; i < existing_scan; ++i) {
            if ((is_required(i) || (!is_prefill && should_allocate(i)))
                && isNullBlockIdx(existing_blocks[static_cast<size_t>(i)])) {
                positions_to_backfill.push_back(static_cast<size_t>(i));
            }
        }
    }

    int need_alloc_blocks = static_cast<int>(positions_to_backfill.size());
    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        if (should_allocate(i)) {
            need_alloc_blocks++;
        }
    }

    if (need_alloc_blocks > 0) {
        const auto free_blocks_num = freeBlocksNum();
        if (free_blocks_num < static_cast<size_t>(need_alloc_blocks)) {
            if (!ensureFreeBlocks(need_alloc_blocks)) {
                RTP_LLM_LOG_WARNING("Insufficient free blocks for SWACacheManager: need %d, have %zu",
                                    need_alloc_blocks,
                                    free_blocks_num);
                return false;
            }
        }
    }

    BlockIndicesType allocated_blocks;
    if (need_alloc_blocks > 0) {
        auto allocated = block_pool_->malloc(static_cast<size_t>(need_alloc_blocks));
        if (!allocated.has_value()) {
            return false;
        }
        allocated_blocks = std::move(*allocated);
        block_pool_->incRef(allocated_blocks);
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
    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        if (should_allocate(i)) {
            new_ids.push_back(allocated_blocks[allocated_idx++]);
        } else {
            new_ids.push_back(NULL_BLOCK_IDX);
        }
    }
    block_ids.add(new_ids);
    check_blocks();
    return true;
}

void SWACacheManager::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    removeSkippedBlocksImpl(block_ids, enable_reuse_cache, reserve_step, std::nullopt);
}

void SWACacheManager::removeSkippedBlocksWithHistory(
    BlockIds& block_ids, bool enable_reuse_cache, int reserve_step, HistorySlots history) {
    removeSkippedBlocksImpl(block_ids, enable_reuse_cache, reserve_step, history);
}

void SWACacheManager::removeSkippedBlocksImpl(
    BlockIds& block_ids, bool enable_reuse_cache, int reserve_step, std::optional<HistorySlots> history) {
    const auto check_blocks = [&]() {
        if (history) {
            checkHistoryBlockIds(block_ids, *history, "SWACacheManager::removeSkippedBlocksWithHistory");
        } else {
            checkSWATailBlockIds(block_ids, "SWACacheManager::removeSkippedBlocks");
        }
    };
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        check_blocks();
        return;
    }
    (void)enable_reuse_cache;
    const int active_tail_blocks = activeTailBlockCount();
    const int block_size         = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - active_tail_blocks - 1 - reserve_step; i >= 0; i--) {
        if (history && static_cast<size_t>(i) >= history->begin && static_cast<size_t>(i) < history->end) {
            continue;
        }
        if (isNullBlockIdx(block_indices[i])) {
            continue;
        }
        blocks_to_free.push_back(block_indices[i]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->decRef(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
    check_blocks();
}

}  // namespace rtp_llm
