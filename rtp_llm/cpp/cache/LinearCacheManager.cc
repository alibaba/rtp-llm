#include "rtp_llm/cpp/cache/LinearCacheManager.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

int estimatePeakFromSlotCosts(std::vector<int> block_costs,
                              int              first_decode_slots,
                              int              first_decode_tail_start,
                              int              final_slots,
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
    advance(first_decode_slots);
    while (static_cast<int>(block_costs.size()) < final_slots) {
        advance(static_cast<int>(block_costs.size()) + 1);
    }
    return peak_blocks;
}

}  // namespace

void LinearCacheManager::releaseRequestStateToDecodeTail(BlockIds& block_ids, int reserve_step) {
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        return;
    }
    const int block_size = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - std::min(retainedDecodeBlocks(reserve_step), block_size) - 1; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[static_cast<size_t>(i)])) {
            continue;
        }
        blocks_to_free.push_back(block_indices[static_cast<size_t>(i)]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->decRef(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
}

int LinearCacheManager::retainedDecodeBlocks(int reserve_step) const {
    return std::max(2, static_cast<int>(activeTailBlocks())) + reserve_step;
}

void LinearCacheManager::filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const {
    out.clear();
    out.reserve(in.size());
    for (auto b : in) {
        if (!isNullBlockIdx(b)) {
            out.push_back(b);
        }
    }
}

int LinearCacheManager::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    int       extra_blocks = reserve_step ? reserve_step - 1 : 0;
    const int block_size   = seqSizePerBlock();
    return std::max((seq_len + block_size - 1) / block_size + extra_blocks - current_blocks, 0);
}

// History-reclaim helpers (opt-in via enable_linear_cache_reclaim). The tail
// counts anchor the committed-slot retirement boundary in historyReclaimEnd.
int LinearCacheManager::materializedTailBlockCount() const {
    return std::max(1, static_cast<int>(activeTailBlocks()));
}

int LinearCacheManager::retainedTailBlockCount() const {
    return std::max(2, materializedTailBlockCount());
}

size_t LinearCacheManager::historyReclaimEnd(int committed_tokens) const {
    // Keep completed tail checkpoints as well as every incomplete/reserved slot.
    // The caller supplies committed tokens, never a speculative allocation bound.
    return static_cast<size_t>(
        std::max(0, std::max(0, committed_tokens) / seqSizePerBlock() - retainedTailBlockCount()));
}


bool LinearCacheManager::shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const {
    (void)reserve_step;
    const int tail_blocks = prefillTailBlocks(tail_blocks_, enable_reuse_cache);
    const int total_slots = needBlocksNum(seq_len, 0);
    return pos >= total_slots - std::min(tail_blocks, total_slots) && pos < total_slots;
}

int LinearCacheManager::estimatePeakNeedBlocks(int                     seq_len,
                                               const BlockIndicesType& current_block_indices,
                                               int                     remaining_tokens,
                                               int                     reserve_step,
                                               bool                    enable_reuse_cache,
                                               bool                    reclaim_history) const {
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
        const int initial_slots = current_seq_slots;
        for (int i = 0; i < initial_slots; ++i) {
            block_costs.push_back(shouldMaterializeBlock(i, seq_len, reserve_step, enable_reuse_cache) ? 1 : 0);
        }
    }

    if (reclaim_history) {
        if (!current_block_indices.empty()) {
            // Existing request/transfer owners can keep these blocks pinned after our
            // reference retires. Forecast only additional private allocations.
            for (auto& cost : block_costs) {
                cost = cost > 0 ? -1 : 0;
            }
        }
        return estimateHistoryReclaimPeak(std::move(block_costs),
                                          seq_len,
                                          remaining_tokens,
                                          reserve_step,
                                          enable_reuse_cache,
                                          !current_block_indices.empty());
    }

    // Main #1574 slot-cost peak model (no reclaim): the first decode slot
    // starts above the active tail, growth is bounded by retained decode blocks.
    if (remaining_tokens <= 0) {
        return std::accumulate(block_costs.begin(), block_costs.end(), 0) - current_physical_blocks;
    }
    const int first_decode_slots = needBlocksNum(seq_len + 1, 0, reserve_step);
    const int first_decode_tail_start =
        std::max(needBlocksNum(seq_len + 1, 0) - static_cast<int>(std::max<size_t>(1, activeTailBlocks())), 0);
    const int peak_blocks = estimatePeakFromSlotCosts(std::move(block_costs),
                                                      first_decode_slots,
                                                      first_decode_tail_start,
                                                      total_slots,
                                                      1,
                                                      retainedDecodeBlocks(reserve_step));
    return std::max(peak_blocks - current_physical_blocks, 0);
}

int LinearCacheManager::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                           int  common_seq_len,
                                                           int  remaining_tokens,
                                                           int  reserve_step,
                                                           bool enable_reuse_cache,
                                                           int  target_batch_size,
                                                           bool reclaim_history) const {
    const int  batch_size = std::max(target_batch_size, 1);
    const auto seq_slots  = [&](int length) { return (length + seqSizePerBlock() - 1) / seqSizePerBlock(); };

    const int        common_slots   = seq_slots(common_seq_len);
    const int        initial_slots  = seq_slots(seq_len);
    const int        reserve_blocks = reserve_step > 0 ? reserve_step - 1 : 0;
    const int        initial_total  = initial_slots;
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

    if (reclaim_history && batch_size == 1) {
        return estimateHistoryReclaimPeak(
            std::move(block_costs), seq_len, remaining_tokens, reserve_step, enable_reuse_cache, false);
    }

    // Main #1574 slot-cost peak model (no reclaim). With reclaim configured
    // but unavailable for this request (KEEP_HISTORY) and an unbounded reuse
    // tail, no release path applies: the fresh prompt plus every decode slot
    // below the needed boundary stays materialized, so the initial peak is the
    // full accumulation to the final length. Otherwise the walk releases
    // everything below the retained decode window.
    if (remaining_tokens <= 0) {
        return std::accumulate(block_costs.begin(), block_costs.end(), 0);
    }
    const int retained_tail_blocks = enable_linear_cache_reclaim_ && tail_blocks_ == std::numeric_limits<int>::max() ?
                                        std::numeric_limits<int>::max() :
                                        retainedDecodeBlocks(reserve_step);
    return estimatePeakFromSlotCosts(
        std::move(block_costs),
        needBlocksNum(seq_len + 1, 0, reserve_step),
        std::max(needBlocksNum(seq_len + 1, 0) - static_cast<int>(std::max<size_t>(1, activeTailBlocks())), 0),
        final_total,
        batch_size,
        retained_tail_blocks);
}

int LinearCacheManager::estimateHistoryReclaimPeak(std::vector<int> block_costs,
                                                   int              seq_len,
                                                   int              remaining_tokens,
                                                   int              reserve_step,
                                                   bool             enable_reuse_cache,
                                                   bool             initialized) const {
    int physical_blocks = 0;
    for (const int cost : block_costs) {
        physical_blocks += std::max(cost, 0);
    }
    int       peak_blocks = physical_blocks;
    size_t    retired     = 0;
    const int block_size  = seqSizePerBlock();
    const int final_len   = seq_len + std::max(remaining_tokens, 0);
    auto      allocate_at = [&](int length) {
        // A decode input contains one token beyond the last committed KV. Use
        // the exact retirement policy applied by the runtime before allocation.
        const size_t end = std::min(historyReclaimEnd(std::max(length - 1, 0)), block_costs.size());
        for (; retired < end; ++retired) {
            physical_blocks -= std::max(block_costs[retired], 0);
            block_costs[retired] = 0;
        }
        const int total_slots = needBlocksNum(length, 0, reserve_step);
        if (static_cast<int>(block_costs.size()) < total_slots) {
            block_costs.resize(total_slots, 0);
        }
        for (size_t pos = retired; pos < static_cast<size_t>(total_slots); ++pos) {
            if (block_costs[pos] == 0
                && shouldMaterializeBlock(static_cast<int>(pos), length, reserve_step, enable_reuse_cache)) {
                block_costs[pos] = 1;
                ++physical_blocks;
            }
        }
        peak_blocks = std::max(peak_blocks, physical_blocks);
    };
    if (initialized) {
        allocate_at(seq_len);
    }
    // A fresh prompt's complete bulk allocation has already been charged above.
    // Allocation and committed-retirement boundaries change only at a new slot.
    for (int64_t length = static_cast<int64_t>(seq_len) + 1; length <= final_len;) {
        allocate_at(static_cast<int>(length));
        length = ((length + block_size - 1) / block_size) * block_size + 1;
    }
    return peak_blocks;
}

NeedBlocksInfo LinearCacheManager::getNeedBlocks(int                      common_seq_len,
                                                 int                      seq_len,
                                                 int                      reserve_step,
                                                 int                      reuse_blocks_len,
                                                 bool                     reuse_enabled,
                                                 const RequiredPositions& required_positions,
                                                 size_t                   reclaimed_prefix_blocks) const {
    (void)reserve_step;
    NeedBlocksInfo info;
    const int      reuse_slots  = reuse_enabled ? reuse_blocks_len : 0;
    const int      common_slots = needBlocksNum(common_seq_len, 0);
    const int      total_slots  = needBlocksNum(seq_len, 0);
    // History-reclaim gating: slots the runtime already retired below the
    // committed prefix never require fresh physical blocks; explicitly
    // required positions (pending loads / transfers) still count.
    const auto reclaimed = [&](int pos) {
        return static_cast<size_t>(pos) >= reclaimed_prefix_blocks;
    };

    for (int pos = 0; pos < common_slots; ++pos) {
        if ((reclaimed(pos) && pos >= reuse_slots && shouldMaterializeBlock(pos, common_seq_len, 0, reuse_enabled))
            || required_positions.count(static_cast<size_t>(pos)) != 0) {
            ++info.common_blocks;
        }
    }
    for (int pos = common_slots; pos < total_slots; ++pos) {
        if ((reclaimed(pos) && pos >= reuse_slots && shouldMaterializeBlock(pos, seq_len, 0, reuse_enabled))
            || required_positions.count(static_cast<size_t>(pos)) != 0) {
            ++info.extra_blocks;
        }
    }
    return info;
}

bool LinearCacheManager::malloc(BlockIds&                block_ids,
                                int                      seq_len,
                                bool                     enable_reuse_cache,
                                int                      reserve_step,
                                std::vector<size_t>*     backfilled_positions,
                                const RequiredPositions& required_positions) {
    (void)reserve_step;
    return mallocImpl(block_ids, seq_len, 0, true, enable_reuse_cache, backfilled_positions, required_positions);
}

bool LinearCacheManager::mallocForDecode(BlockIds&            block_ids,
                                         int                  seq_len,
                                         bool                 enable_reuse_cache,
                                         int                  reserve_step,
                                         std::vector<size_t>* backfilled_positions) {
    return mallocImpl(block_ids, seq_len, reserve_step, false, enable_reuse_cache, backfilled_positions, {});
}

bool LinearCacheManager::mallocImpl(BlockIds&                block_ids,
                                    int                      seq_len,
                                    int                      reserve_step,
                                    bool                     is_prefill,
                                    bool                     enable_reuse_cache,
                                    std::vector<size_t>*     backfilled_positions,
                                    const RequiredPositions& required_positions) {
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
    const int current_blocks_len = static_cast<int>(block_ids.blocksNum());
    const int total_slots        = needBlocksNum(seq_len, 0, reserve_step);
    const int new_blocks_len     = std::max(total_slots - current_blocks_len, 0);

    const size_t reclaimed      = block_ids.reclaimedPrefixBlocks();
    const int    sequence_slots = needBlocksNum(seq_len, 0, 0);
    if ((reclaimed > 0 && sequence_slots <= static_cast<int>(reclaimed))
        || std::any_of(required_positions.begin(), required_positions.end(), [reclaimed](size_t pos) {
               return pos < reclaimed;
           })) {
        // Reallocating an old read slot would expose an uninitialized state.
        return false;
    }
    auto should_materialize = [&](int pos) {
        // History-reclaim: slots retired below the committed prefix never
        // re-materialize (required positions were prechecked to be above it).
        if (static_cast<size_t>(pos) < reclaimed) {
            return false;
        }
        if (required_positions.count(static_cast<size_t>(pos)) != 0) {
            return true;
        }
        if (is_prefill) {
            return pos >= current_blocks_len && shouldMaterializeBlock(pos, seq_len, 0, enable_reuse_cache);
        }
        const int seq_slots = needBlocksNum(seq_len, 0);
        return pos >= std::max(0, seq_slots - std::max(1, static_cast<int>(activeTailBlocks())));
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
                RTP_LLM_LOG_WARNING("Insufficient free blocks for LinearCacheManager: need %d, have %zu",
                                    need_alloc_blocks,
                                    free_blocks_num);
                return false;
            }
        }
    }

    BlockIndicesType allocated_blocks;
    if (need_alloc_blocks > 0) {
        auto allocated = block_pool_->malloc(static_cast<size_t>(need_alloc_blocks));
        if (!allocated.has_value() || allocated->size() != static_cast<size_t>(need_alloc_blocks)) {
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

void LinearCacheManager::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    const auto& block_indices = block_ids.blocks();
    if (block_indices.empty()) {
        return;
    }
    (void)enable_reuse_cache;
    const int block_size = static_cast<int>(block_indices.size());

    BlockIndicesType    blocks_to_free;
    std::vector<size_t> pos_to_remove;
    for (int i = block_size - std::min(retainedDecodeBlocks(reserve_step), block_size) - 1; i >= 0; i--) {
        if (isNullBlockIdx(block_indices[static_cast<size_t>(i)])) {
            continue;
        }
        blocks_to_free.push_back(block_indices[static_cast<size_t>(i)]);
        pos_to_remove.push_back(static_cast<size_t>(i));
    }
    if (!blocks_to_free.empty()) {
        block_pool_->decRef(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
}

}  // namespace rtp_llm
