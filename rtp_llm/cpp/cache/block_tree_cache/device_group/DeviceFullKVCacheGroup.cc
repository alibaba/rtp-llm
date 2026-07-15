#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceFullKVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int DeviceFullKVCacheGroup::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    return std::max((seq_len + reserve_step + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

NeedBlocksInfo DeviceFullKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    NeedBlocksInfo info;
    const int      common_slots        = needBlocksNum(common_seq_len, /*current_blocks=*/0);
    const int      total_slots         = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step);
    const int      reused_common_slots = reuse_enabled ? std::min(std::max(reuse_blocks_len, 0), common_slots) : 0;
    info.common_blocks                 = std::max(common_slots - reused_common_slots, 0);
    info.extra_blocks                  = std::max(total_slots - common_slots, 0);
    return info;
}

bool DeviceFullKVCacheGroup::malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    int need_blocks_num = needBlocksNum(seq_len, static_cast<int>(block_ids.blocksNum()), reserve_step);
    if (need_blocks_num == 0) {
        return true;
    }
    auto free_blocks_num = freeBlocksNum();
    if (free_blocks_num < need_blocks_num) {
        if (!ensureFreeBlocks(need_blocks_num)) {
            RTP_LLM_LOG_WARNING(
                "Insufficient free blocks for common part: need %d, have %zu", need_blocks_num, free_blocks_num);
            return false;
        }
    }

    auto result = block_pool_->malloc(static_cast<size_t>(need_blocks_num));
    if (!result.has_value() || result->empty()) {
        return false;
    }
    // malloc() only reserves capacity at refCount 0; take the request holder ref to hold the blocks.
    block_pool_->incRef(*result);
    block_ids.add(*result);
    return true;
}

void DeviceFullKVCacheGroup::free(const BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }

    block_pool_->decRef(block_indices);
    RTP_LLM_LOG_DEBUG("Freed %zu blocks", block_indices.size());
}

void DeviceFullKVCacheGroup::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    block_pool_->incRef(new_block_indices);
}

void DeviceFullKVCacheGroup::removeSkippedBlocks(BlockIds& /*block_ids*/,
                                                 bool /*enable_reuse_cache*/,
                                                 int /*reserve_step*/) {}

}  // namespace rtp_llm
