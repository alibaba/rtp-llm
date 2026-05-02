#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

bool dsv4DebugReuseEnabled() {
    return std::getenv("DSV4_DEBUG_REUSE") != nullptr;
}

std::string blockIdsDebugString(const BlockIndicesType& blocks) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << blocks[i];
    }
    oss << "]";
    return oss.str();
}

}  // namespace

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

NeedBlocksInfo SWAKVCacheGroup::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)reuse_enabled;
    NeedBlocksInfo info;
    const int      reuse_begin  = std::max(reuse_blocks_len, 0);
    const int      common_slots = needBlocksNum(common_seq_len, /*current_blocks=*/0);
    const int      total_slots  = needBlocksNum(seq_len, /*current_blocks=*/0, reserve_step);

    info.common_blocks = countTailAllocations(reuse_begin, common_slots, common_slots);
    info.extra_blocks  = countTailAllocations(common_slots, total_slots, total_slots);
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
    if (dsv4DebugReuseEnabled()) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa malloc begin gid=%d seq_len=%d reserve_step=%d current=%d new=%d "
                         "before=%s",
                         group_id_,
                         seq_len,
                         reserve_step,
                         current_blocks_len,
                         new_blocks_len,
                         blockIdsDebugString(block_ids.blocks()).c_str());
    }
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
    if (dsv4DebugReuseEnabled()) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa malloc end gid=%d total_slots=%d tail_begin=%d need_alloc=%d "
                         "new_ids=%s after=%s",
                         group_id_,
                         total_slots,
                         tail_begin,
                         need_alloc_blocks,
                         blockIdsDebugString(new_ids).c_str(),
                         blockIdsDebugString(block_ids.blocks()).c_str());
    }
    return true;
}

void SWAKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                      const BlockIndicesType& block_indices,
                                      bool                    is_resident) {
    if (cache_keys.empty() || block_indices.empty()) {
        return;
    }
    const size_t n = std::min(cache_keys.size(), block_indices.size());
    if (dsv4DebugReuseEnabled()) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa insert begin gid=%d keys=%zu blocks=%zu n=%zu is_resident=%d "
                         "block_indices=%s",
                         group_id_,
                         cache_keys.size(),
                         block_indices.size(),
                         n,
                         is_resident,
                         blockIdsDebugString(block_indices).c_str());
    }
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
        if (dsv4DebugReuseEnabled()) {
            RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa insert item gid=%d pos=%zu key=%ld block=%d",
                             group_id_,
                             i,
                             cache_keys[i],
                             b);
        }
        if (block_cache_->put(item)) {
            block_pool_->blockCacheReference(b);
        }
    }
}

void SWAKVCacheGroup::removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache, int reserve_step) {
    (void)enable_reuse_cache;
    (void)reserve_step;
    const auto& block_indices = block_ids.blocks();
    if (dsv4DebugReuseEnabled()) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa removeSkipped begin gid=%d size=%zu blocks=%s",
                         group_id_,
                         block_indices.size(),
                         blockIdsDebugString(block_indices).c_str());
    }
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
        if (dsv4DebugReuseEnabled()) {
            RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa removeSkipped remove gid=%d keep_begin=%zu free=%s",
                             group_id_,
                             keep_begin,
                             blockIdsDebugString(blocks_to_free).c_str());
        }
        block_pool_->requestFree(blocks_to_free);
        block_ids.remove(pos_to_remove);
    }
    if (dsv4DebugReuseEnabled()) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE swa removeSkipped end gid=%d blocks=%s",
                         group_id_,
                         blockIdsDebugString(block_ids.blocks()).c_str());
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
