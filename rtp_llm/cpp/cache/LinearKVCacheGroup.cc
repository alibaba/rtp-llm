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

int LinearKVCacheGroup::needBlocksNum(int seq_len, int current_blocks) const {
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
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
    return {};
}

bool LinearKVCacheGroup::malloc(BlockIndicesType& block_indices, int seq_len) {
    return malloc(block_indices, seq_len, /*enable_reuse_cache=*/true);
}

bool LinearKVCacheGroup::malloc(BlockIndicesType& block_indices, int seq_len, bool enable_reuse_cache) {
    // LinearKVCacheGroup::malloc is responsible for:
    // 1. allocating blocks for the current sequence length;
    // 2. free unused blocks to reduce kvcache block usage;

    // Two policies to follow:
    // 1. Linear Steps: keep N * linear_step blocks if cache reuse enabled;
    // 2. Allocate Tail Blocks: allocate the last partial block when initialization and keep last 2 block during
    // decoding;
    const int step               = std::max(1, linear_step_);
    const int current_blocks_len = static_cast<int>(block_indices.size());
    const int new_blocks_len     = needBlocksNum(seq_len, static_cast<int>(block_indices.size()));

    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        const bool is_tail      = (i == current_blocks_len + new_blocks_len - 1);
        const bool step_hit     = (((i + 1) % step) == 0);
        const bool should_alloc = enable_reuse_cache ? (step_hit || is_tail) : is_tail;
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

void LinearKVCacheGroup::removeSkippedBlocks(BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }
    const int step = std::max(1, linear_step_);
    // TODO(chanyin): avoid traversing the block_indices array in reverse order
    // keep the last 2 blocks and every N * linear_step blocks
    for (int i = block_indices.size() - 3; i >= 0; i--) {
        if (((i + 1) % step) == 0 || isNullBlockIdx(block_indices[i])) {
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
