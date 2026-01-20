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
    // For linear-attn layers, correctness generally requires that the block map has enough slots.
    // We allocate full block count (like full attention) and then allow allocator to free skipped blocks
    // after writeback (prefill) or after crossing boundary (decode).
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
    // do not need LinearKVCacheGroup::match for linear attention
    // // return all matched blocks
    // MatchResult final_result;
    // final_result.block_indices.reserve(cache_keys.size());

    // for (const auto& cache_key : cache_keys) {
    //     auto matched = block_cache_->match(cache_key, group_id_);
    //     if (isNullBlockIdx(matched.matched_index)) {
    //         continue;
    //     }
    //     // TODOL need to return cache_key and block_index pairs
    //     final_result.block_indices.push_back(matched.matched_index);
    // }
    // return final_result;
}

bool LinearKVCacheGroup::malloc(BlockIndicesType& block_indices, int seq_len) {
    // LinearKVCacheGroup::malloc is responsible for:
    // 1. allocating blocks for the current sequence length;
    // 2. free unused blocks to reduce kvcache block usage;

    // Two policies to follow:
    // 1. Linear Steps: keep N * linear_step blocks if cache reuse enabled;
    // 2. Allocate Tail Blocks: allocate the last partial block when initialization and keep last 2 block during
    // decoding;

    // TODO(chanyin): add a flag to judge if malloc is for init or incr
    int current_blocks_len = block_indices.size();
    int new_blocks_len     = needBlocksNum(seq_len, block_indices.size());

    for (int i = current_blocks_len; i < current_blocks_len + new_blocks_len; i++) {
        if (i % linear_step_ == 0 || i == current_blocks_len + new_blocks_len - 1) {
            auto result = block_pool_->malloc(1);
            if (result.empty()) {
                return false;
            }
            block_indices.push_back(result[0]);
        } else {
            block_indices.push_back(NULL_BLOCK_IDX);
        }
    }
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

    // Always keep the last 2 valid blocks (decode edge case: read block i, write block i+1).
    int                        keep_needed = 2;
    std::unordered_set<size_t> keep_pos;
    keep_pos.reserve(2);

    for (size_t pos = block_indices.size(); pos-- > 0;) {
        if (!isNullBlockIdx(block_indices[pos])) {
            keep_pos.insert(pos);
            if (--keep_needed == 0) {
                break;
            }
        }
    }

    BlockIndicesType to_free;
    to_free.reserve(block_indices.size());
    for (size_t i = 0; i < block_indices.size(); ++i) {
        auto b = block_indices[i];
        if (isNullBlockIdx(b)) {
            continue;
        }
        if (keep_pos.find(i) == keep_pos.end()) {
            to_free.push_back(b);
            block_indices[i] = NULL_BLOCK_IDX;
        }
    }

    if (!to_free.empty()) {
        block_pool_->requestFree(to_free);
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
