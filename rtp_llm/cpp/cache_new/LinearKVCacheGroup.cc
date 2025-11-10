#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/cache_new/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

MatchResult LinearKVCacheGroup::match(const CacheKeysType& cache_keys) {
    MatchResult final_result;
    final_result.block_indices.resize(cache_keys.size(), NULL_BLOCK_IDX);

    int pos = cache_keys.size();
    for (auto it = cache_keys.rbegin(); it != cache_keys.rend(); ++it, pos--) {
        auto result = block_cache_->match(*it);
        if (isNullBlockIdx(result.matched_index)) {
            continue;
        }
        final_result.block_indices[pos - 1] = result.matched_index;
        final_result.reuse_blocks           = pos;
        break;
    }
    final_result.block_indices.resize(pos);

    final_result.reuse_length = final_result.reuse_blocks * seqSizePerBlock();

    return final_result;
}

// TODO, 保留一个有效的block即可。优化下效率，提前退出
void LinearKVCacheGroup::removeSkippedBlocks(BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }

    bool found_valid_block = false;
    for (int i = block_indices.size() - 1; i >= 0; i--) {
        if (!isNullBlockIdx(block_indices[i])) {
            if (!found_valid_block) {
                found_valid_block = true;
            } else {
                BlockIndicesType blocks = {block_indices[i]};
                block_pool_->free(blocks);
                block_indices[i] = NULL_BLOCK_IDX;
            }
        }
    }
}

// not reuse cache
// reuse cache
// prefill
// decode
int LinearKVCacheGroup::needBlocksNum(int seq_len, int current_blocks) const {
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

bool LinearKVCacheGroup::malloc(const CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) {
    int new_blocks = needBlocksNum(seq_len, block_indices.size());
    if (new_blocks == 0) {
        return true;
    }

    auto result = block_pool_->malloc(new_blocks);
    if (result.empty()) {
        return false;
    }

    // TODO, set blocks to position, not append
    for (int i = 0; i < result.size(); i++) {
        block_indices.push_back(result[i]);
    }

    // TODO, insert new allocate blocks to block cache?

    return true;
}

void LinearKVCacheGroup::reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) {
    block_indices.insert(block_indices.end(), new_block_indices.begin(), new_block_indices.end());
    block_pool_->reference(new_block_indices);
}

void LinearKVCacheGroup::free(const BlockIndicesType& block_indices) {
    block_pool_->free(block_indices);
}

void LinearKVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                         const BlockIndicesType& block_indices,
                                         bool                    is_resident) {
    RTP_LLM_CHECK_WITH_INFO(cache_keys.size() == block_indices.size(),
                            "cache keys size is " + std::to_string(cache_keys.size()) + ", block indices size is "
                                + std::to_string(block_indices.size()));

    for (int i = 0; i < cache_keys.size(); i++) {
        BlockCacheV1::CacheItem item{cache_keys[i], block_indices[i], is_resident};
        if (block_cache_->put(item)) {
            block_pool_->incrBlockRefCounter({block_indices[i]});
        }
    }
}

}  // namespace rtp_llm
