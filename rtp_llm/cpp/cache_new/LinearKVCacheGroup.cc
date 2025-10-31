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

MatchResult LinearKVCacheGroup::match(CacheKeysType& cache_keys) {
    MatchResult match_result;
    match_result.block_indices.resize(cache_keys.size());

    int pos = cache_keys.size() - 1;
    for (auto it = cache_keys.rbegin(); it != cache_keys.rend(); ++it) {
        auto result = block_cache_->match(*it);
        if (isNullBlockIdx(result.matched_index)) {
            continue;
        }
        match_result.block_indices[pos] = result.matched_index;
        match_result.reuse_length       = pos + 1;
        break;
    }
    return match_result;
}

// 保留一个有效的block即可。优化下效率，及时退出
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

void LinearKVCacheGroup::malloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) {
    // block_indices
    int new_blocks = needBlocksNum(seq_len, block_indices.size());

    auto result = block_pool_->malloc(new_blocks);
    // check aloc error

    // set blocks to position, not append
    for (int i = 0; i < result.size(); i++) {
        block_indices.push_back(result[i]);
    }

    // insert new allocate blocks to block cache?
}

void LinearKVCacheGroup::free(const BlockIndicesType& block_indices) {
    block_pool_->free(block_indices);
}

void LinearKVCacheGroup::insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices, bool is_resident) {
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
