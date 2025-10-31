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

MatchResult LinearKVCacheGroup::match(vector<CacheKeyType>& cache_keys) {
    MatchResult match_result;
    match_result.block_indices.resize(1);
    match_result.block_indices[0].resize(cache_keys.size());

    int pos = cache_keys.size() - 1;
    for (auto it = cache_keys.rbegin(); it != cache_keys.rend(); ++it) {
        auto result = block_cache_->match();
        if (isNullBlockIdx(result.matched_index)) {
            continue;
        }
        match_result.block_indices[0][pos] = result.matched_index;
        match_result.reuse_length          = pos + 1;
        break;
    }
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
                block_pool_->free({block_indices[i]});
            }
        }
    }
}

// not reuse cache
// reuse cache
// prefill
// decode
int LinearKVCacheGroup::newBlocks(int seq_len, int current_blocks) const {
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

void LinearKVCacheGroup::alloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) {
    // block_indices
    int new_blocks = newBlocks(seq_len, block_indices.size());

    auto result = block_pool_->alloc(new_blocks);
    // check aloc error

    // set blocks to position, not append
    for (int i = 0; i < result.size(); i++) {
        block_indices.append(result[i]);
    }

    // insert new allocate blocks to block cache?
}

void LinearKVCacheGroup::free(BlockIndicesType& block_indices) {
    for (int i = 0; i < cache_keys.size(); i++) {
        block_pool_->free({block_indices[i]});
    }
    block_indices.clear();
}

void LinearKVCacheGroup::insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices) {
    RTP_LLM_CHECK_WITH_INFO(cache_keys.size() == block_indices.size(),
                            "cache keys size is " + std::to_string(cache_keys.size()) + ", block indices size is "
                                + std::to_string(block_indices.size()));

    for (int i = 0; i < cache_keys.size(); i++) {
        CacheItem item{cache_keys[i], block_indices[i], {}, false};
        if (block_cache_->put(item)) {
            block_pool_->incrBlockRefCounter({block_indices[i]});
        }
    }
}

}  // namespace rtp_llm
