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

}  // namespace rtp_llm
