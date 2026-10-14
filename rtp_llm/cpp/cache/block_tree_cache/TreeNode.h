#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

struct TreeNode {
    CacheKeyType                                cache_key{0};
    size_t                                      index{std::numeric_limits<size_t>::max()};
    std::vector<int>                            token_ids;
    std::unordered_map<CacheKeyType, TreeNode*> children;
    TreeNode*                                   parent{nullptr};
    std::vector<GroupSetResource>               group_set_resources;
    bool                                        is_resident{false};  // Must never change from true to false.
    // Monotonic observability timestamps. A node can outlive its usable cache state.
    int64_t attached_at_us{0};
    int64_t cache_live_since_us{0};
};

}  // namespace rtp_llm
