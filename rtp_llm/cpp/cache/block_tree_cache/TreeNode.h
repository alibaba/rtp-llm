#pragma once

#include <array>
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
    // Absolute token endpoint. A span shorter than the tree reuse unit is a terminal short leaf.
    size_t                                      token_end{0};
    std::vector<int>                            token_ids;
    std::unordered_map<CacheKeyType, TreeNode*> children;
    TreeNode*                                   parent{nullptr};
    std::vector<GroupSetResource>               group_set_resources;
    bool                                        is_resident{false};  // Must never change from true to false.
    // Prefix-observability intervals. A negative value means the interval is inactive;
    // zero is a valid first-microsecond start.
    struct Observation {
        int64_t                     content_since_us{-1};
        int64_t                     recovery_since_us{-1};
        std::array<int64_t, 3>      tier_since_us{-1, -1, -1};
    };
    Observation                                 observation;
};

}  // namespace rtp_llm
