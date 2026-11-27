#pragma once

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeKey.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

struct OpaqueTreeNodeState {
    explicit OpaqueTreeNodeState(std::string_view bytes, size_t groups): key(bytes), device_groups(groups) {}
    const std::string key;
    std::vector<bool> device_groups;
};

struct TreeNode {
    CacheKeyType                                cache_key{0};
    std::unique_ptr<OpaqueTreeNodeState>         opaque;
    uint64_t                                    identity{0};
    uint64_t                                    parent_identity{0};
    size_t                                      index{std::numeric_limits<size_t>::max()};
    size_t                                      depth{0};
    std::vector<int>                            token_ids;
    std::unordered_map<BlockTreeKeyView, TreeNode*, BlockTreeKeyHash> children;
    TreeNode*                                   parent{nullptr};
    std::vector<GroupSetResource>               group_set_resources;
    size_t                                      non_owner_transfer_ref_count{0};
    bool                                        is_resident{false};  // Must never change from true to false.

    BlockTreeKeyView key() const {
        return opaque ? BlockTreeKeyView(opaque->key) : BlockTreeKeyView(cache_key);
    }
};

}  // namespace rtp_llm
