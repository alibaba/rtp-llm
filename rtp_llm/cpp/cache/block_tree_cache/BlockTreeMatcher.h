#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

struct ReusableGroupLocation {
    size_t group_set_id{0};
    size_t member_index{0};
};

using ReusableGroupLocations = std::unordered_map<size_t, ReusableGroupLocation>;

class BlockTreeMatcher {
public:
    BlockTreeMatcher(BlockTree*                    tree,
                     const ReusableGroupLocations& reusable_group_locations,
                     BlockTreeEvictor&             evictor);

    // Locked methods require the caller to hold BlockTreeCache's shared mutex.
    std::pair<BlockTreeMatchResult, std::vector<TreeNode*>> matchLocked(const CacheKeysType& cache_keys);

    void releaseMatchedResourcesLocked(const std::vector<MultiNodeResource>& resources);

    BlockIndicesType matchedBlocksForGroup(size_t                                group_id,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

private:
    void prepareReadyMatchedResourcesLocked(const std::vector<TreeNode*>& matched_path,
                                            const std::vector<bool>&      candidate_valid,
                                            BlockTreeMatchResult&         result);

    BlockTree*                    tree_;
    const ReusableGroupLocations& reusable_group_locations_;
    BlockTreeEvictor&             evictor_;
};

}  // namespace rtp_llm
