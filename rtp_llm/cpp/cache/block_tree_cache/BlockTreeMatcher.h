#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTreeMatcher {
public:
    BlockTreeMatcher(BlockTree* tree, BlockTreeEvictor& evictor);

    // Locked methods require the caller to hold BlockTreeCache's shared mutex.
    std::pair<BlockTreeMatchResult, std::vector<TreeNode*>> matchLocked(const CacheKeysType& cache_keys);

    void releaseMatchedResourcesLocked(const std::vector<MultiNodeResource>& resources);

    BlockIndicesType matchedBlocksForGroup(size_t                                group_id,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

private:
    void prepareReadyMatchedResourcesLocked(const std::vector<TreeNode*>& matched_path,
                                            const std::vector<bool>&      candidate_valid,
                                            BlockTreeMatchResult&         result);

    BlockTree*        tree_;
    BlockTreeEvictor& evictor_;
};

}  // namespace rtp_llm
