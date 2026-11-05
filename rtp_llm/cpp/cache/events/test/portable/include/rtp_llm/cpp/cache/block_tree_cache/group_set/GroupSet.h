#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

// Only this standalone test's include path selects this pool double. Production
// BlockTree, TreeNode and GroupSetResource are compiled unchanged. Physical pool
// allocation, reference counting and copying are outside this test's scope.
namespace rtp_llm {
struct TreeNode;
using BlockIdList = std::vector<BlockIdxType>;
enum class BlockTreeRefType : uint8_t { CACHE, LOAD, EVICTION, STORE, FROZEN, COUNT };
struct MultiNodeResource {
    size_t group_set_id;
    Tier tier;
    std::vector<std::pair<TreeNode*, std::vector<BlockIdxType>>> node_blocks;
};

class GroupSet {
public:
    explicit GroupSet(std::vector<std::string> tags): tags_(std::move(tags)) {}
    const std::vector<std::string>& groupTags() const { return tags_; }
    CacheGroupType groupType() const { return CacheGroupType::FULL; }
    void referenceBlocks(const MultiNodeResource&, BlockTreeRefType) const {}
    void unreferenceBlocks(const MultiNodeResource&, BlockTreeRefType) const {}
    void releaseBlocks(Tier, const BlockIdList&, BlockTreeRefType) const {}

private:
    std::vector<std::string> tags_;
};
using GroupSetPtr = std::shared_ptr<GroupSet>;
}  // namespace rtp_llm
