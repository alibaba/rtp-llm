#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

// Transfer descriptor for data migration between tiers.
// Produced by ComponentGroup::buildTransfer(), consumed by CopyEngine (L1-L3)
// and StorageBackend (L4).
struct TransferDescriptor {
    Tier source_tier{Tier::NONE};
    Tier target_tier{Tier::NONE};
    int  component_group_id{-1};

    std::vector<TreeNode*>                 nodes;
    std::vector<std::vector<BlockIdxType>> source_blocks;
    std::vector<BlockIdxType>              target_blocks;
    std::vector<std::string>               storage_keys;
};

}  // namespace rtp_llm
