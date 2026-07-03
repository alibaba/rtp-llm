#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// LinearComponentGroup: manages linear attention / SSM hidden states.
// Point-state: only specific nodes hold state snapshots.
// Uses Any-node heaps like SWA.
class LinearComponentGroup: public ComponentGroup {
public:
    explicit LinearComponentGroup(CacheReusePolicy reuse         = CacheReusePolicy::REUSABLE,
                                  EvictionPolicy   device_policy = EvictionPolicy::LRU,
                                  EvictionPolicy   host_policy   = EvictionPolicy::LRU,
                                  EvictionPolicy   disk_policy   = EvictionPolicy::FIFO);

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    void                            finalizeMatchResult(BlockTreeMatchResult& result) override;

    void commitInsertData(TreeNode* node, GroupSlot& slot, const std::vector<BlockIdxType>& block_indices) override;
    void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot) override;

    void                          evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) override;
    std::optional<EvictionResult> driveEviction(int num_blocks, Tier tier) override;

    TransferDescriptor buildTransfer(TreeNode* node, TransferType type) override;

    void tryAddToDeviceHeap(TreeNode* node) override;
};

class LinearMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSlot& slot) override;
};

}  // namespace rtp_llm
