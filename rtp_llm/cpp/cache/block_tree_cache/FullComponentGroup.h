#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// FullComponentGroup: manages full KV cache indices.
// Uses Leaf-based heaps: only nodes without children having data at
// the same tier are eligible for eviction.
class FullComponentGroup: public ComponentGroup {
public:
    explicit FullComponentGroup(EvictionPolicy device_policy = EvictionPolicy::LRU,
                                EvictionPolicy host_policy   = EvictionPolicy::LRU,
                                EvictionPolicy disk_policy   = EvictionPolicy::FIFO);

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    void                            finalizeMatchResult(BlockTreeMatchResult& result) override;

    void commitInsertData(TreeNode* node, GroupSlot& slot, const std::vector<BlockIdxType>& block_indices) override;
    void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot) override;

    void                          evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) override;
    std::optional<EvictionResult> driveEviction(int num_blocks, Tier tier) override;

    TransferDescriptor buildTransfer(TreeNode* node, TransferType type) override;

    // Full-specific: only DeviceLeaf nodes enter heap.
    void tryAddToDeviceHeap(TreeNode* node) override;

    bool isDeviceLeaf(const TreeNode* node, int group_id) const;
    bool isHostLeaf(const TreeNode* node, int group_id) const;
    bool isDiskLeaf(const TreeNode* node, int group_id) const;
};

class FullMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSlot& slot) override;
};

}  // namespace rtp_llm
