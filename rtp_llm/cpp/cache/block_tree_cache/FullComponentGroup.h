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

    // Full-specific: only DeviceLeaf nodes enter heap.
    void tryAddToDeviceHeap(TreeNode* node) override;
    // Full-specific: only HostLeaf/DiskLeaf nodes enter heap.
    void tryAddToHostHeap(TreeNode* node) override;
    void tryAddToDiskHeap(TreeNode* node) override;
};

class FullMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSlot& slot) override;
};

}  // namespace rtp_llm
