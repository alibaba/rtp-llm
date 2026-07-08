#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

namespace rtp_llm {

class BlockTreeEvictor {
public:
    struct EvictionPlan {
        EvictionMove            primary;
        std::vector<EvictionMove> cascade_moves;

        bool needsCopy() const;
        bool empty() const {
            return primary.node == nullptr;
        }
    };

    struct CopyResultSet {
        bool              primary_success{false};
        std::vector<bool> cascade_success;
    };

    BlockTreeEvictor(std::vector<ComponentGroupPtr>& component_groups,
                     CopyEnginePtr                   copy_engine,
                     bool                            enable_reverse_eviction);

    void init(const std::vector<Component>& components);

    // The caller owns synchronization. Methods that select, prepare, complete,
    // or rollback eviction mutate tree/group/pool state and must be called while
    // holding BlockTreeCache's tree mutex. performCopy and writeRemoteThrough are
    // lock-free I/O phases.
    std::optional<EvictionMove> chooseVictim(Tier tier);
    std::vector<EvictionMove>   chooseWatermarkVictims(ComponentGroup& group, Tier tier, double watermark_ratio);
    std::optional<EvictionPlan>   buildPlan(EvictionMove er);
    CopyResultSet                 performCopy(const EvictionPlan& plan);
    void                          complete(BlockTree& tree, const EvictionPlan& plan, const CopyResultSet& results);
    void                          rollbackPreparedPlan(const EvictionPlan& plan);
    void                          writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                                                     CacheKeyType                           cache_key,
                                                     int                                    component_group_id);

private:
    void buildGroupLayerTagSlots(const std::vector<Component>& components);

    EvictionMove makeMove(TreeNode* node, int component_group_id, Tier source_tier, Tier target_tier) const;
    std::vector<int> selectCascadeGroups(const TreeNode* node,
                                         int source_group_id,
                                         Tier tier,
                                         bool enable_reverse_eviction) const;
    bool executeTierCopy(const EvictionMove& move);
    void releaseBlocks(int component_group_id, Tier tier, const std::vector<BlockIdxType>& blocks);
    BlockIdxType allocateBlock(int component_group_id, Tier tier);
    bool prepareMove(EvictionMove& move);
    void reserveSourceHeap(const EvictionMove& move);
    void restoreSourceHeap(const EvictionMove& move);
    void releaseTargetBlock(const EvictionMove& move);
    void setTargetSlot(ComponentGroupPtr& group,
                       GroupSlot& slot,
                       TreeNode* node,
                       Tier target_tier,
                       BlockIdxType target_block);
    void finalizeEviction(BlockTree& tree, TreeNode* node);
    bool shouldDeleteNode(const BlockTree&                         tree,
                          const TreeNode*                          node) const;
    std::vector<int> reusableGroupIds() const;
    size_t computeGroupExcess(const ComponentGroup& group, Tier tier, double ratio) const;

    std::shared_ptr<HostBlockPool> hostPoolForGroup(int component_group_id) const;
    std::shared_ptr<DiskBlockPool> diskPoolForGroup(int component_group_id) const;

    std::vector<ComponentGroupPtr>&                         component_groups_;
    CopyEnginePtr                                           copy_engine_;
    bool                                                    enable_reverse_eviction_{false};
    std::vector<std::vector<MemoryBlockLayerTagSlot>>       group_layer_tag_slots_;
    std::vector<MemoryBlockLayerTagSlot>                    empty_layer_tag_slots_;
};

}  // namespace rtp_llm
