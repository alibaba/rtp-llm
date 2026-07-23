#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

// Aggregated candidate counts across all groups, one number per tier.
struct CandidateStats {
    size_t device_candidates{0};
    size_t host_candidates{0};
    size_t disk_candidates{0};
};

// BlockTreeEvictor owns every EvictionHeap and is the only class that mutates
// heap membership. BlockTreeCache reports semantic events; ComponentGroup only
// provides group-specific evictability and block/slot lifecycle operations.
class BlockTreeEvictor {
public:
    using ExecuteTransferFn = std::function<TransferStatus(const TransferDescriptor&)>;

    struct EvictionPlan {
        EvictionMove              primary;
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
                     ExecuteTransferFn               execute_transfer,
                     bool                            enable_reverse_eviction);

    bool init(EvictionPolicy device_policy, EvictionPolicy host_policy, EvictionPolicy disk_policy);

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    // Initialize candidate meta for new nodes and refresh their candidacy.
    void onInsertCommitted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void onMatched(const std::vector<TreeNode*>& path);
    // A match-protection reference was released: re-evaluate candidacy (lazy ref).
    void refreshCandidatesAfterRelease(const GroupBlockSet& set);
    // Re-evaluate candidacy for every tree node across all component groups.
    // Called after external refcount changes (e.g. request free) that may make
    // previously non-evictable blocks evictable.
    void refreshAllCandidates(const BlockTree& tree);
    // A node's topology changed (e.g. became a leaf after child deletion).
    void onTopologyChanged(TreeNode* parent);
    // A node is about to be removed from the tree: drop it from all heaps.
    void           onNodeAboutToRemove(TreeNode* node);
    CandidateStats candidateStats() const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, prepare, finish, and rollback mutate tree/group/pool/heap state
    // and must run under BlockTreeCache's mutex. performCopy is a lock-free phase.
    std::optional<EvictionMove> chooseVictim(Tier tier);
    std::optional<EvictionMove> chooseVictim(int component_group_id, Tier tier);
    std::vector<EvictionMove>   chooseWatermarkVictims(ComponentGroup& group, Tier tier, double watermark_ratio);
    std::optional<EvictionPlan> buildPlan(EvictionMove eviction_move);
    CopyResultSet               performCopy(const EvictionPlan& plan);
    void                        complete(BlockTree& tree, const EvictionPlan& plan, const CopyResultSet& results);
    void                        rollbackPreparedPlan(const EvictionPlan& plan);
    void                        writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                                                   CacheKeyType                           cache_key,
                                                   int                                    component_group_id);
    static bool buildTransferDescriptor(const EvictionMove& eviction_move, TransferDescriptor& descriptor);

    // ---- Load-back state transitions (owned here; driven by BlockTreeCache) ----
    bool reserveLoadBack(TreeNode*                       node,
                         int                             group_id,
                         Tier                            source,
                         const std::vector<BlockIdxType>& source_blocks);
    bool abortPendingLoadBack(TreeNode*                       node,
                              int                             group_id,
                              Tier                            source,
                              const std::vector<BlockIdxType>& source_blocks);
    bool beginLoadBack(TreeNode* node, int group_id, Tier source);
    bool finishLoadBack(TreeNode* node, int group_id, Tier source, bool copy_ok);

private:
    struct GroupTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
    };

    EvictionHeap*       heapFor(int group_id, Tier tier);
    const EvictionHeap* heapFor(int group_id, Tier tier) const;
    // The single candidate-eligibility gate (design section 4.3). Upserts the
    // node when ready, erases it otherwise. Idempotent.
    void refreshCandidate(ComponentGroup& group, TreeNode* node, Tier tier);

    std::optional<EvictionMove> chooseVictimInGroup(ComponentGroup& group, Tier tier);
    static Tier                 defaultTargetTier(Tier source);

    EvictionMove makeMove(TreeNode* node, int component_group_id, Tier source_tier, Tier target_tier) const;
    std::vector<int>
         selectCascadeGroups(const TreeNode* node, int source_group_id, Tier tier, bool enable_reverse_eviction) const;
    bool executeTierCopy(const EvictionMove& eviction_move);
    bool prepareMove(EvictionMove& eviction_move);
    void reserveSource(const EvictionMove& eviction_move);
    bool restoreSource(const EvictionMove& eviction_move);
    void releaseTargetBlocks(const EvictionMove& eviction_move);
    bool applyMoveCompletion(ComponentGroupPtr& group, const EvictionMove& move);
    void finalizeEviction(BlockTree& tree, TreeNode* node);
    bool shouldDeleteNode(const BlockTree& tree, const TreeNode* node) const;
    std::vector<int> reusableGroupIds() const;
    size_t           computeGroupExcess(const ComponentGroup& group, Tier tier, double ratio) const;

    std::vector<ComponentGroupPtr>& component_groups_;
    ExecuteTransferFn               execute_transfer_;
    bool                            enable_reverse_eviction_{false};

    // Heap ownership: vector index is the declared component_group_id. init()
    // rejects null groups and any id/index mismatch by returning false.
    std::vector<GroupTierHeaps> heaps_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
