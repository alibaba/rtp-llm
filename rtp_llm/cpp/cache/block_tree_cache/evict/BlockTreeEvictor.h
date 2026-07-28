#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;
class EvictionTaskRunner;

struct EvictionReleaseCredit {
    DeviceBlockPoolPtr pool;
    BlockIdxType       block{NULL_BLOCK_IDX};
};

// Aggregated candidate counts across all groups, one number per tier.
struct CandidateStats {
    size_t device_candidates{0};
    size_t host_candidates{0};
    size_t disk_candidates{0};
};

// BlockTreeEvictor owns every EvictionHeap and is the only class that mutates
// heap membership. BlockTreeCache reports semantic events; GroupSet only
// provides group-specific evictability and block/slot lifecycle operations.
class BlockTreeEvictor {
public:
    using ExecuteTransferFn = std::function<bool(const TransferDescriptor&)>;
    using IsTierEnabledFn   = std::function<bool(Tier)>;
    using CreditsFn         = std::function<void(const std::vector<EvictionReleaseCredit>&)>;
    using SettledFn         = std::function<void(bool tree_data_mutated, bool check_watermark)>;
    using RemoteWriteFn     = std::function<void(CacheKeyType cache_key, size_t group_set_id)>;

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

    BlockTreeEvictor(std::vector<GroupSetPtr>& group_sets,
                     ExecuteTransferFn         execute_transfer,
                     bool                      enable_reverse_eviction);
    BlockTreeEvictor(std::vector<GroupSetPtr>&      group_sets,
                     ExecuteTransferFn              execute_transfer,
                     bool                           enable_reverse_eviction,
                     BlockTree*                     tree,
                     const BlockTransferDispatcher* transfer_dispatcher,
                     BlockTreeTaskPool*             task_pool,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     std::mutex&                    mutex,
                     int                            memory_timeout_ms,
                     int                            disk_timeout_ms,
                     IsTierEnabledFn                is_tier_enabled,
                     CreditsFn                      reserve_credits,
                     CreditsFn                      settle_credits,
                     SettledFn                      settled,
                     RemoteWriteFn                  remote_write);
    ~BlockTreeEvictor();

    bool init(EvictionPolicy device_policy, EvictionPolicy host_policy, EvictionPolicy disk_policy);

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    // Initialize candidate meta for new nodes and refresh their candidacy.
    void onInsertCommitted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void onMatched(const std::vector<TreeNode*>& path);
    // A match-protection reference was released: re-evaluate candidacy (lazy ref).
    void refreshCandidatesAfterRelease(const MultiNodeResource& set);
    // Re-evaluate candidacy for every tree node across all group sets.
    // Called after external refcount changes (e.g. request free) that may make
    // previously non-evictable blocks evictable.
    void refreshAllCandidates(const BlockTree& tree);
    // A node's topology changed (e.g. became a leaf after child deletion).
    void onTopologyChanged(TreeNode* parent);
    // A node is about to be removed from the tree: drop it from all heaps.
    void                   onNodeAboutToRemove(TreeNode* node);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, prepare, finish, and rollback mutate tree/group/pool/heap state
    // and must run under BlockTreeCache's mutex. Task execution is lock-free.
    std::optional<EvictionMove> chooseVictim(Tier tier);
    std::optional<EvictionMove> chooseVictim(size_t group_set_id, Tier tier);
    std::vector<EvictionMove>   chooseWatermarkVictims(GroupSet& group, Tier tier, double watermark_ratio);
    std::optional<EvictionPlan> buildPlan(EvictionMove eviction_move);
    bool submitLocked(EvictionMove& eviction_move, std::vector<EvictionReleaseCredit>* release_credits = nullptr);
    void                        complete(BlockTree& tree, const EvictionPlan& plan, const CopyResultSet& results);
    void                        rollbackPreparedPlan(const EvictionPlan& plan);
    void                        writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                                                   CacheKeyType                           cache_key,
                                                   size_t                                 group_set_id);

    EvictionTaskRunner&       taskRunner();
    const EvictionTaskRunner& taskRunner() const;

    // Refresh one node after an external owner changes its transfer state.
    void refreshCandidate(TreeNode* node, size_t group_set_id);
    // Record successful entry into a tier and refresh eviction candidacy.
    void onTierEntered(TreeNode* node, size_t group_set_id, Tier tier);

private:
    struct GroupTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
    };

    EvictionHeap*       heapFor(size_t group_set_id, Tier tier);
    const EvictionHeap* heapFor(size_t group_set_id, Tier tier) const;
    // The single candidate-eligibility gate (design section 4.3). Upserts the
    // node when ready, erases it otherwise. Idempotent.
    void refreshCandidate(GroupSet& group, TreeNode* node, Tier tier);

    std::optional<EvictionMove> chooseVictimInGroup(GroupSet& group, Tier tier);
    static Tier                 defaultTargetTier(Tier source);

    EvictionMove        makeMove(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) const;
    std::vector<size_t> selectCascadeGroups(const TreeNode* node,
                                            size_t          source_group_set_id,
                                            Tier            tier,
                                            bool            enable_reverse_eviction) const;
    bool                prepareMove(EvictionMove& eviction_move);
    void                reserveSource(const EvictionMove& eviction_move);
    bool                restoreSource(const EvictionMove& eviction_move);
    void                releaseTargetBlocks(const EvictionMove& eviction_move);
    bool                applyMoveCompletion(GroupSetPtr& group, const EvictionMove& move);
    void                finalizeEviction(BlockTree& tree, TreeNode* node);
    bool                shouldDeleteNode(const BlockTree& tree, const TreeNode* node) const;
    std::vector<size_t> reusableGroupSetIds() const;
    size_t              computeGroupExcess(const GroupSet& group, Tier tier, double ratio) const;

    std::vector<GroupSetPtr>&           group_sets_;
    std::unique_ptr<EvictionTaskRunner> task_runner_;
    bool                                enable_reverse_eviction_{false};

    // Heap ownership: vector index is the declared group_set_id.
    std::vector<GroupTierHeaps> heaps_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
