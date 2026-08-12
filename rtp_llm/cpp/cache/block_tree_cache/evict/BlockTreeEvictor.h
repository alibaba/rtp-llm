#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;
class EvictionTaskRunner;

// Aggregated candidate counts across all group sets, one number per tier.
struct CandidateStats {
    size_t device_candidates{0};
    size_t host_candidates{0};
    size_t disk_candidates{0};
};

// BlockTreeEvictor owns every EvictionHeap and is the only class that mutates
// heap membership. BlockTreeCache reports semantic events; GroupSet only
// provides group-set-specific evictability and resource lifecycle operations.
class BlockTreeEvictor {
public:
    using IsTierEnabledFn   = std::function<bool(Tier)>;
    using SettledFn         = std::function<void(bool tree_data_mutated, bool check_watermark)>;
    using RemoteWriteFn     = std::function<void(CacheKeyType cache_key, size_t group_set_id)>;

    struct EvictionTimingSnapshot {
        int64_t tier_enter_time_us{0};
        int64_t insert_time_us{0};
        int64_t last_access_time_us{0};
        int64_t selected_time_us{0};
    };

    struct EvictionPlan {
        TransferDescriptor                   primary_desc;
        EvictionTimingSnapshot               primary_timing;
        std::vector<TransferDescriptor>      cascade_descs;
        std::vector<EvictionTimingSnapshot>  cascade_timings;
        // FULL prune closure only. Every dependent descriptor targets NONE,
        // and nodes stay valid because such plans are committed synchronously.
        std::vector<TransferDescriptor>     dependent_prune_descs;
        std::vector<EvictionTimingSnapshot> dependent_prune_timings;
        std::vector<TreeNode*>                full_prune_nodes_bottom_up;

        bool needsCopy() const;
        bool empty() const {
            return primary_desc.node == nullptr;
        }
        bool hasFullPruneClosure() const {
            return !full_prune_nodes_bottom_up.empty();
        }
    };

    struct CopyResultSet {
        bool              primary_success{false};
        std::vector<bool> cascade_success;
    };

    BlockTreeEvictor(BlockTree*                     tree,
                     EvictionPolicy                 device_policy,
                     EvictionPolicy                 host_policy,
                     EvictionPolicy                 disk_policy,
                     const BlockTransferDispatcher* transfer_dispatcher,
                     BlockTreeTaskPool*             task_pool,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     std::mutex&                    mutex,
                     int                            memory_timeout_ms,
                     int                            disk_timeout_ms,
                     IsTierEnabledFn                is_tier_enabled,
                     SettledFn                      settled,
                     RemoteWriteFn                  remote_write);
    ~BlockTreeEvictor();

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    // Initialize candidate metadata after an insert.
    void onInserted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void onMatched(const std::vector<TreeNode*>& path);
    // A match-protection reference was released: re-evaluate candidacy (lazy ref).
    void refreshCandidatesAfterRelease(const MultiNodeResource& resource);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, prepare, finish, and rollback mutate tree/group-set/pool/heap state
    // and must run under BlockTreeCache's mutex. Task execution is lock-free.
    std::optional<TransferDescriptor> chooseVictim(size_t group_set_id, Tier tier);
    void                              scheduleWatermarkEvictionsLocked(Tier tier, double watermark_ratio);
    std::optional<EvictionPlan>       buildPlan(TransferDescriptor eviction_desc);
    bool                              submitLocked(TransferDescriptor& eviction_desc);
    void complete(const EvictionPlan& plan, const CopyResultSet& results);
    void rollbackPreparedPlan(const EvictionPlan& plan);
    // Discard a detached operation's source without publishing its target.
    void discardDetachedTransfer(const TransferDescriptor& transfer_desc);
    void writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                            CacheKeyType                           cache_key,
                            size_t                                 group_set_id);

    // Refresh one node after an external owner changes its transfer state.
    void refreshCandidate(TreeNode* node, size_t group_set_id);
    // Refresh candidate metadata after a successful load.
    void onLoaded(TreeNode* node, size_t group_set_id);

private:
    friend class EvictionTaskRunner;

    // A node's topology changed (e.g. became a leaf after child deletion).
    void onTopologyChanged(TreeNode* parent);

    struct FullPruneClosure {
        std::vector<TransferDescriptor>           dependent_descs;
        std::vector<std::pair<TreeNode*, size_t>> detached_resources;
        std::vector<TreeNode*>                    nodes_bottom_up;
    };

    struct GroupSetTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
    };

    EvictionHeap* heapFor(size_t group_set_id, Tier tier) const;
    // The single candidate-eligibility gate (design section 4.3). Upserts the
    // node when ready, erases it otherwise. Idempotent.
    void refreshCandidate(GroupSet& group_set, TreeNode* node, Tier tier);
    bool isEvictable(const GroupSet& group_set, const TreeNode* node, Tier tier) const;

    static Tier defaultTargetTier(Tier source);

    TransferDescriptor  makeDesc(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) const;
    EvictionTimingSnapshot makeTimingSnapshot(const TransferDescriptor& eviction_desc) const;
    std::vector<size_t> selectCascadeGroupSets(const TreeNode* node,
                                               size_t          source_group_set_id,
                                               Tier            tier) const;
    bool                prepareDesc(TransferDescriptor& eviction_desc);
    FullPruneClosure    collectFullPruneClosure(const TransferDescriptor& eviction_desc) const;
    void                reserveSource(const TransferDescriptor& eviction_desc);
    bool                restoreSource(const TransferDescriptor& eviction_desc);
    void                releaseTargetBlocks(const TransferDescriptor& eviction_desc);
    bool                applyDescCompletion(const GroupSetPtr& group_set, const TransferDescriptor& eviction_desc);
    void                finalizeEviction(TreeNode* node);
    void                finalizeFullPrune(const EvictionPlan& plan);
    void                eraseNodeFromAllHeaps(TreeNode* node);
    void                reservePendingReleases(const EvictionPlan& plan);
    void                settlePendingReleases(const EvictionPlan& plan);
    void                updatePendingReleases(const EvictionPlan& plan, bool reserve);
    size_t              poolWatermarkExcess(IBlockPool* pool, double ratio) const;
    size_t              computeGroupSetExcess(const GroupSet& group_set, Tier tier, double ratio) const;

    BlockTree*                          tree_;
    std::unique_ptr<EvictionTaskRunner> task_runner_;

    // Heap ownership: vector index is the declared group_set_id.
    std::vector<GroupSetTierHeaps>          heaps_;
    mutable std::mutex                      pending_release_mutex_;
    std::unordered_map<IBlockPool*, size_t> pending_release_counts_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
