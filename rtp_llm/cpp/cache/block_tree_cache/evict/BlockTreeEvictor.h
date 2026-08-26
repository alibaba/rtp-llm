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
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"
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
    using IsTierEnabledFn = std::function<bool(Tier)>;
    using SettledFn       = std::function<void(bool tree_data_mutated, bool check_watermark)>;

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
                     SettledFn                      settled);
    ~BlockTreeEvictor();

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    // Initialize candidate metadata after an insert.
    void onInserted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void                   onMatched(const std::vector<TreeNode*>& path);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, task preparation, settlement, and abort mutate tree/group-set/pool/heap
    // state and must run under BlockTreeCache's mutex. Task execution is lock-free.
    bool evictLocked(size_t group_set_id, Tier source_tier, bool force_drop);
    void scheduleWatermarkEvictionsLocked(Tier tier, double watermark_ratio);
    // Discard a detached operation's source without publishing its target.
    void discardDetachedTransfer(const TransferDescriptor& transfer_desc);

    // Exact candidate updates for callers that already know the affected tier.
    void suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier);
    void admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier);
    // Refresh candidate metadata after a successful load.
    void onLoaded(TreeNode* node, size_t group_set_id);
    // Refresh the affected FULL parent after a resource changes tier.
    void onTierChanged(TreeNode* node, size_t group_set_id);

private:
    // A node's topology changed (e.g. became a leaf after child deletion).
    void onTopologyChanged(TreeNode* parent);

    struct GroupSetTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
    };

    EvictionHeap*                     heapFor(size_t group_set_id, Tier tier) const;
    bool                              isEvictable(TreeNode* node, size_t group_set_id, Tier source_tier) const;
    std::optional<TransferDescriptor> chooseVictim(size_t group_set_id, Tier tier, bool force_drop = false);
    std::optional<EvictionTask>       prepareEvictionLocked(TransferDescriptor eviction_desc);
    void runEvictionTask(std::shared_ptr<const EvictionTask> task) noexcept;
    void scheduleEvictionSettlement(std::shared_ptr<const EvictionTask> task,
                                    EvictionTaskResult                  task_result) noexcept;
    void finalizeEvictionLocked(const EvictionTask& task, const EvictionTaskResult& task_result) noexcept;
    void settleEvictionLocked(const EvictionTask& task, const EvictionTaskResult& task_result);
    void abortEvictionLocked(const EvictionTask& task);
    void updateFullCandidateForTopology(TreeNode* node, size_t group_set_id);

    void   selectCascades(EvictionTask& task, std::vector<std::pair<TreeNode*, size_t>>& detached_resources);
    void   selectUpwardCascades(EvictionTask& task);
    void   collectFullPrune(const TransferDescriptor&                  eviction_desc,
                            EvictionTask&                              task,
                            std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const;
    bool   allocateTargets(EvictionTask& task);
    void   activateTaskLocked(const EvictionTask&                              task,
                              const std::vector<std::pair<TreeNode*, size_t>>& detached_resources);
    void   reserveSource(const TransferDescriptor& eviction_desc);
    void   restoreSource(const TransferDescriptor& eviction_desc);
    void   releaseTargetBlocks(const TransferDescriptor& eviction_desc);
    void   completeEvict(const TransferDescriptor& desc);
    void   settleSingleEviction(TreeNode* node);
    void   settleFullPrune(const EvictionTask& task);
    void   eraseNodeFromAllHeaps(TreeNode* node);
    void   updatePendingReleases(const EvictionTask& task, bool reserve);
    size_t poolWatermarkExcess(IBlockPool* pool, double ratio) const;
    size_t computeGroupSetExcess(const GroupSet& group_set, Tier tier, double ratio) const;

    BlockTree*                          tree_;
    BlockTreeTaskPool*                  task_pool_{nullptr};
    BlockTreeCacheMetricsReporter*      metrics_reporter_{nullptr};
    std::mutex*                         mutex_{nullptr};
    IsTierEnabledFn                     is_tier_enabled_;
    SettledFn                           settled_;
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
