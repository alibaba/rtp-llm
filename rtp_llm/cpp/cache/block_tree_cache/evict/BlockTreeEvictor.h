#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;
class EvictionTaskRunner;
struct CacheStageContext;

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
    BlockTreeEvictor(BlockTree*                     tree,
                     const BlockTreeCacheConfig&    config,
                     const BlockTransferDispatcher* transfer_dispatcher,
                     BlockTreeTaskPool*             task_pool,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     std::mutex&                    mutex,
                     std::function<void()>          schedule_capacity_check);
    ~BlockTreeEvictor();

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    void                   onInserted(const BlockTreeInsertResult& result);
    uint64_t               nextAccessSeq();
    // A real match hit: re-sort the node in its heap. Access clock / hit_count
    // bookkeeping is owned by the caller (BlockTreeLoader), which also decides
    // which group sets participate (e.g. skip_linear filtering).
    void                   onMatched(TreeNode* node, size_t group_set_id);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;
    // Caller holds the cache mutex. Drain interval counters for reporting outside the lock.
    EvictionVictimMetrics getAndResetEvictionVictimMetrics() const;

    bool capacityCheckEnabled() const;
    void checkCapacity();

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, task preparation, settlement, and abort mutate tree/group-set/pool/heap
    // state and must run under BlockTreeCache's mutex. Task execution is lock-free.
    EvictionBatchResult batchDropLocked(size_t                group_set_id,
                                        Tier                  source_tier,
                                        size_t                required_block_count,
                                        std::optional<size_t> initial_free,
                                        std::optional<size_t> member_group_id);
    // Release a detached operation's source CACHE reference after it settles, without publishing its target.
    void discardDetachedTransfer(DescriptorView transfer_descs);

    // Snapshot before settlement can delete nodes; paths remain valid across cascading drops.
    struct InvalidationSource {
        CacheKeysType cache_keys;
        size_t        group_set_id;
        Tier          source_tier;
    };
    void invalidateSources(const std::vector<InvalidationSource>& sources);

    // Exact candidate updates for callers that already know the affected tier.
    void suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier);
    void admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier);
    // Refresh candidate metadata and topology after a successful load.
    void onLoaded(TreeNode* node, size_t group_set_id);

private:
    enum class DropScope {
        CASCADE,
        PRIMARY_ONLY
    };

    // Reuse only within one eviction pass for the same group, tier and mode,
    // while holding the cache mutex, with only synchronous drops mutating the tree.
    // A new pass must recheck released references and matchability.
    struct EvictionBatchContext {
        TreeNode*                           previous_candidate{nullptr};
        const IBlockPool*                   drop_pool{nullptr};
        bool                                watermark_drop{false};
        std::unordered_set<TreeNode*>       rejected_candidates;
        std::unordered_map<TreeNode*, bool> endpoint_matchable;
    };

    struct GroupSetTierHeaps {
        std::unique_ptr<EvictionHeap>                         device;
        std::unique_ptr<EvictionHeap>                         host;
        std::unique_ptr<EvictionHeap>                                   disk;
        std::map<std::pair<TreeNode*, uint64_t>, EvictionTimingMetrics> pending_timings;
        std::vector<std::chrono::steady_clock::time_point>              next_device_evict_log;
    };

    EvictionHeap* heapFor(size_t group_set_id, Tier tier) const;
    bool          isEvictable(TreeNode* node, size_t group_set_id, Tier source_tier) const;
    std::optional<EvictionVictimSkipReason>
                                      watermarkEvictionSkipReason(TreeNode* node, size_t group_set_id, Tier tier) const;
    std::optional<TransferDescriptor> chooseVictim(size_t                group_set_id,
                                                   Tier                  tier,
                                                   bool                  force_drop             = false,
                                                   EvictionBatchContext* eviction_batch_context = nullptr,
                                                   DropScope             scope                  = DropScope::CASCADE);
    bool                              canDropDeviceBlocks(const TransferDescriptor& desc,
                                                          EvictionBatchContext*     eviction_batch_context,
                                                          DropScope                 scope) const;
    void                              logZeroDeviceEvictLocked(size_t group_set_id,
                                                               size_t member_group_id,
                                                               size_t required_blocks,
                                                               size_t rejected_candidates);

    // Builds the drop closure without reserving or detaching its resources.
    EvictionDropTask planDropTask(TransferDescriptor                         eviction_desc,
                                  std::vector<std::pair<TreeNode*, size_t>>& detached_resources,
                                  EvictionBatchContext*                      eviction_batch_context,
                                  DropScope                                  scope) const;
    EvictionDropTask createDropTask(TransferDescriptor    eviction_desc,
                                    EvictionBatchContext* eviction_batch_context = nullptr,
                                    DropScope             scope                  = DropScope::CASCADE);
    bool submitEvictionTask(std::vector<TransferDescriptor> descriptors, std::vector<EvictionTimingMetrics> timings);
    bool scheduleWatermarkEvictions(Tier tier, std::optional<TierWatermark> watermark = std::nullopt);
    WatermarkEvictionStrategy* watermarkStrategy(Tier tier) const;
    bool                       scheduleWatermarkGroupLocked(size_t                       group_set_id,
                                                            Tier                         tier,
                                                            std::optional<TierWatermark> watermark,
                                                            bool&                        waiting_for_budget);
    size_t                     batchEvictLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count);
    void scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task, bool success);
    bool settleEvictLocked(DescriptorView           descs,
                           bool                     success,
                           size_t&                  cancelled_blocks,
                           std::vector<bool>&       skip_eviction_metrics,
                           const CacheStageContext& stage_context);
    void runDropTask(TransferDescriptor    eviction_desc,
                     EvictionBatchContext* eviction_batch_context = nullptr,
                     DropScope             scope                  = DropScope::CASCADE);
    void rollbackTransferLocked(DescriptorView descs);
    void updateFullCandidate(TreeNode* node, size_t group_set_id);
    void updateFullCandidate(TreeNode* parent);

    void selectUpwardCascades(EvictionDropTask& task, EvictionBatchContext* eviction_batch_context) const;
    void collectFullPrune(const TransferDescriptor&                  eviction_desc,
                          EvictionDropTask&                          task,
                          std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const;
    void restoreSource(DescriptorView eviction_descs);
    void unreferenceTargetBlocks(DescriptorView descs);
    void commitDropLocked(const TransferDescriptor& desc);
    void commitTransferLocked(DescriptorView descs);
    void cleanupTransferNodesLocked(DescriptorView descs);
    void settleSingleEviction(TreeNode* node);
    void eraseNodeFromAllHeaps(TreeNode* node);
    void updatePendingRelease(DescriptorView descs, bool reserve);
    void updatePendingReleaseBlocks(DescriptorView descs, bool reserve, bool revoked);

    BlockTree*                                                tree_;
    const BlockTreeCacheConfig&                               config_;
    BlockTreeTaskPool*                                        task_pool_{nullptr};
    BlockTreeCacheMetricsReporter*                            metrics_reporter_{nullptr};
    std::mutex*                                               mutex_{nullptr};
    std::function<void()>                                     schedule_capacity_check_;
    std::unique_ptr<EvictionTaskRunner>                       task_runner_;
    std::unordered_map<IBlockPool*, size_t>                   pending_release_counts_;
    std::array<std::unique_ptr<WatermarkEvictionStrategy>, 3> watermark_strategies_;
    OnDemandEvictionStrategy                                  on_demand_strategy_;
    std::mutex                                                watermark_mutex_;

    // Heap ownership: vector index is the declared group_set_id.
    std::vector<GroupSetTierHeaps>          heaps_;
    mutable EvictionVictimMetrics           eviction_victim_metrics_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
