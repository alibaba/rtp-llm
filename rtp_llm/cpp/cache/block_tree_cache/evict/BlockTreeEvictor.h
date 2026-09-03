#pragma once

#include <array>
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

struct TierWatermark {
    double low_ratio{0.0};
    double high_ratio{0.0};

    bool enabled() const {
        return high_ratio > 0.0;
    }
};

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
                     size_t                         max_device_host_batch,
                     size_t                         max_non_device_host_batch,
                     IsTierEnabledFn                is_tier_enabled,
                     SettledFn                      settled);
    ~BlockTreeEvictor();

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    void onInserted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void                   onMatched(const std::vector<TreeNode*>& path);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, task preparation, settlement, and abort mutate tree/group-set/pool/heap
    // state and must run under BlockTreeCache's mutex. Task execution is lock-free.
    bool   dropLocked(size_t group_set_id, Tier source_tier, bool notify_settled);
    bool   batchEvictLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count);
    void   scheduleWatermarkEvictionsLocked(Tier tier, const TierWatermark& watermark);
    size_t computeWatermarkEvictCount(const GroupSet& group_set, Tier tier, const TierWatermark& watermark);
    // Discard a detached operation's source without publishing its target.
    void discardDetachedTransfer(const std::vector<TransferDescriptor>& transfer_descs);

    // Exact candidate updates for callers that already know the affected tier.
    void suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier);
    void admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier);
    // Refresh candidate metadata and topology after a successful load.
    void onLoaded(TreeNode* node, size_t group_set_id);

private:
    struct GroupSetTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
        std::array<bool, 3>           watermark_triggered{};
    };

    EvictionHeap*                     heapFor(size_t group_set_id, Tier tier) const;
    bool                              isEvictable(TreeNode* node, size_t group_set_id, Tier source_tier) const;
    std::optional<TransferDescriptor> chooseVictim(size_t group_set_id, Tier tier, bool force_drop = false);
    EvictionDropTask                  createDropTask(TransferDescriptor eviction_desc);
    bool                              batchDropLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count);
    bool                              submitEvictionTask(EvictionTransferTask task);
    Tier                              watermarkTargetTier(Tier source_tier) const;
    size_t                            watermarkLogicalBatchLimit(Tier source_tier, Tier target_tier) const;
    void                              runEvictionTask(std::shared_ptr<const EvictionTransferTask> task) noexcept;
    void scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task, bool success) noexcept;
    void runDropTask(TransferDescriptor eviction_desc, bool notify_settled = true);
    void rollbackTransferLocked(const std::vector<TransferDescriptor>& descs);
    void updateFullCandidate(TreeNode* node, size_t group_set_id);
    void updateFullCandidate(TreeNode* parent);

    void                            selectUpwardCascades(EvictionDropTask& task);
    void                            collectFullPrune(const TransferDescriptor&                  eviction_desc,
                                                     EvictionDropTask&                          task,
                                                     std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const;
    void                            reserveSource(const std::vector<TransferDescriptor>& eviction_descs);
    std::vector<TransferDescriptor> restoreSource(const std::vector<TransferDescriptor>& eviction_descs);
    void                            releaseTargetBlocks(const std::vector<TransferDescriptor>& descs);
    void                            completeDrop(const TransferDescriptor& desc);
    void                            completeEvict(const std::vector<TransferDescriptor>& descs);
    void                            settleEviction(const std::vector<TransferDescriptor>& descs);
    void                            settleSingleEviction(TreeNode* node);
    void                            eraseNodeFromAllHeaps(TreeNode* node);
    void                            updatePendingRelease(const std::vector<TransferDescriptor>& descs, bool reserve);
    size_t                          computePoolWatermarkRequired(IBlockPool*          pool,
                                                                 size_t               pending_count,
                                                                 const TierWatermark& watermark,
                                                                 bool&                high_reached) const;

    BlockTree*                          tree_;
    BlockTreeTaskPool*                  task_pool_{nullptr};
    BlockTreeCacheMetricsReporter*      metrics_reporter_{nullptr};
    std::mutex*                         mutex_{nullptr};
    IsTierEnabledFn                     is_tier_enabled_;
    SettledFn                           settled_;
    std::unique_ptr<EvictionTaskRunner> task_runner_;
    int                                 disk_timeout_ms_{0};
    size_t                              max_device_host_batch_{8};
    size_t                              max_non_device_host_batch_{16};

    // Heap ownership: vector index is the declared group_set_id.
    std::vector<GroupSetTierHeaps>          heaps_;
    mutable std::mutex                      pending_release_mutex_;
    std::unordered_map<IBlockPool*, size_t> pending_release_counts_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
