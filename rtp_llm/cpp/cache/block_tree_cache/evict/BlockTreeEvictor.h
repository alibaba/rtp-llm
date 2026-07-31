#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

struct EvictionReleaseCredit {
    DeviceBlockPoolPtr pool;
    BlockIdxType       block{NULL_BLOCK_IDX};
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
    using ExecuteTransferFn = std::function<bool(const TransferDescriptor&)>;
    using IsTierEnabledFn   = std::function<bool(Tier)>;
    using CreditsFn         = std::function<void(const std::vector<EvictionReleaseCredit>&)>;
    using SettledFn         = std::function<void(bool tree_data_mutated, bool check_watermark)>;
    using RemoteWriteFn     = std::function<void(CacheKeyType cache_key, size_t group_set_id)>;

    struct EvictionPlan {
        TransferDescriptor              primary_desc;
        std::vector<TransferDescriptor> cascade_descs;

        bool needsCopy() const;
        bool empty() const {
            return primary_desc.node == nullptr;
        }
    };

    struct CopyResultSet {
        bool              primary_success{false};
        std::vector<bool> cascade_success;
    };

    BlockTreeEvictor(BlockTree*                     tree,
                     ExecuteTransferFn              execute_transfer,
                     bool                           enable_reverse_eviction,
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

    void init(EvictionPolicy device_policy, EvictionPolicy host_policy, EvictionPolicy disk_policy);

    // ---- Semantic events (must be called while holding BlockTreeCache mutex) ----
    // Initialize candidate meta for new nodes and refresh their candidacy.
    void onInsertCommitted(const BlockTreeInsertResult& result);
    // A real match hit: bump access clock / hit_count and re-sort in-heap entries.
    void onMatched(const std::vector<TreeNode*>& path);
    // A match-protection reference was released: re-evaluate candidacy (lazy ref).
    void refreshCandidatesAfterRelease(const MultiNodeResource& set);
    // A node's topology changed (e.g. became a leaf after child deletion).
    void onTopologyChanged(TreeNode* parent);
    CandidateStats         candidateStats() const;
    size_t                 candidateCount(size_t group_set_id, Tier tier) const;
    std::vector<TreeNode*> candidateNodes(size_t group_set_id, Tier tier) const;

    // ---- Eviction selection & migration (caller owns synchronization) ----
    // Selection, prepare, finish, and rollback mutate tree/group-set/pool/heap state
    // and must run under BlockTreeCache's mutex. Task execution is lock-free.
    std::optional<TransferDescriptor> chooseVictim(Tier tier);
    std::optional<TransferDescriptor> chooseVictim(size_t group_set_id, Tier tier);
    std::vector<TransferDescriptor>   chooseWatermarkVictims(GroupSet& group_set, Tier tier, double watermark_ratio);
    std::optional<EvictionPlan>       buildPlan(TransferDescriptor eviction_desc);
    bool submitLocked(TransferDescriptor& eviction_desc, std::vector<EvictionReleaseCredit>* release_credits = nullptr);
    void complete(const EvictionPlan& plan, const CopyResultSet& results);
    void rollbackPreparedPlan(const EvictionPlan& plan);
    void writeRemoteThrough(const std::shared_ptr<StorageBackend>& storage_backend,
                            CacheKeyType                           cache_key,
                            size_t                                 group_set_id);

    EvictionTaskRunner&       taskRunner();
    const EvictionTaskRunner& taskRunner() const;

    // Refresh one node after an external owner changes its transfer state.
    void refreshCandidate(TreeNode* node, size_t group_set_id);
    // Record successful entry into a tier and refresh eviction candidacy.
    void onTierEntered(TreeNode* node, size_t group_set_id, Tier tier);

private:
    struct GroupSetTierHeaps {
        std::unique_ptr<EvictionHeap> device;
        std::unique_ptr<EvictionHeap> host;
        std::unique_ptr<EvictionHeap> disk;
    };

    EvictionHeap*       heapFor(size_t group_set_id, Tier tier);
    const EvictionHeap* heapFor(size_t group_set_id, Tier tier) const;
    // The single candidate-eligibility gate (design section 4.3). Upserts the
    // node when ready, erases it otherwise. Idempotent.
    void refreshCandidate(GroupSet& group_set, TreeNode* node, Tier tier);
    bool isEvictable(const GroupSet& group_set, const TreeNode* node, Tier tier) const;

    std::optional<TransferDescriptor> chooseVictimInGroupSet(GroupSet& group_set, Tier tier);
    static Tier                 defaultTargetTier(Tier source);

    TransferDescriptor  makeDesc(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) const;
    std::vector<size_t> selectCascadeGroupSets(const TreeNode* node,
                                               size_t          source_group_set_id,
                                               Tier            tier,
                                               bool            enable_reverse_eviction) const;
    bool                prepareDesc(TransferDescriptor& eviction_desc);
    void                reserveSource(const TransferDescriptor& eviction_desc);
    bool                restoreSource(const TransferDescriptor& eviction_desc);
    void                releaseTargetBlocks(const TransferDescriptor& eviction_desc);
    bool                applyDescCompletion(const GroupSetPtr& group_set, const TransferDescriptor& eviction_desc);
    void                finalizeEviction(TreeNode* node);
    size_t              computeGroupSetExcess(const GroupSet& group_set, Tier tier, double ratio) const;

    BlockTree*                          tree_;
    std::unique_ptr<EvictionTaskRunner> task_runner_;
    bool                                enable_reverse_eviction_{false};

    // Heap ownership: vector index is the declared group_set_id.
    std::vector<GroupSetTierHeaps> heaps_;
    // Process-local logical clocks (read/written only under the cache mutex).
    uint64_t access_seq_{0};
    uint64_t admission_seq_{0};
};

}  // namespace rtp_llm
