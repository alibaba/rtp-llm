#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "autil/LoopThread.h"
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

class BlockTree;
struct TreeNode;

constexpr size_t kFullPrefixScanNodesPerRound      = 1024;
constexpr size_t kFullPrefixScanNodesPerRoundLimit = 4096;
constexpr size_t kFullPrefixScanMaxDetailsPerCycle = 10;

enum class FullViolationType : uint8_t {
    LOWER_TO_DEVICE,
    GAP_TO_DATA,
    INVALID_RESOURCE,
};

enum class InvalidResourceReason : uint8_t {
    NONE,
    MULTI_TIER,
    PARTIAL_DEVICE,
    IDLE_DETACHED,
    BUSY_EMPTY,
};

const char* fullViolationTypeName(FullViolationType type);
const char* invalidResourceReasonName(InvalidResourceReason reason);

// Value-type snapshot of one node's resource, extracted under the cache mutex so that
// no TreeNode pointer or resource reference survives past the batch.
struct NodeBrief {
    CacheKeyType          cache_key{0};
    Tier                  tier{Tier::NONE};  // getTopTier(); Tier::NONE renders as EMPTY
    GroupSetTransferState transfer_state{GroupSetTransferState::IDLE};
    // Bit i set means hasTier(Tier(i)); only used by the multi-tier invalid_resource report.
    uint8_t tier_mask{0};
};

struct FullViolationDetail {
    size_t            group_set_id{0};
    FullViolationType type{FullViolationType::INVALID_RESOURCE};
    // Only set for INVALID_RESOURCE.
    InvalidResourceReason reason{InvalidResourceReason::NONE};
    bool                  stable{false};
    NodeBrief             parent;
    NodeBrief             current;
};

struct FullPrefixScanOptions {
    int64_t interval_ms{0};  // 0 = disabled, no thread
    int     world_rank{0};
    int     local_rank{0};
    size_t  nodes_per_round{kFullPrefixScanNodesPerRound};
    size_t  max_details_per_cycle{kFullPrefixScanMaxDetailsPerCycle};
};

struct FullPrefixScanStats {
    // Cumulative.
    uint64_t batches_started{0};
    uint64_t cycles_completed{0};
    // Current cycle.
    bool   cycle_active{false};
    size_t nodes_scanned{0};
    size_t stable_violations{0};
    size_t transient_violations{0};
    size_t details_logged{0};
    size_t details_suppressed{0};
};

InvalidResourceReason invalidResourceReason(const GroupSetResource& resource);
NodeBrief             makeNodeBrief(const TreeNode& node, const GroupSetResource& resource);
// Appends every violation observable from `node` alone (plus its parent edge) to `details`.
void detectNodeViolations(const BlockTree& tree, const TreeNode& node, std::vector<FullViolationDetail>& details);

std::string formatViolationDetail(const FullViolationDetail& detail, int world_rank, int local_rank);

// Periodically checks the FULL prefix invariant in bounded batches. Observation only:
// it never mutates the tree, resources, block pools or refcounts.
class FullPrefixInvariantScanner {
public:
    FullPrefixInvariantScanner(const BlockTree& tree, std::mutex& cache_mutex, FullPrefixScanOptions options);
    ~FullPrefixInvariantScanner();

    FullPrefixInvariantScanner(const FullPrefixInvariantScanner&)            = delete;
    FullPrefixInvariantScanner& operator=(const FullPrefixInvariantScanner&) = delete;

    bool                start();
    void                stop();
    void                runOneBatch();
    FullPrefixScanStats stats() const;

private:
    void runBatchGuarded();
    void publishBatch(const std::vector<FullViolationDetail>& details,
                      size_t                                  nodes_scanned,
                      size_t                                  tree_size,
                      bool                                    cycle_complete);

    const BlockTree&            tree_;
    std::mutex&                 cache_mutex_;
    const FullPrefixScanOptions options_;

    // LoopThread::stop() cannot interrupt a batch that is already waiting for the cache
    // mutex, so the batch re-checks this flag once it owns the lock.
    std::atomic<bool>    stopping_{false};
    autil::LoopThreadPtr loop_thread_;

    // Touched only by the thread running a batch.
    size_t cursor_{0};
    size_t cycle_end_index_{0};
    bool   cycle_active_{false};
    size_t cycle_tree_size_{0};
    size_t cycle_nodes_scanned_{0};
    size_t cycle_stable_{0};
    size_t cycle_transient_{0};
    size_t cycle_details_logged_{0};
    size_t cycle_details_suppressed_{0};

    // Mirror of the counters above plus the cumulative totals, for stats().
    mutable std::mutex  stats_mutex_;
    FullPrefixScanStats stats_;
};

}  // namespace rtp_llm
