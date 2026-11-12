#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
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

InvalidResourceReason invalidResourceReason(const GroupSetResource& resource);
NodeBrief             makeNodeBrief(const TreeNode& node, const GroupSetResource& resource);
// Appends every violation observable from `node` alone (plus its parent edge) to `details`.
void detectNodeViolations(const BlockTree& tree, const TreeNode& node, std::vector<FullViolationDetail>& details);

std::string formatViolationDetail(const FullViolationDetail& detail);

// Each interval targets 5% of the tree (at least one batch), spreading batches of at
// most 1024 nodes across the interval. Cache lock acquisition retries with sleeps
// for up to 10 ms; a timeout skips the window/batch without advancing the cursor.
// Observation only: it never mutates the tree, resources, block pools or refcounts.
class BlockTreeCacheMetricsReporter;

class FullPrefixInvariantScanner {
public:
    FullPrefixInvariantScanner(const BlockTree&                     tree,
                               std::mutex&                          cache_mutex,
                               int64_t                              interval_ms,
                               const BlockTreeCacheMetricsReporter* metrics_reporter = nullptr);
    ~FullPrefixInvariantScanner();

    FullPrefixInvariantScanner(const FullPrefixInvariantScanner&)            = delete;
    FullPrefixInvariantScanner& operator=(const FullPrefixInvariantScanner&) = delete;

    bool start();
    void stop();

private:
    friend class FullPrefixScannerTest;

    void                         runScanGuarded();
    void                         runScan();
    void                         runBatch();
    CacheLockGuard               tryLockCacheUntil(std::chrono::steady_clock::time_point deadline);
    void                         publishBatch(const std::vector<FullViolationDetail>& details,
                                              size_t                                  nodes_scanned,
                                              size_t                                  tree_size,
                                              bool                                    cycle_complete);

    const BlockTreeCacheMetricsReporter* metrics_reporter_;
    const BlockTree& tree_;
    std::mutex&      cache_mutex_;
    const int64_t    interval_ms_;

    // Cancels scheduled batches and wakes lock-retry/inter-batch waits on shutdown.
    std::atomic<bool>       stopping_{false};
    autil::LoopThreadPtr    loop_thread_;
    std::mutex              wait_mutex_;
    std::condition_variable stop_condition_;

    // Touched only by the thread running a batch.
    size_t   cursor_{0};
    size_t   cycle_end_index_{0};
    bool     cycle_active_{false};
    size_t   cycle_tree_size_{0};
    size_t   cycle_nodes_scanned_{0};
    size_t   cycle_stable_{0};
    size_t   cycle_transient_{0};
    size_t   cycle_details_logged_{0};
    size_t   cycle_details_suppressed_{0};
    uint64_t cycles_completed_{0};
};

}  // namespace rtp_llm
