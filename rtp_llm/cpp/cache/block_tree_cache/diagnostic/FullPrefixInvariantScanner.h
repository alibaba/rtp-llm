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

enum class FullViolationType : uint8_t {
    LOWER_TO_DEVICE,
    GAP_TO_DATA,
    INVALID_RESOURCE,
};

enum class InvalidResourceReason : uint8_t {
    NONE,
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
    // Bit i set means hasTier(Tier(i)); included in invalid-resource diagnostics.
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

// Periodically checks the FULL prefix invariant in bounded batches. Observation only:
// it never mutates the tree, resources, block pools or refcounts.
class FullPrefixInvariantScanner {
public:
    FullPrefixInvariantScanner(const BlockTree& tree, std::mutex& cache_mutex, int64_t interval_ms);
    ~FullPrefixInvariantScanner();

    FullPrefixInvariantScanner(const FullPrefixInvariantScanner&)            = delete;
    FullPrefixInvariantScanner& operator=(const FullPrefixInvariantScanner&) = delete;

    bool start();
    void stop();

private:
    void runBatchGuarded();
    void runBatch();
    void publishBatch(const std::vector<FullViolationDetail>& details,
                      size_t                                  nodes_scanned,
                      size_t                                  tree_size,
                      bool                                    cycle_complete);

    const BlockTree& tree_;
    std::mutex&      cache_mutex_;
    const int64_t    interval_ms_;

    // LoopThread::stop() cannot interrupt a batch that is already waiting for the cache
    // mutex, so the batch re-checks this flag once it owns the lock.
    std::atomic<bool>    stopping_{false};
    autil::LoopThreadPtr loop_thread_;

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
