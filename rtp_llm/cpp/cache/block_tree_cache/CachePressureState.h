#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"

namespace rtp_llm {

struct PoolEvidence {
    uint64_t published_host_blocks{0};
};

struct TransferEvidence {
    uint64_t copied_blocks{0}, copied_bytes{0};
    uint64_t published_blocks{0}, published_bytes{0}, failed_operations{0}, physical_failed_operations{0};
};

using PressureOwner       = std::pair<size_t, int64_t>;       // group, node cache key
using PressureBlock       = std::pair<std::string, int64_t>;  // physical pool, block
using PressureTransferKey = std::tuple<size_t, std::string, CacheTransferOperation, Tier, Tier>;

// Ownership mutations require the cache mutex. Lock order is cache mutex then
// ledger mutex, never the reverse. Copy callbacks and readers take only mutex_.
// Callers must use configured group/pool IDs, never arbitrary request strings:
// cumulative maps are bounded by topology, operation and tier combinations.
// Resident owner/per-block entries are erased when their last owner is removed.
// No external metrics reporter instance is required for evidence collection.
class CachePressureState {
public:
    void         setHostOwner(PressureOwner owner, std::optional<PressureBlock> block);
    PoolEvidence pool(const std::string& pool_id) const;
    // blocks/bytes describe successful descriptors, even for a partially failed
    // operation. success describes the operation as a whole.
    void recordCopy(const PressureTransferKey& key, uint64_t blocks, uint64_t bytes, bool success);
    void recordWorkflowFailure(const PressureTransferKey& key);
    // Call once for each accepted settlement transition, never from polling.
    void             recordPublication(const PressureTransferKey& key, uint64_t blocks, uint64_t bytes);
    TransferEvidence transfer(const PressureTransferKey& key) const;
    // DEVICE high-watermark transitions only; HOST/DISK hysteresis stays separate.
    void     recordWatermarkTrigger(size_t group_id);
    uint64_t watermarkTriggers(size_t group_id) const;

private:
    mutable std::mutex                              mutex_;
    std::map<PressureOwner, PressureBlock>          owners_;
    std::map<PressureBlock, uint64_t>               owners_per_block_;
    std::map<std::string, PoolEvidence>             pools_;
    std::map<PressureTransferKey, TransferEvidence> transfers_;
    std::map<size_t, uint64_t>                      triggers_;
};

}  // namespace rtp_llm
