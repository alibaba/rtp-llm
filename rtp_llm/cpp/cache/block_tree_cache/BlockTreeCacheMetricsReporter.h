#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

struct BlockTreePoolMetricsSnapshot {
    Tier        tier{Tier::DEVICE};
    std::string pool_name;
    size_t      block_size_bytes{0};
    size_t      total_blocks{0};
    size_t      free_blocks{0};
    size_t      available_blocks{0};
    size_t      active_tree_cached_blocks{0};
    size_t      request_ref_count{0};
    size_t      connector_ref_count{0};
    size_t      block_cache_ref_count{0};
    size_t      eviction_ref_count{0};
};

enum class CacheTransferOperation : uint8_t {
    LOAD,
    EVICT,
    STORE,
};

const char* cacheTransferOperationName(CacheTransferOperation operation);

struct BlockTreeEvictableMetricsSnapshot {
    Tier           tier{Tier::DEVICE};
    CacheGroupType group_type{CacheGroupType::FULL};
    size_t         evictable_blocks{0};
};

class BlockTreeCacheMetricsReporter final {
public:
    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter);

    std::vector<BlockTreePoolMetricsSnapshot> collectPoolMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets,
                                                                          const BlockTreeEvictor& evictor) const;
    std::vector<BlockTreeEvictableMetricsSnapshot>
    collectEvictableMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets, const BlockTreeEvictor& evictor) const;
    void reportEvictableBlockCount(const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const;
    void reportEvictionFinished(const BlockTreeEvictor::EvictionPlan&  plan,
                                const BlockTreeEvictor::CopyResultSet& results,
                                const std::vector<GroupSetPtr>&        group_sets) const;

    int64_t reportTransferStarted(CacheTransferOperation operation, Tier source_tier, Tier target_tier);
    void    reportTransferFinished(CacheTransferOperation operation,
                                   Tier                   source_tier,
                                   Tier                   target_tier,
                                   size_t                 block_count,
                                   int64_t                begin_time_us,
                                   bool                   success);
    void    reportStorePublish(Tier target_tier, size_t accepted_blocks, size_t duplicate_blocks) const;

private:
    static constexpr size_t kOperationCount = 3;
    static constexpr size_t kDirectionCount = 5;

    static int transferDirectionIndex(Tier source_tier, Tier target_tier);
    void       reportEvictionTransfer(const TransferDescriptor&       desc,
                                      const std::vector<GroupSetPtr>& group_sets,
                                      int64_t                         finish_time_us) const;
    void       reportStoreBlocks(Tier target_tier, const char* outcome, size_t block_count) const;

    std::shared_ptr<kmonitor::MetricsReporter>                                     metrics_reporter_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
