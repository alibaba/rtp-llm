#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

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
};

struct BlockTreeEvictableMetricsSnapshot {
    Tier           tier{Tier::DEVICE};
    CacheGroupType group_type{CacheGroupType::FULL};
    size_t         evictable_blocks{0};
};

class BlockTreeCacheMetricsReporter final {
public:
    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter);

    std::vector<BlockTreePoolMetricsSnapshot>
    collectPoolMetricsSnapshots(const std::vector<ComponentGroupPtr>& component_groups,
                                const BlockTreeEvictor&               evictor) const;
    std::vector<BlockTreeEvictableMetricsSnapshot>
         collectEvictableMetricsSnapshots(const std::vector<ComponentGroupPtr>& component_groups,
                                          const BlockTreeEvictor&               evictor) const;
    void reportEvictableBlockCount(const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const;
    void reportEvictionFinished(const BlockTreeEvictor::EvictionPlan&  plan,
                                const BlockTreeEvictor::CopyResultSet& results,
                                const std::vector<ComponentGroupPtr>&  component_groups) const;

    int64_t reportTransferStarted(Tier source_tier, Tier target_tier);
    void
    reportTransferFinished(Tier source_tier, Tier target_tier, size_t block_count, int64_t begin_time_us, bool success);

private:
    static int transferDirectionIndex(Tier source_tier, Tier target_tier);
    void       reportEvictionMove(const EvictionMove&                   eviction_move,
                                  const std::vector<ComponentGroupPtr>& component_groups,
                                  int64_t                               finish_time_us) const;

    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
    std::array<std::atomic<int64_t>, 4>        transfer_in_flight_{};
};

}  // namespace rtp_llm
