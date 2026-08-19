#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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
    size_t      used_blocks{0};
    size_t      available_blocks{0};
    size_t      active_tree_cached_blocks{0};
    size_t      request_ref_blocks{0};
    size_t      connector_ref_blocks{0};
    size_t      block_cache_ref_blocks{0};
    size_t      eviction_ref_blocks{0};
    size_t      store_ref_blocks{0};
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
    size_t         evictable_candidate_count{0};
};

struct BlockTreeCacheReuseTimeSample {
    Tier           tier{Tier::DEVICE};
    CacheGroupType group_type{CacheGroupType::FULL};
    int64_t        insert_time_us{0};
    int64_t        last_access_time_us{0};
    int64_t        access_time_us{0};
};

struct BlockTreeCacheReuseTimeMetricsSnapshot {
    Tier           tier{Tier::DEVICE};
    CacheGroupType group_type{CacheGroupType::FULL};
    int64_t        reuse_interval_avg_ms{0};
    int64_t        reuse_interval_max_ms{0};
    int64_t        hit_entry_age_avg_ms{0};
    int64_t        hit_entry_age_max_ms{0};
};

struct BlockTreeTransferBytesKey {
    std::string pool_name;
    CacheGroupType group_type{CacheGroupType::FULL};

    bool operator==(const BlockTreeTransferBytesKey& other) const {
        return pool_name == other.pool_name && group_type == other.group_type;
    }
};

struct BlockTreeTransferBytesKeyHash {
    size_t operator()(const BlockTreeTransferBytesKey& key) const {
        return std::hash<std::string>{}(key.pool_name) ^ (static_cast<size_t>(key.group_type) << 1);
    }
};

using BlockTreeTransferBytes = std::unordered_map<BlockTreeTransferBytesKey, size_t, BlockTreeTransferBytesKeyHash>;

class BlockTreeCacheMetricsReporter final {
public:
    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter);
    bool enabled() const;

    std::vector<BlockTreePoolMetricsSnapshot> collectPoolMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets,
                                                                          const BlockTreeEvictor& evictor) const;
    std::vector<BlockTreeEvictableMetricsSnapshot>
    collectEvictableMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets, const BlockTreeEvictor& evictor) const;
    void reportEvictableCandidateCount(const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const;
    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot>
         collectCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeSample>& samples) const;
    void reportCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeMetricsSnapshot>& snapshots) const;
    void reportEvictionFinished(const EvictionTask&             task,
                                const EvictionTaskResult&       task_result,
                                const std::vector<GroupSetPtr>& group_sets) const;

    int64_t reportTransferStarted(CacheTransferOperation operation, Tier source_tier, Tier target_tier);
    void    reportTransferFinished(CacheTransferOperation                  operation,
                                   Tier                                    source_tier,
                                   Tier                                    target_tier,
                                   size_t                                  block_count,
                                   int64_t                                 begin_time_us,
                                   bool                                    success,
                                   const std::vector<TransferDescriptor>& successful_descriptors,
                                   const std::vector<GroupSetPtr>&        group_sets);
    void    reportStorePublish(Tier target_tier, size_t accepted_blocks, size_t duplicate_blocks) const;

private:
    static constexpr size_t kOperationCount = 3;
    static constexpr size_t kDirectionCount = 5;

    static int transferDirectionIndex(Tier source_tier, Tier target_tier);
    void       reportEvictionTransfer(const TransferDescriptor&       desc,
                                      const EvictionTimingSnapshot&   timing,
                                      const std::vector<GroupSetPtr>& group_sets,
                                      int64_t                         finish_time_us,
                                      bool                            report_candidate_times) const;
    void       reportStoreBlocks(Tier target_tier, const char* outcome, size_t block_count) const;
    void       accumulateTransferBytes(const TransferDescriptor& desc,
                                       const GroupSetPtr&        group_set,
                                       BlockTreeTransferBytes&   transfer_bytes) const;
    void       accumulateTransferBytes(const std::vector<TransferDescriptor>& descs,
                                       const std::vector<GroupSetPtr>&        group_sets,
                                       BlockTreeTransferBytes&                transfer_bytes) const;

    std::shared_ptr<kmonitor::MetricsReporter>                                     metrics_reporter_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
