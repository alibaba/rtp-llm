#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

struct BlockTreeInsertHardStopInfo;
struct BlockTreeQueueSizes;
class CacheStageMetrics;

struct BlockTreePoolMetricsSnapshot {
    Tier        tier{Tier::DEVICE};
    std::string pool_name;
    size_t      block_size_bytes{0};
    size_t      total_blocks{0};
    size_t      free_blocks{0};
    size_t      used_blocks{0};
    size_t      available_blocks{0};
    size_t      active_blocks{0};
    size_t      request_ref_blocks{0};
    size_t      remote_connector_ref_blocks{0};
    size_t      pd_connector_ref_blocks{0};
    size_t      block_cache_ref_blocks{0};
    size_t      load_ref_blocks{0};
    size_t      eviction_target_ref_blocks{0};
    size_t      store_ref_blocks{0};
    size_t      frozen_ref_blocks{0};

    std::optional<int64_t> read_bytes;
    std::optional<int64_t> write_bytes;
    std::optional<double>  read_bandwidth_bytes_per_sec;
    std::optional<double>  write_bandwidth_bytes_per_sec;
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

struct BlockTreeCacheReuseTimeMetricsSnapshot {
    Tier                   tier{Tier::DEVICE};
    CacheGroupType         group_type{CacheGroupType::FULL};
    std::optional<int64_t> reuse_block_first_hit_idletime_ms;
    int64_t                reuse_block_hit_idletime_ms{0};
};

struct BlockTreeTransferBytesKey {
    std::string    pool_name;
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
    explicit BlockTreeCacheMetricsReporter(std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
                                           bool                                       enable_stage_metrics = true,
                                           std::chrono::milliseconds stage_interval = std::chrono::seconds(1));
    ~BlockTreeCacheMetricsReporter();
    bool stageMetricsEnabled() const noexcept {
        return stage_metrics_ != nullptr;
    }
    void flushStageMetrics() const;

    // Durations are per worker batch, except coordinator multi-rank stages.
    // Request/transfer IDs belong in the timeline, never in metric tags.
    void reportDiskLoadStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank  = -1,
                             const char* scope = "worker_batch") const noexcept;
    void reportTransferStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank,
                             const char* scope,
                             const char* operation,
                             const char* source_tier,
                             const char* target_tier) const noexcept;

    std::vector<BlockTreePoolMetricsSnapshot>
    collectPoolMetricsSnapshots(const std::vector<GroupSetPtr>&       group_sets,
                                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
    std::vector<BlockTreeEvictableMetricsSnapshot>
    collectEvictableMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets, const BlockTreeEvictor& evictor) const;
    void reportEvictableCandidateCount(const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const;
    void reportCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeMetricsSnapshot>& snapshots) const;
    void reportEvictionFinished(const EvictionTransferTask& task, const std::vector<GroupSetPtr>& group_sets) const;
    void reportEvictionFinished(const EvictionDropTask&         task,
                                const std::vector<GroupSetPtr>& group_sets,
                                const char*                     trigger_type) const;
    void reportEvictedDescriptor(const TransferDescriptor&       desc,
                                 const EvictionTimingSnapshot&   timing,
                                 const std::vector<GroupSetPtr>& group_sets,
                                 const char*                     trigger_type) const;
    void reportEvictionTriggered(Tier source_tier, CacheGroupType group_type, bool force_drop) const;
    void reportEvictionBlocks(Tier           source_tier,
                              CacheGroupType group_type,
                              bool           force_drop,
                              size_t         required_blocks,
                              size_t         scheduled_blocks) const;

    void reportLoadJoin(size_t dependency_count) const;
    void reportLoadJoinWait(int64_t join_wait_latency_us) const;
    void reportInsertHardStop(const BlockTreeInsertHardStopInfo& hard_stop) const;

    int64_t reportTransferStarted(CacheTransferOperation operation, Tier source_tier, Tier target_tier);
    // Report once after all batches/ranks of a logical copy have completed.
    void maybeReportCopyError(Tier source_tier, Tier target_tier, const TransferResult& result) const;
    void reportTransferFinished(CacheTransferOperation                 operation,
                                Tier                                   source_tier,
                                Tier                                   target_tier,
                                size_t                                 descriptors_per_transfer,
                                int64_t                                begin_time_us,
                                bool                                   success,
                                const std::vector<TransferDescriptor>& successful_descriptors,
                                const std::vector<GroupSetPtr>&        group_sets);
    void reportQueueWaitMetric(bool        callback,
                               const char* pool_type,
                               const char* operation,
                               Tier        source_tier,
                               Tier        target_tier,
                               int64_t     latency_us) const noexcept;
    void reportStorePublish(Tier target_tier, size_t accepted_blocks, size_t duplicate_blocks) const;
    void reportQueueBacklog(const BlockTreeQueueSizes& queue_sizes, const char* pool_type) const;

private:
    struct PoolIOSample {
        size_t                                read_bytes{0};
        size_t                                write_bytes{0};
        std::chrono::steady_clock::time_point time;
    };

    std::unordered_map<const IBlockPool*, PoolIOSample> pool_io_samples_;

    static constexpr size_t kOperationCount = 3;
    static constexpr size_t kDirectionCount = 5;

    static int transferDirectionIndex(Tier source_tier, Tier target_tier);
    void
    reportEvictionTrigger(Tier source_tier, CacheGroupType group_type, const char* trigger_type, int64_t count) const;
    void reportStoreBlocks(Tier target_tier, const char* outcome, size_t block_count) const;
    void accumulateTransferBytes(const TransferDescriptor& desc,
                                 const GroupSetPtr&        group_set,
                                 BlockTreeTransferBytes&   transfer_bytes) const;
    void accumulateTransferBytes(const std::vector<TransferDescriptor>& descs,
                                 const std::vector<GroupSetPtr>&        group_sets,
                                 BlockTreeTransferBytes&                transfer_bytes) const;
    std::shared_ptr<kmonitor::MetricsReporter>                                     metrics_reporter_;
    std::unique_ptr<CacheStageMetrics>                                             stage_metrics_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
