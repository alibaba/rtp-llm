#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/AsymmetricTpUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeTargetWriteLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtp_llm {

class P2PConnectorWorkerDecode {
public:
    P2PConnectorWorkerDecode(P2PConnectorWorkerConfig                    config,
                             const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                             const kmonitor::MetricsReporterPtr&         metrics_reporter,
                             const transfer::IKVCacheReceiverPtr&        receiver);
    ~P2PConnectorWorkerDecode() = default;

public:
    ErrorInfo read(int64_t                                               request_id,
                   const std::string&                                    unique_key,
                   int64_t                                               deadline_ms,
                   const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                   int                                                   remote_tp_size = 1);

    bool cancelRead(const std::string& unique_key);

    // Query the local lease state for a given unique_key (called by QUERY_LEASE_STATUS handler).
    // Returns false if no lease is found (lease already cleaned up = transfers done).
    bool
    queryLeaseStatus(const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped);

private:
    int calculateRecvPartitionCount(int remote_tp_size) const;

private:
    struct ReadTaskGroup {
        std::vector<std::string>                   partition_keys;
        std::vector<transfer::IKVCacheRecvTaskPtr> tasks;
        std::atomic<bool>                          cancelled{false};
        std::shared_ptr<DecodeTargetWriteLease>    lease;
    };

    enum class ReadWaitOutcome {
        AllDone,
        Cancelled,
        ReturnDeadlineIncomplete
    };

    struct RecvResultInfo {
        bool        success = true;
        ErrorCode   error_code{ErrorCode::NONE_ERROR};
        std::string error_msg;
    };

    ErrorInfo buildRecvTasks(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             int                                                   recv_partition_count,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             const std::shared_ptr<ReadTaskGroup>&                 task_group,
                             int&                                                  total_block_count) const;

    /// 等待 recv 完成、cancel，或到达 return_deadline_ms（D - return_before）；到达 steal 时刻时从 store steal 各
    /// partition key。
    ReadWaitOutcome waitRecvTasksWithReadDeadlinePolicy(const std::shared_ptr<ReadTaskGroup>& task_group,
                                                        int64_t                               deadline_ms,
                                                        int64_t                               request_id,
                                                        const std::string&                    unique_key) const;

    RecvResultInfo aggregateRecvTaskResults(const std::shared_ptr<ReadTaskGroup>& task_group) const;

    void reportReadMetrics(int total_block_count, bool success, int64_t read_start_time_us) const;

    void cleanupRecvTaskStore(const std::shared_ptr<ReadTaskGroup>& task_group, bool cancel_pending_tasks) const;

private:
    P2PConnectorWorkerConfig             config_;
    std::shared_ptr<LayerBlockConverter> layer_block_converter_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
    transfer::IKVCacheReceiverPtr        receiver_;

    mutable std::mutex                                              read_tasks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ReadTaskGroup>> read_tasks_;
    std::unordered_set<std::string>                                 building_read_keys_;
    std::unordered_set<std::string>                                 pending_cancel_keys_;

    // Leases kept alive after read() returns (for TRANSFER_NOT_DONE path) until all ops finish.
    // Keyed by unique_key. Cleaned up lazily when polled via queryLeaseStatus.
    struct LeaseMapEntry {
        std::shared_ptr<ReadTaskGroup> task_group;
        int                            finish_counted{0};  // how many tasks have been counted as finished so far
        int64_t                        create_time_ms{0};
    };

    static constexpr int64_t kLeaseMapTtlMs = 600000;  // 10 min

    void evictStaleLeases();
    mutable std::mutex                             lease_map_mutex_;
    std::unordered_map<std::string, LeaseMapEntry> lease_map_;
};

}  // namespace rtp_llm
