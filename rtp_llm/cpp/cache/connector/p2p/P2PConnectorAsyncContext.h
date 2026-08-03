#pragma once

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/PrefillLoadCaller.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/LoopThread.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

/// @brief PD 分离场景下的匹配上下文，始终全量匹配
class P2PConnectorAsyncMatchContext: public AsyncMatchContext {
public:
    P2PConnectorAsyncMatchContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncMatchContext() {}

public:
    size_t matchedBlockCount() const override;
    bool   done() const override;
    bool   success() const override;
    void   waitDone() override {}

private:
    const KVCacheResourcePtr resource_;
};

class P2PConnectorAsyncReadContext: public AsyncContext {
public:
    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                               resource,
                                 std::string                                             unique_key,
                                 const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                                 int64_t                                                 transfer_not_done_hold_ms,
                                 bool                                                    no_transfer = false,
                                 int64_t                                                 request_deadline_ms = 0):
        resource_(resource),
        unique_key_(std::move(unique_key)),
        collector_(collector),
        transfer_not_done_hold_ms_(transfer_not_done_hold_ms),
        no_transfer_(no_transfer),
        request_deadline_ms_(request_deadline_ms),
        done_(false),
        success_(false),
        error_code_(ErrorCode::NONE_ERROR) {}

    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                               resource,
                                 const std::shared_ptr<P2PBroadcastClient::Result>&      tp_sync_result,
                                 const std::shared_ptr<PrefillLoadCaller::Result>&       server_call_result,
                                 const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                                 int64_t                                                 transfer_not_done_hold_ms,
                                 bool                                                    no_transfer = false,
                                 int64_t                                                 request_deadline_ms = 0):
        resource_(resource),
        tp_sync_result_(tp_sync_result),
        server_call_result_(server_call_result),
        collector_(collector),
        transfer_not_done_hold_ms_(transfer_not_done_hold_ms),
        no_transfer_(no_transfer),
        request_deadline_ms_(request_deadline_ms),
        done_(false),
        success_(false),
        error_code_(ErrorCode::NONE_ERROR),
        calls_ready_(true),
        kickoff_state_(KickoffState::CALLS_READY) {}
    virtual ~P2PConnectorAsyncReadContext() = default;

public:
    void waitDone() override;
    bool done() const override;
    bool success() const override;

    void checkDone();
    bool needCancel() const;
    void cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);

    /// Atomically claim the queued kickoff. If cancellation won while the task
    /// was still queued, returns false and no RPC may be created.
    bool beginKickoff();

    /// Bind the RPC results produced by the asynchronous kickoff task.
    /// Returns true when cancellation was requested before/during kickoff and
    /// the newly-created calls should be cancelled immediately.
    bool setCallResults(const std::shared_ptr<P2PBroadcastClient::Result>& tp_sync_result,
                        const std::shared_ptr<PrefillLoadCaller::Result>&  server_call_result);
    void markStartFailed(const ErrorInfo& error_info);
    bool cancelRequested() const {
        return cancel_requested_.load(std::memory_order_acquire);
    }

    // Called periodically while Decode target resources are retained after the
    // request has already completed with TRANSFER_NOT_DONE/CANCELLED.
    // Broadcasts QUERY_LEASE_STATUS to all TP workers. The hold ends when all
    // ranks stop or when the configured resource-hold deadline is reached.
    void pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);

    // Release an expired resource hold without issuing an RPC. The checker
    // calls this for every context before selecting one lease to poll.
    bool expireLeaseHoldIfNeeded();

    bool needLeasePoll() const {
        if (!lease_hold_pending_.load(std::memory_order_acquire)) {
            return false;
        }
        const int64_t now_ms        = currentTimeMs();
        const int64_t hold_until_ms = lease_hold_until_ms_.load(std::memory_order_relaxed);
        return (hold_until_ms > 0 && now_ms >= hold_until_ms)
            || (!lease_all_ranks_stopped_.load(std::memory_order_acquire)
                && now_ms >= lease_poll_next_ms_.load(std::memory_order_relaxed));
    }

    bool resourceHoldPending() const {
        return lease_hold_pending_.load(std::memory_order_acquire);
    }

    std::string uniqueKey() const {
        if (!unique_key_.empty()) {
            return unique_key_;
        }
        return tp_sync_result_ ? tp_sync_result_->uniqueKey() : "";
    }

    // Access side-channel payload parsed from Prefill response (for downstream apply in waitLoadCacheDone)
    const P2PSideChannelPayload* sideChannelPayload() const {
        if (!calls_ready_.load(std::memory_order_acquire) || !server_call_result_
            || !server_call_result_->side_channel_payload.has_data) {
            return nullptr;
        }
        return &server_call_result_->side_channel_payload;
    }

    ErrorInfo errorInfo() const override;

    size_t matchedBlockCount() const {
        return resource_ ? resource_->cacheKeys().size() : 0;
    }

private:
    enum class KickoffState {
        QUEUED,
        STARTING,
        CALLS_READY,
    };

    struct MergedReadOutcome {
        bool        success{false};
        ErrorCode   error_code{ErrorCode::NONE_ERROR};
        std::string error_message;
    };

    MergedReadOutcome mergeReadResultsWhenBothDone() const;
    void              applyMergedReadOutcome(const MergedReadOutcome& outcome);

    const KVCacheResourcePtr                               resource_;
    const std::string                                      unique_key_;
    std::shared_ptr<P2PBroadcastClient::Result>            tp_sync_result_;
    std::shared_ptr<PrefillLoadCaller::Result>             server_call_result_;
    const std::shared_ptr<DecodeSchedulerMetricsCollector> collector_;

    const int64_t transfer_not_done_hold_ms_;
    const bool    no_transfer_;
    const int64_t request_deadline_ms_;

    mutable std::mutex      state_mutex_;
    std::condition_variable done_cv_;
    bool                    done_{false};
    bool                    success_{false};
    ErrorCode               error_code_;
    std::string             error_message_;
    std::atomic<bool>       lease_hold_pending_{false};
    std::atomic<int64_t>    lease_hold_until_ms_{0};
    std::atomic<bool>       tp_cancel_broadcast_triggered_{false};

    // Lease polling state (active while lease_hold_pending_ is true).
    std::atomic<bool>    lease_all_ranks_stopped_{false};  // set when poll confirms all ranks stopped
    std::atomic<int64_t> lease_poll_next_ms_{0};           // rate limit: earliest time for next poll
    std::atomic<int64_t> lease_poll_interval_ms_{10};      // backoff interval, starts 10ms, max 100ms
    std::atomic<int>     lease_poll_retry_count_{0};
    std::atomic<bool>    calls_ready_{false};
    std::atomic<bool>    cancel_requested_{false};
    KickoffState         kickoff_state_{KickoffState::QUEUED};  // guarded by state_mutex_
};

/// @brief P2P 按层写入的异步上下文。
/// Write-by-layer is fire-and-forget; actual transfer status is tracked separately.
/// @note done()/success() 恒为 true，仅满足 AsyncContext 接口形态，不得据此推断真实传输结果。
class P2PConnectorAsyncWriteByLayerContext: public AsyncContext {
public:
    P2PConnectorAsyncWriteByLayerContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncWriteByLayerContext() {}

public:
    void waitDone() override;  // done() always true, no blocking
    bool done() const override;
    bool success() const override;

private:
    const KVCacheResourcePtr resource_;
};

/// @brief 后台线程定期检查 in-flight 异步 read 上下文，超时时自动取消
class P2PConnectorAsyncReadContextChecker {
public:
    P2PConnectorAsyncReadContextChecker() = default;
    ~P2PConnectorAsyncReadContextChecker();

public:
    /// @brief 启动后台检查线程
    bool init(const kmonitor::MetricsReporterPtr&        metrics_reporter,
              const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    void stop();
    /// @brief 添加需要跟踪的异步 read 上下文
    void   addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);
    size_t inflightContextCount() const;

private:
    void checkOnce();

private:
    kmonitor::MetricsReporterPtr                               metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                        tp_broadcast_client_;
    mutable std::mutex                                         async_contexts_mutex_;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> async_contexts_;
    autil::LoopThreadPtr                                       check_done_thread_;
    size_t                                                     lease_poll_cursor_{0};
};

}  // namespace rtp_llm
