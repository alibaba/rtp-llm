#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"

#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeLoadHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include <algorithm>
#include "autil/ThreadPool.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtp_llm {

struct P2PConnectorSchedulerConfig;

class P2PConnectorAsyncReadContext: public AsyncContext {
public:
    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                            resource,
                                 std::string                                          unique_key,
                                 const std::shared_ptr<P2PConnectorMetricsCollector>& collector,
                                 int64_t                                              lease_query_timeout_ms,
                                 bool                                                 no_transfer          = false,
                                 int64_t                                              request_deadline_ms  = 0,
                                 int64_t                                              transfer_deadline_ms = 0):
        resource_(resource),
        unique_key_(std::move(unique_key)),
        collector_(collector),
        lease_query_timeout_ms_(std::max<int64_t>(1, lease_query_timeout_ms)),
        no_transfer_(no_transfer),
        request_deadline_ms_(request_deadline_ms),
        transfer_deadline_ms_(transfer_deadline_ms),
        transfer_stop_at_ms_(localTransferDeadline(transfer_deadline_ms)),
        done_(false),
        success_(false),
        error_code_(ErrorCode::NONE_ERROR) {}

    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                            resource,
                                 const std::shared_ptr<P2PBroadcastClient::Result>&   tp_sync_result,
                                 const std::shared_ptr<DecodeLoadHelper::Result>&     server_call_result,
                                 const std::shared_ptr<P2PConnectorMetricsCollector>& collector,
                                 int64_t                                              lease_query_timeout_ms,
                                 bool                                                 no_transfer          = false,
                                 int64_t                                              request_deadline_ms  = 0,
                                 int64_t                                              transfer_deadline_ms = 0):
        resource_(resource),
        tp_sync_result_(tp_sync_result),
        server_call_result_(server_call_result),
        collector_(collector),
        lease_query_timeout_ms_(std::max<int64_t>(1, lease_query_timeout_ms)),
        no_transfer_(no_transfer),
        request_deadline_ms_(request_deadline_ms),
        transfer_deadline_ms_(transfer_deadline_ms),
        transfer_stop_at_ms_(localTransferDeadline(transfer_deadline_ms)),
        done_(false),
        success_(false),
        error_code_(ErrorCode::NONE_ERROR),
        calls_ready_(true),
        kickoff_state_(KickoffState::CALLS_READY) {}
    virtual ~P2PConnectorAsyncReadContext() = default;

public:
    void waitDone() override;
    void onDone(DoneCallback callback) override;
    // Fires once the read is terminal and no transport can still write the target.
    void onTargetSafeToRelease(std::function<void()> callback);
    bool done() const override;
    bool success() const override;

    void checkDone();
    void checkCancelDone();
    bool expireTransferDeadlineIfNeeded();
    bool needCancel() const;
    void cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);

    /// Atomically claim the queued kickoff. If cancellation won while the task
    /// was still queued, returns false and no RPC may be created.
    bool beginKickoff();

    /// Bind the RPC results produced by the asynchronous kickoff task.
    /// Returns true when cancellation was requested before/during kickoff and
    /// the newly-created calls should be cancelled immediately.
    bool setCallResults(const std::shared_ptr<P2PBroadcastClient::Result>& tp_sync_result,
                        const std::shared_ptr<DecodeLoadHelper::Result>&  server_call_result);
    void markStartFailed(const ErrorInfo& error_info);
    // An exception may follow a partial RPC dispatch: retain KV until workers confirm stop.
    void markKickoffUnconfirmed(const ErrorInfo& error_info);
    void    setWakeupCallback(std::function<void()> callback);
    // Next due event on the local steady clock.
    int64_t nextWakeupMs(bool allow_control) const;
    bool    needsControl() const;
    void    runControl(const std::shared_ptr<P2PBroadcastClient>& client);
    bool cancelRequested() const {
        return cancel_requested_.load(std::memory_order_acquire);
    }

    // Called periodically while Decode target resources are retained after the
    // request has already completed with timeout/cancellation.
    // Broadcasts QUERY_LEASE_STATUS to all TP workers. The hold ends when all
    // ranks stop. An unconfirmed lease remains held without a time limit.
    void pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);

    bool needLeasePoll() const {
        if (lease_query_timeout_ms_ <= 0 || !calls_ready_.load(std::memory_order_acquire)
            || !lease_hold_pending_.load(std::memory_order_acquire)) {
            return false;
        }
        if (!cancel_confirmed_.load(std::memory_order_acquire)) {
            return false;
        }
        return !lease_all_ranks_stopped_.load(std::memory_order_acquire)
               && P2PNotification::nowMs() >= lease_poll_next_ms_.load(std::memory_order_relaxed);
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
        if (!calls_ready_.load(std::memory_order_acquire) || !server_call_result_ || !server_call_result_->done()
            || !server_call_result_->success() || !server_call_result_->side_channel_payload.has_data) {
            return nullptr;
        }
        return &server_call_result_->side_channel_payload;
    }

    int64_t handoffReadyTimeUs() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return handoff_ready_time_us_;
    }

    ErrorInfo errorInfo() const override;

    size_t matchedBlockCount() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return resource_ ? resource_->cacheKeys().size() : 0;
    }

    std::shared_ptr<P2PConnectorMetricsCollector> metricsCollector() const {
        return collector_;
    }

private:
    friend class P2PConnectorAsyncReadContextChecker;

    void dispatchDoneCallbacks();
    void scheduleCancelRetry(const char* reason);
    struct DoneNotifier {
        P2PConnectorAsyncReadContext& context;
        ~DoneNotifier() { context.dispatchDoneCallbacks(); }
    };
    std::vector<DoneCallback> done_callbacks_;
    std::vector<std::function<void()>> target_release_callbacks_;

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
    void              beginLeaseHold();
    void              releaseLeaseHoldLocked(bool all_ranks_stopped, int64_t now_us);
    std::function<void()> completionCallback() const;
    void                  registerReadCompletionCallbacksLocked();
    void                  notify() const;

    KVCacheResourcePtr                                     resource_;
    const std::string                                      unique_key_;
    std::shared_ptr<P2PBroadcastClient::Result>            tp_sync_result_;
    std::shared_ptr<DecodeLoadHelper::Result>              server_call_result_;
    const std::shared_ptr<P2PConnectorMetricsCollector>    collector_;

    const int64_t lease_query_timeout_ms_;
    const bool    no_transfer_;
    const int64_t request_deadline_ms_;
    const int64_t transfer_deadline_ms_;  // original wire deadline/identity
    const int64_t transfer_stop_at_ms_;   // anchored once in the local steady-clock domain

    static int64_t localTransferDeadline(int64_t wall_deadline_ms) {
        if (wall_deadline_ms <= 0) {
            return 0;
        }
        const auto remaining = std::max<int64_t>(0, wall_deadline_ms - currentTimeMs());
        const auto now = P2PNotification::nowMs();
        return now + std::min(remaining, std::numeric_limits<int64_t>::max() - now);
    }

    std::mutex                                   check_mutex_;
    mutable std::mutex      state_mutex_;
    std::condition_variable done_cv_;
    bool                    done_{false};
    bool                    success_{false};
    int64_t                 handoff_ready_time_us_{0};
    ErrorCode               error_code_;
    std::string             error_message_;
    std::atomic<int64_t>                         lease_hold_start_us_{0};  // steady clock
    bool                                         lease_released_{false};  // guarded by state_mutex_
    std::atomic<bool>       lease_hold_pending_{false};
    std::atomic<bool>                            tp_cancel_broadcast_triggered_{false};
    std::shared_ptr<P2PBroadcastClient::Result> cancel_result_;  // guarded by state_mutex_
    std::atomic<bool>                            cancel_confirmed_{false};

    // Lease polling state (active while lease_hold_pending_ is true).
    std::atomic<bool>    lease_all_ranks_stopped_{false};  // set when poll confirms all ranks stopped
    std::atomic<int64_t> lease_poll_next_ms_{0};           // steady-clock earliest next poll
    std::atomic<int64_t> lease_poll_interval_ms_{10};      // backoff interval, starts 10ms, max 100ms
    std::atomic<int>     lease_poll_retry_count_{0};
    enum class LeaseQueryState {
        Pending,
        Failed,
        WaitingForStop
    };
    std::atomic<LeaseQueryState>     lease_query_state_{LeaseQueryState::Pending};
    std::atomic<bool>    calls_ready_{false};
    std::atomic<bool>    cancel_requested_{false};
    KickoffState         kickoff_state_{KickoffState::QUEUED};  // guarded by state_mutex_
    std::shared_ptr<const std::function<void()>> wakeup_callback_;  // accessed with atomic_load/store
    std::atomic<int64_t>             control_retry_ms_{0};
    int64_t                          cancel_retry_interval_ms_{10};
    int64_t                          next_cancel_warning_ms_{0};
    std::atomic<bool>                control_active_{false};
};

// Owns live reads through transport stop confirmation and advances them on the shared RPC CQ.
class P2PConnectorAsyncReadContextChecker:
    public std::enable_shared_from_this<P2PConnectorAsyncReadContextChecker> {
public:
    P2PConnectorAsyncReadContextChecker() = default;
    ~P2PConnectorAsyncReadContextChecker();

public:
    bool init(const kmonitor::MetricsReporterPtr&               metrics_reporter,
              const std::shared_ptr<P2PBroadcastClient>&        tp_broadcast_client,
              const std::shared_ptr<autil::LockFreeThreadPool>& control_pool,
              const P2PConnectorSchedulerConfig&               config);
    void stop();
    /// @brief 添加需要跟踪的异步 read 上下文
    void   addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);
    size_t inflightContextCount() const;

private:
    struct TrackedRead {
        std::shared_ptr<P2PConnectorAsyncReadContext> context;
        std::function<void()>                        cancel_wakeup;
        int64_t                                     wakeup_ms = std::numeric_limits<int64_t>::max();
        uint64_t                                    generation = 0;
    };
    void scheduleRead(const std::shared_ptr<P2PConnectorAsyncReadContext>& context, int64_t wakeup_ms);
    void advanceRead(const P2PConnectorAsyncReadContext* key, uint64_t generation);
    void scheduleRetainedCheck(int64_t delay_ms);
    void runRetainedCheck();
    void checkRetainedResources(const std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>>& contexts);

    // 上报一个终态的 read 上下文采样（必要时补算 total），保证每个上下文恰好上报一次。
    void reportDecodeScheduleMetrics(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);

private:
    kmonitor::MetricsReporterPtr                               metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                        tp_broadcast_client_;
    mutable std::mutex                                         async_contexts_mutex_;
    std::unordered_map<const P2PConnectorAsyncReadContext*, TrackedRead> async_contexts_;
    uint64_t next_generation_{0};  // guarded by async_contexts_mutex_
    // Serializes shutdown with callbacks already executing on the CQ.
    std::mutex                                                  processing_mutex_;
    bool                                                        stopping_{false};  // guarded by async_contexts_mutex_
    std::unordered_map<const P2PConnectorAsyncReadContext*, std::shared_ptr<P2PConnectorAsyncReadContext>>
        retained_contexts_;  // guarded by processing_mutex_
    std::function<void()>                                      cancel_retained_check_;
    std::shared_ptr<autil::LockFreeThreadPool>                 control_pool_;
    // Independent leases may be queried concurrently on the dedicated pool.
    std::shared_ptr<std::atomic<size_t>> control_inflight_{std::make_shared<std::atomic<size_t>>(0)};
    size_t                               control_capacity_{0};
    std::map<std::string, size_t>                              group_block_bytes_;
    size_t                                                     total_kv_bytes_{0};
    int64_t                                                    last_leak_log_ms_{0};
};

class P2PConnectorAsyncWriteContext:
    public AsyncContext,
    public std::enable_shared_from_this<P2PConnectorAsyncWriteContext> {
public:
    using PeerCancel = std::function<std::shared_ptr<DecodeWriteHelper::CancelResult>()>;

    P2PConnectorAsyncWriteContext(std::string                         unique_key,
                                  int64_t                             deadline_ms,
                                  std::shared_ptr<P2PBroadcastClient> client,
                                  int64_t                             control_timeout_ms,
                                  int64_t                             lease_query_timeout_ms,
                                  std::function<void()>               on_released      = {},
                                  kmonitor::MetricsReporterPtr        metrics_reporter = nullptr,
                                  KVCacheResourcePtr                  source_resource  = nullptr);

    void               waitDone() override;
    void               onDone(DoneCallback callback) override;
    bool               done() const override;
    bool               success() const override;
    void               checkDone();
    void               cancel(const ErrorInfo& error = ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled"));
    bool               beginKickoff();
    bool               resourceHoldPending() const;
    const std::string& uniqueKey() const {
        return unique_key_;
    }
    ErrorInfo errorInfo() const override;

    void    finishWithoutTransfer(const ErrorInfo& error = ErrorInfo::OkStatus());
    int64_t deadlineMs() const;
    void    setPlannedBytes(int64_t bytes);
    bool    beginPeerCall(PeerCancel cancel);
    void    setPeerCallResult(std::shared_ptr<DecodeWriteHelper::Result> result);

private:
    friend class P2PConnectorAsyncWriteContextChecker;

    void                  expireTransferDeadlineIfNeeded();
    void                  checkCancelDone();
    void                  setNotification(const std::shared_ptr<P2PNotification>& notification);
    int64_t               nextWakeupMs() const;
    void                  runControl(const std::shared_ptr<autil::ThreadPool>& control_pool);
    void                  dispatchDoneCallbacks();
    std::function<void()> completionCallback() const;
    void                  registerWriteCompletionCallbacksLocked();
    void                  notify() const;

    void failLocked(const ErrorInfo& error);
    void checkPeerResultLocked();
    void tryFinishLocked(const ErrorInfo& error);
    void finishRelease();
    struct DoneNotifier {
        P2PConnectorAsyncWriteContext& context;
        ~DoneNotifier() {
            context.finishRelease();
            context.dispatchDoneCallbacks();
        }
    };

    const std::string                   unique_key_;
    int64_t                             deadline_ms_;
    int64_t                             transfer_stop_at_ms_;
    std::weak_ptr<KVCacheResource>      source_resource_;
    std::shared_ptr<P2PBroadcastClient> client_;
    const int64_t                       control_timeout_ms_;
    const int64_t                       lease_query_timeout_ms_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    WriteSchedulerMetricsCollector      metrics_;
    const int64_t                       start_time_us_{currentTimeUs()};
    int64_t                             drain_start_time_us_{0};
    int64_t                             peer_start_time_us_{0};

    mutable std::mutex               mutex_;
    std::condition_variable          cv_;
    std::vector<DoneCallback>        done_callbacks_;
    std::shared_ptr<P2PNotification> notification_;
    bool                             kickoff_started_{false};
    bool                             cancelled_{false};
    bool                             done_{false};
    bool                             released_{false};
    bool                             release_ready_{false};
    bool                             release_in_progress_{false};
    bool                             transfer_stopped_{false};
    bool                             worker_cancelled_{false};
    bool                             control_submitting_{false};
    int64_t                          control_next_ms_{0};
    int64_t                          control_interval_ms_{10};

    std::shared_ptr<P2PBroadcastClient::TpBroadcastResult> control_result_;
    std::shared_ptr<P2PBroadcastClient::TpBroadcastResult> lease_result_;

    bool                                             peer_pending_{false};
    bool                                             peer_stopped_{false};
    PeerCancel                                       peer_cancel_;
    std::shared_ptr<DecodeWriteHelper::Result>       peer_result_;
    std::shared_ptr<DecodeWriteHelper::CancelResult> peer_cancel_result_;
    std::function<void()>                            on_released_;
    ErrorInfo                                        error_;
};

class P2PConnectorAsyncWriteContextChecker {
public:
    ~P2PConnectorAsyncWriteContextChecker();
    bool   init(const P2PConnectorSchedulerConfig& config, kmonitor::MetricsReporterPtr metrics_reporter = nullptr);
    bool   addContext(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context);
    void   cancelAll();
    void   stop(int64_t steady_deadline_ms = 0);
    size_t inflightContextCount() const;

private:
    void    checkOnce();
    int64_t nextWakeupMs() const;

    mutable std::mutex                                                    mutex_;
    std::map<std::string, std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts_;
    std::thread                                                           thread_;
    bool                                                                  accepting_{false};
    bool                                                                  stopped_{false};
    std::atomic<bool>                                                     stopping_{false};
    std::shared_ptr<P2PNotification>   notification_{std::make_shared<P2PNotification>()};
    std::shared_ptr<autil::ThreadPool> control_pool_;
    WritebackRetainedKv                retained_kv_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
};

}  // namespace rtp_llm
