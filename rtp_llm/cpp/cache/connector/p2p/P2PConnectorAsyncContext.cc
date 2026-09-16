#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <exception>
#include <functional>
#include <limits>

namespace rtp_llm {

namespace {
constexpr int64_t kLeasePollInitialIntervalMs = 10;
constexpr int64_t kLeasePollMaxIntervalMs     = 100;
constexpr int64_t kLeasePollRpcTimeoutMs      = 500;

const char* readOutcomeHoldReason(ErrorCode error_code) {
    switch (error_code) {
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE:
            return "TRANSFER_NOT_DONE";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED:
            return "READ_CANCELLED";
        default:
            return "READ_RESULT_UNCONFIRMED";
    }
}
}  // namespace

/*----------------------------------------------- P2PConnectorAsyncReadContext
 * -------------------------------------------------*/
bool P2PConnectorAsyncReadContext::done() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return done_;
}

bool P2PConnectorAsyncReadContext::success() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return success_;
}

void P2PConnectorAsyncReadContext::waitDone() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    done_cv_.wait(lock, [this]() { return done_; });
}

bool P2PConnectorAsyncReadContext::setCallResults(
    const std::shared_ptr<P2PBroadcastClient::Result>& tp_sync_result,
    const std::shared_ptr<DecodeLoadHelper::Result>&  server_call_result) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        tp_sync_result_     = tp_sync_result;
        server_call_result_ = server_call_result;
        kickoff_state_      = KickoffState::CALLS_READY;
    }
    calls_ready_.store(true, std::memory_order_release);
    return cancelRequested();
}

bool P2PConnectorAsyncReadContext::beginKickoff() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (kickoff_state_ != KickoffState::QUEUED || done_ || cancel_requested_.load(std::memory_order_acquire)) {
        return false;
    }
    kickoff_state_ = KickoffState::STARTING;
    return true;
}

void P2PConnectorAsyncReadContext::markStartFailed(const ErrorInfo& error_info) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return;
        }
        done_          = true;
        success_       = false;
        error_code_    = error_info.code();
        error_message_ = error_info.ToString();
    }
    if (collector_) {
        collector_->success            = false;
        collector_->total_cost_time_us = currentTimeUs() - collector_->start_time_us;
    }
    done_cv_.notify_all();
}

// 生产路径由 P2PConnectorAsyncReadContextChecker 单线程按间隔调用
// `checkDone()`，不存在与其它调用方并发重入，故无实际竞态。UT 为同线程同步调用。若未来多线程驱动
// checkDone，需整体重审。
void P2PConnectorAsyncReadContext::checkDone() {
    if (done()) {
        return;
    }
    if (!calls_ready_.load(std::memory_order_acquire)) {
        return;
    }
    if (!tp_sync_result_->done()) {
        tp_sync_result_->checkDone();
    }
    if (!server_call_result_->done()) {
        server_call_result_->checkDone();
    }
    const bool both_done = tp_sync_result_->done() && server_call_result_->done();
    if (!both_done) {
        return;
    }

    applyMergedReadOutcome(mergeReadResultsWhenBothDone());
}

P2PConnectorAsyncReadContext::MergedReadOutcome P2PConnectorAsyncReadContext::mergeReadResultsWhenBothDone() const {
    MergedReadOutcome outcome;
    outcome.success = tp_sync_result_->success() && server_call_result_->success();
    if (!outcome.success) {
        if (tp_sync_result_->done() && !tp_sync_result_->success()) {
            outcome.error_code    = tp_sync_result_->errorCode();
            outcome.error_message = tp_sync_result_->errorMessage();
        } else if (server_call_result_->done() && !server_call_result_->success()) {
            outcome.error_code    = server_call_result_->error_code;
            outcome.error_message = server_call_result_->error_message;
        }
    }
    return outcome;
}

void P2PConnectorAsyncReadContext::applyMergedReadOutcome(const MergedReadOutcome& outcome) {
    const bool  success    = outcome.success;
    ErrorCode   error_code = outcome.error_code;
    std::string error_message{outcome.error_message};

    const bool read_result_unconfirmed =
        !no_transfer_ && tp_sync_result_ && tp_sync_result_->done() && !tp_sync_result_->success();
    // StartLoad 侧的 TRANSFER_NOT_DONE / READ_CANCELLED 同样意味着物理传输结果未定：
    // prefill 还没存完（或刚被取消）时，decode worker 的 RDMA 写入可能仍在途，目标块
    // 必须走 lease 持有路径，不能立即归还。
    const bool holdable_outcome = error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
                                  || error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
    if (!success && transfer_not_done_hold_ms_ > 0 && (read_result_unconfirmed || holdable_outcome)) {
        const int64_t now_ms = currentTimeMs();
        const int64_t hold_until_ms =
            transfer_not_done_hold_ms_ > std::numeric_limits<int64_t>::max() - now_ms ?
                std::numeric_limits<int64_t>::max() :
                now_ms + transfer_not_done_hold_ms_;
        RTP_LLM_LOG_WARNING("[PD-DIAG] %s, retaining Decode target blocks for at most %ldms, unique_key=%s, "
                            "hold_until_ms=%ld, tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                            readOutcomeHoldReason(error_code),
                            transfer_not_done_hold_ms_,
                            uniqueKey().c_str(),
                            hold_until_ms,
                            tp_sync_result_->totalCostTimeUs(),
                            server_call_result_->totalCostTimeUs());
        // The request may finish now, but the checker keeps this context (and
        // resource_) alive until all Decode workers report their physical RDMA
        // operations stopped or the configured hold deadline is reached.
        lease_all_ranks_stopped_.store(false, std::memory_order_relaxed);
        lease_hold_until_ms_.store(hold_until_ms, std::memory_order_relaxed);
        lease_poll_interval_ms_.store(kLeasePollInitialIntervalMs, std::memory_order_relaxed);
        lease_poll_next_ms_.store(std::min(now_ms + kLeasePollInitialIntervalMs, hold_until_ms),
                                  std::memory_order_relaxed);
        lease_poll_retry_count_.store(0, std::memory_order_relaxed);
        lease_hold_pending_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            done_          = true;
            success_       = false;
            error_code_    = error_code;
            error_message_ = std::move(error_message);
        }
        done_cv_.notify_all();
        collector_->success                  = false;
        collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
        collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
        collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        done_       = true;
        success_    = success;
        error_code_ = error_code;
        error_message_.assign(error_message);
        done_cv_.notify_all();
    }
    RTP_LLM_LOG_DEBUG("[PD-DIAG] P2PAsyncRead done, unique_key=%s, success=%d, error_code=%d, "
                      "total_cost_us=%ld, tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                     uniqueKey().c_str(),
                     success_,
                     static_cast<int>(error_code_),
                     currentTimeUs() - collector_->start_time_us,
                     tp_sync_result_->totalCostTimeUs(),
                     server_call_result_->totalCostTimeUs());
    collector_->success                  = success_;
    collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
    collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
    collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
}

ErrorInfo P2PConnectorAsyncReadContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ErrorInfo(error_code_, error_message_);
}

bool P2PConnectorAsyncReadContext::needCancel() const {
    if (!calls_ready_.load(std::memory_order_acquire)) {
        return false;
    }
    if (lease_hold_pending_.load()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return false;
        }
    }
    if (tp_sync_result_->done() && !tp_sync_result_->success()) {
        return true;
    }
    if (server_call_result_->done() && !server_call_result_->success()) {
        return true;
    }
    return false;
}

void P2PConnectorAsyncReadContext::cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    cancel_requested_.store(true, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return;
        }
        if (kickoff_state_ == KickoffState::QUEUED) {
            // Cancellation won before the async task claimed kickoff, so no
            // StartLoad/READ RPC can be created after the target is released.
            if (!done_) {
                done_          = true;
                success_       = false;
                error_code_    = ErrorCode::CANCELLED;
                error_message_ = "P2P async read cancelled before kickoff";
            }
            done_cv_.notify_all();
            return;
        }
        if (kickoff_state_ == KickoffState::STARTING) {
            // The task may already be creating StartLoad/READ calls. It owns
            // the context until setCallResults()/markStartFailed(); completing
            // here would allow Decode target blocks to be released too early.
            return;
        }
    }

    std::string unique_key = uniqueKey();

    if (!server_call_result_->done()) {
        server_call_result_->cancel();
    }

    // 如果 tp_sync_result_ 未完成，通过 P2PBroadcastClient 发送 CANCEL 请求（至多成功发起一次）
    if (!tp_sync_result_->done() && tp_broadcast_client) {
        bool expected = false;
        if (tp_cancel_broadcast_triggered_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            auto cancel_result = tp_broadcast_client->cancel(
                unique_key, P2PConnectorBroadcastType::CANCEL_READ, request_deadline_ms_);
            if (!cancel_result) {
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
            } else if (!cancel_result->done()) {
                cancel_result->checkDone();
            }
        }
    }
}

bool P2PConnectorAsyncReadContext::expireLeaseHoldIfNeeded() {
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return false;
    }
    const int64_t now           = currentTimeMs();
    const int64_t hold_until_ms = lease_hold_until_ms_.load(std::memory_order_relaxed);
    if (hold_until_ms <= 0 || now < hold_until_ms) {
        return false;
    }
    RTP_LLM_LOG_ERROR("expireLeaseHoldIfNeeded: lease hold deadline reached with active transfers; releasing Decode "
                      "target resources, unique_key=%s retries=%d hold_until_ms=%ld",
                      uniqueKey().c_str(),
                      lease_poll_retry_count_.load(std::memory_order_relaxed),
                      hold_until_ms);
    lease_hold_pending_.store(false, std::memory_order_release);
    lease_hold_until_ms_.store(0, std::memory_order_relaxed);
    lease_poll_next_ms_.store(0, std::memory_order_relaxed);
    return true;
}

void P2PConnectorAsyncReadContext::pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    if (!calls_ready_.load(std::memory_order_acquire) || expireLeaseHoldIfNeeded()) {
        return;
    }
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return;
    }
    const int64_t now           = currentTimeMs();
    const int64_t hold_until_ms = lease_hold_until_ms_.load(std::memory_order_relaxed);
    if (lease_all_ranks_stopped_.load(std::memory_order_acquire) || !tp_broadcast_client) {
        return;
    }
    if (now < lease_poll_next_ms_.load(std::memory_order_relaxed)) {
        return;
    }

    const std::string unique_key = uniqueKey();
    const int         retry      = lease_poll_retry_count_.fetch_add(1, std::memory_order_relaxed);

    const int64_t remaining_hold_ms = hold_until_ms > 0 ? hold_until_ms - now : kLeasePollRpcTimeoutMs;
    const int64_t poll_timeout_ms   = std::max<int64_t>(1, std::min(kLeasePollRpcTimeoutMs, remaining_hold_ms));
    auto          result            = tp_broadcast_client->queryLeaseStatus(unique_key, poll_timeout_ms);
    const int64_t after_poll_ms     = currentTimeMs();
    if (hold_until_ms > 0 && after_poll_ms >= hold_until_ms) {
        RTP_LLM_LOG_ERROR("pollLeaseIfNeeded: lease hold deadline reached after status poll; releasing Decode target "
                          "resources, unique_key=%s retries=%d hold_until_ms=%ld",
                          unique_key.c_str(),
                          retry,
                          hold_until_ms);
        lease_hold_pending_.store(false, std::memory_order_release);
        lease_hold_until_ms_.store(0, std::memory_order_relaxed);
        lease_poll_next_ms_.store(0, std::memory_order_relaxed);
        return;
    }
    if (!result.success) {
        RTP_LLM_LOG_WARNING("pollLeaseIfNeeded: QUERY_LEASE_STATUS broadcast failed, unique_key=%s retry=%d",
                            unique_key.c_str(),
                            retry);
        // Backoff: double interval up to max.
        const int64_t interval =
            std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
        lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
        lease_poll_next_ms_.store(std::min(after_poll_ms + interval, hold_until_ms), std::memory_order_relaxed);
        return;
    }

    if (result.allStopped()) {
        RTP_LLM_LOG_DEBUG("pollLeaseIfNeeded: all ranks stopped, unique_key=%s retry=%d", unique_key.c_str(), retry);
        lease_all_ranks_stopped_.store(true, std::memory_order_release);
        lease_hold_pending_.store(false, std::memory_order_release);
        lease_hold_until_ms_.store(0, std::memory_order_relaxed);
        lease_poll_next_ms_.store(0, std::memory_order_relaxed);
        return;
    }

    // Not yet stopped — continue polling with backoff.
    const int64_t interval =
        std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
    lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
    lease_poll_next_ms_.store(std::min(after_poll_ms + interval, hold_until_ms), std::memory_order_relaxed);
}

/*----------------------------------------------- P2PConnectorAcceptedWriteContext
 * -------------------------------------------------*/
void P2PConnectorAcceptedWriteContext::waitDone() {
    // done() is always true, no blocking
}

bool P2PConnectorAcceptedWriteContext::done() const {
    return true;
}

bool P2PConnectorAcceptedWriteContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContextChecker
 * -------------------------------------------------*/
P2PConnectorAsyncReadContextChecker::~P2PConnectorAsyncReadContextChecker() {
    stop();
}

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                               const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    metrics_reporter_    = metrics_reporter;
    tp_broadcast_client_ = tp_broadcast_client;
    check_done_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&P2PConnectorAsyncReadContextChecker::checkOnce, this),
                                            5 * 1000,  // 5ms
                                            "P2PConnectorAsyncReadContextCheckerThread");
    if (!check_done_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorAsyncReadContextChecker init failed: check_done_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorAsyncReadContextChecker init success");
    return true;
}

void P2PConnectorAsyncReadContextChecker::stop() {
    if (check_done_thread_) {
        check_done_thread_->stop();
        check_done_thread_.reset();
    }
}

void P2PConnectorAsyncReadContextChecker::addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (!context) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    async_contexts_.push_back(context);
}

size_t P2PConnectorAsyncReadContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    return async_contexts_.size();
}

void P2PConnectorAsyncReadContextChecker::checkOnce() {
    int64_t start_time_us = currentTimeUs();

    // Three-phase structure to keep async_contexts_mutex_ off the slow check/cancel path —
    // see DingTalk doc §7 for the 8-min production stall this fixes:
    //   Phase 1 (under lock): snapshot the shared_ptr list only.
    //   Phase 2 (no lock):    run checkDone / lease poll / cancel decisions on the snapshot.
    //   Phase 3 (under lock): reclaim done contexts from the live vector.
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> to_poll;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> to_cancel;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> snapshot;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        snapshot = async_contexts_;
    }

    for (const auto& async_context : snapshot) {
        async_context->checkDone();
        async_context->expireLeaseHoldIfNeeded();
        if (async_context->needLeasePoll()) {
            to_poll.push_back(async_context);
        }
        if (async_context->needCancel()) {
            RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContextChecker checkOnce: needCancel, unique_key: %s",
                              async_context->uniqueKey().c_str());
            to_cancel.push_back(async_context);
        }
    }

    // A lease query can block up to 500ms. Poll at most one context per sweep
    // and rotate the selection so a failure burst cannot linearly stall normal
    // async-read completion checks. Expired holds were already released above.
    if (!to_poll.empty()) {
        const size_t poll_index = lease_poll_cursor_++ % to_poll.size();
        to_poll[poll_index]->pollLeaseIfNeeded(tp_broadcast_client_);
    }

    // cancel() is idempotent (server_call_result_->done() / tp_sync_result_->done() guards inside).
    // shared_ptr held in to_cancel keeps each context alive even if Phase 3's erase removes it.
    for (auto& async_context : to_cancel) {
        async_context->cancel(tp_broadcast_client_);
    }

    size_t inflight_after = 0;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> failed_contexts;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        auto it = async_contexts_.begin();
        while (it != async_contexts_.end()) {
            if ((*it)->done() && !(*it)->resourceHoldPending()) {
                if (!(*it)->success()) {
                    failed_contexts.push_back(*it);
                }
                it = async_contexts_.erase(it);
                continue;
            }
            ++it;
        }
        inflight_after = async_contexts_.size();
    }

    for (const auto& async_context : failed_contexts) {
        auto error = async_context->errorInfo();
        RTP_LLM_LOG_WARNING("P2PConnectorAsyncReadContextChecker checkOnce: async read failed, unique_key: %s, error: %s",
                            async_context->uniqueKey().c_str(),
                            error.ToString().c_str());
    }

    if (metrics_reporter_) {
        auto collector                     = std::make_shared<DecodeSchedulerStatusMetricsCollector>();
        collector->check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->inflight_context_count  = inflight_after;
        metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerStatusMetricsCollector>(nullptr, collector.get());
    }
}

P2PConnectorAsyncWriteContext::P2PConnectorAsyncWriteContext(KVCacheResourcePtr                  resource,
                                                             std::string                         unique_key,
                                                             int64_t                             deadline_ms,
                                                             P2PConnectorBroadcastType           type,
                                                             std::shared_ptr<P2PBroadcastClient> client,
                                                             int64_t                             control_timeout_ms,
                                                             Settle                              settle,
                                                             std::function<void()>               on_released,
                                                             kmonitor::MetricsReporterPtr        metrics_reporter):
    resource_(std::move(resource)),
    unique_key_(std::move(unique_key)),
    deadline_ms_(deadline_ms),
    type_(type),
    client_(std::move(client)),
    control_timeout_ms_(control_timeout_ms),
    settle_(std::move(settle)),
    on_released_(std::move(on_released)),
    metrics_reporter_(std::move(metrics_reporter)) {
    metrics_.prefill = type_ == HANDLE_WRITE;
}

void P2PConnectorAsyncWriteContext::setPlannedBytes(int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.planned_bytes = bytes;
}

void P2PConnectorAsyncWriteContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return done_; });
}

bool P2PConnectorAsyncWriteContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool P2PConnectorAsyncWriteContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && error_.ok();
}

ErrorInfo P2PConnectorAsyncWriteContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

bool P2PConnectorAsyncWriteContext::beginKickoff() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (kickoff_started_ || released_) {
        return false;
    }
    if (cancelled_ || currentTimeMs() >= deadline_ms_) {
        finishLocked(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired or cancelled before start"));
        return false;
    }
    kickoff_started_ = true;
    return true;
}

void P2PConnectorAsyncWriteContext::setCallResults(std::shared_ptr<P2PBroadcastClient::Result> result) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_result_      = std::move(result);
    calls_ready_       = true;
    metrics_.submitted = true;
    if (!start_result_) {
        registration_done_ = true;
        cancelled_         = true;
        error_ = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback START broadcast failed");
    }
}

void P2PConnectorAsyncWriteContext::finishLocked(const ErrorInfo& error) {
    if (released_) {
        return;
    }
    if (error_.ok()) {
        error_ = error;
    }
    done_     = true;
    released_ = true;
    if (on_released_) {
        on_released_();
        on_released_ = {};
    }
    settle_ = {};
    resource_.reset();
    if (metrics_reporter_) {
        metrics_.error              = error_;
        metrics_.no_transfer        = !calls_ready_ && error_.ok();
        metrics_.total_cost_time_us = currentTimeUs() - start_time_us_;
        metrics_.hold_time_us       = metrics_.total_cost_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, WriteSchedulerMetricsCollector>(nullptr, &metrics_);
    }
    start_result_.reset();
    control_result_.reset();
    cv_.notify_all();
}

void P2PConnectorAsyncWriteContext::finishWithoutTransfer(const ErrorInfo& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!calls_ready_) {
        finishLocked(error.ok() && currentTimeMs() >= deadline_ms_ ?
                         ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired before completion") :
                         error);
    }
}

void P2PConnectorAsyncWriteContext::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_) {
        return;
    }
    cancelled_ = true;
    if (error_.ok()) {
        error_ = ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled");
    }
    done_ = true;
    cv_.notify_all();
    if (!kickoff_started_) {
        finishLocked(error_);
    }
}

bool P2PConnectorAsyncWriteContext::registrationSucceeded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registration_done_ && registration_success_ && !cancelled_;
}

bool P2PConnectorAsyncWriteContext::registrationDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registration_done_;
}

bool P2PConnectorAsyncWriteContext::resourceHoldPending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !released_;
}

void P2PConnectorAsyncWriteContext::checkDone(const std::shared_ptr<autil::ThreadPool>& control_pool) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (released_) {
        return;
    }
    if (currentTimeMs() >= deadline_ms_) {
        cancelled_ = true;
        if (error_.ok()) {
            error_ = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback transfer deadline exceeded");
        }
    }
    if (cancelled_) {
        done_ = true;
        cv_.notify_all();
        if (!kickoff_started_) {
            finishLocked(error_);
            return;
        }
    }
    if (!calls_ready_) {
        return;
    }
    if (!registration_done_ && start_result_) {
        start_result_->checkDone();
        if (start_result_->done()) {
            registration_done_    = true;
            registration_success_ = start_result_->success();
            if (!registration_success_) {
                cancelled_ = true;
                if (error_.ok()) {
                    error_ = ErrorInfo(start_result_->errorCode(), start_result_->errorMessage());
                }
            }
        }
    }
    if (!registration_done_ && !cancelled_) {
        return;
    }
    if (!control_result_) {
        if (control_submitting_) {
            return;
        }
        control_submitting_ = true;
        const auto operation = cancelled_ ? WRITE_CANCEL : WRITE_QUERY;
        auto submit = [self = shared_from_this(), operation]() {
            std::shared_ptr<P2PBroadcastClient::TpBroadcastResult> result;
            try {
                result = self->client_->controlWriteAsync(self->unique_key_, self->type_, operation,
                                                          self->deadline_ms_, self->control_timeout_ms_);
            } catch (...) {
                // An unavailable control path never proves that borrowed buffers are no longer in use.
            }
            std::lock_guard<std::mutex> guard(self->mutex_);
            self->control_result_ = std::move(result);
            self->control_submitting_ = false;
        };
        lock.unlock();
        if (!control_pool) {
            submit();
        } else if (control_pool->pushTask(std::move(submit), false, false) != autil::ThreadPoolBase::ERROR_NONE) {
            lock.lock();
            control_submitting_ = false;
        }
        return;
    }
    if (!control_result_->waitDone(1)) {
        return;
    }
    const auto status = P2PBroadcastClient::writeStatus(control_result_);
    control_result_.reset();
    for (const auto& rank : status.ranks) {
        if (rank.error_code() != ErrorCodePB::NONE_ERROR
            || (rank.has_lease_status() && rank.lease_status().stopped() && !rank.write_success())) {
            cancelled_ = true;
            if (error_.ok()) {
                error_ = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback worker failed");
            }
            done_ = true;
            cv_.notify_all();
            break;
        }
    }
    if (!status.allStopped()) {
        return;
    }
    ErrorInfo result = error_;
    if (result.ok() && (!status.allSucceeded() || !registration_success_)) {
        result = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback transfer failed");
    }
    if (result.ok() && currentTimeMs() >= deadline_ms_) {
        result = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired before settle");
    }
    if (result.ok() && settle_) {
        try {
            result = settle_(resource_);
        } catch (const std::exception& e) {
            result = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what());
        } catch (...) {
            result = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback settle failed");
        }
    }
    finishLocked(result);
}

P2PConnectorAsyncWriteContextChecker::~P2PConnectorAsyncWriteContextChecker() {
    stop();
}

bool P2PConnectorAsyncWriteContextChecker::init(int interval_ms) {
    if (interval_ms <= 0 || thread_) {
        return false;
    }
    interval_ms_ = interval_ms;
    control_pool_ = std::make_shared<autil::ThreadPool>(4, 1024, nullptr, "P2PWriteControl");
    if (!control_pool_->start()) {
        return false;
    }
    thread_ =
        autil::LoopThread::createLoopThread([this]() { checkOnce(); }, int64_t(interval_ms) * 1000, "P2PWriteChecker");
    return thread_ != nullptr;
}

bool P2PConnectorAsyncWriteContextChecker::addContext(
    const std::shared_ptr<P2PConnectorAsyncWriteContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || !context) {
        return false;
    }
    const auto it = contexts_.find(context->uniqueKey());
    if (it != contexts_.end() && !it->second->resourceHoldPending()) {
        contexts_.erase(it);
    }
    return contexts_.emplace(context->uniqueKey(), context).second;
}

void P2PConnectorAsyncWriteContextChecker::cancelAll() {
    std::vector<std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        for (const auto& [key, context] : contexts_) {
            contexts.push_back(context);
        }
    }
    for (const auto& context : contexts) {
        context->cancel();
    }
}

void P2PConnectorAsyncWriteContextChecker::stop() {
    cancelAll();
    if (thread_) {
        thread_->stop();
        thread_.reset();
    }
    while (inflightContextCount() != 0) {
        checkOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
    if (control_pool_) {
        control_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    }
}

size_t P2PConnectorAsyncWriteContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.size();
}

void P2PConnectorAsyncWriteContextChecker::checkOnce() {
    std::vector<std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [key, context] : contexts_) {
            contexts.push_back(context);
        }
    }
    for (const auto& context : contexts) {
        context->checkDone(control_pool_);
        if (!context->resourceHoldPending()) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  it = contexts_.find(context->uniqueKey());
            if (it != contexts_.end() && it->second == context) {
                contexts_.erase(it);
            }
        }
    }
}

}  // namespace rtp_llm
