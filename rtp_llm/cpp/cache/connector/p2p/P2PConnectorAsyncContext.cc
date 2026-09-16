#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <cstdlib>
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
        const auto notify_done = completionCallback();
        tp_sync_result_->setDoneCallback([notify_done,
                                          collector   = collector_,
                                          no_transfer = no_transfer_,
                                          weak_result = std::weak_ptr<P2PBroadcastClient::Result>(tp_sync_result_)] {
            if (auto result = weak_result.lock(); result && collector && !no_transfer && result->done()) {
                collector->tp_sync_cost_time_us = result->totalCostTimeUs();
            }
            notify_done();
        });
        server_call_result_->setDoneCallback(
            [notify_done,
             collector   = collector_,
             weak_result = std::weak_ptr<DecodeLoadHelper::Result>(server_call_result_)] {
                if (auto result = weak_result.lock(); result && collector) {
                    collector->server_call_cost_time_us = result->totalCostTimeUs();
                }
                notify_done();
            });
    }
    calls_ready_.store(true, std::memory_order_release);
    notify();
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
            // startAsyncReadCalls failed before publishing any RPC result, so
            // there is no transfer that can still target resource_.
            lease_hold_pending_.store(false, std::memory_order_release);
            notify();
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
    notify();
}

void P2PConnectorAsyncReadContext::beginLeaseHold() {
    if (no_transfer_) {
        return;
    }
    const int64_t now_ms      = currentTimeMs();
    const int64_t deadline_ms = transfer_deadline_ms_ > 0 ? transfer_deadline_ms_ : now_ms;
    const int64_t hold_until_ms = lease_query_timeout_ms_ > std::numeric_limits<int64_t>::max() - deadline_ms ?
                                      std::numeric_limits<int64_t>::max() :
                                      deadline_ms + lease_query_timeout_ms_;
    lease_all_ranks_stopped_.store(false, std::memory_order_relaxed);
    lease_hold_until_ms_.store(hold_until_ms, std::memory_order_relaxed);
    lease_poll_interval_ms_.store(kLeasePollInitialIntervalMs, std::memory_order_relaxed);
    lease_poll_next_ms_.store(std::min(now_ms + kLeasePollInitialIntervalMs, hold_until_ms),
                              std::memory_order_relaxed);
    lease_poll_retry_count_.store(0, std::memory_order_relaxed);
    lease_hold_start_us_.store(currentTimeUs(), std::memory_order_relaxed);
    lease_hold_pending_.store(true, std::memory_order_release);
}

bool P2PConnectorAsyncReadContext::expireTransferDeadlineIfNeeded() {
    checkDone();
    if (transfer_deadline_ms_ <= 0 || currentTimeMs() < transfer_deadline_ms_) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return false;
        }
        done_          = true;
        success_       = false;
        error_code_    = ErrorCode::GENERATE_TIMEOUT;
        error_message_ = "P2P transfer deadline exceeded";
        if (kickoff_state_ != KickoffState::QUEUED) {
            beginLeaseHold();
        }
    }
    cancel_requested_.store(true, std::memory_order_release);
    if (collector_) {
        collector_->success            = false;
        collector_->total_cost_time_us = currentTimeUs() - collector_->start_time_us;
    }
    done_cv_.notify_all();
    notify();
    return true;
}

// Only the checker merges outcomes; CQ callbacks publish results and notify.
void P2PConnectorAsyncReadContext::checkDone() {
    std::lock_guard<std::mutex> check_lock(check_mutex_);
    if (done()) {
        return;
    }
    if (!calls_ready_.load(std::memory_order_acquire)) {
        return;
    }
    tp_sync_result_->checkDone();  // Non-blocking metric snapshot.
    const auto first = FirstError::earlier(tp_sync_result_->firstError(), server_call_result_->firstError());
    if (first.error.hasError()) {
        applyMergedReadOutcome({false, first.error.code(), first.error.ToString()});
        return;
    }
    if (tp_sync_result_->done() && server_call_result_->done()) {
        applyMergedReadOutcome(mergeReadResultsWhenBothDone());
    }
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

    // Failure reporting does not imply physical completion of the READ handlers.
    const bool read_result_unconfirmed = !no_transfer_ && tp_sync_result_ && !tp_sync_result_->success();
    // StartLoad can report an unconfirmed transfer even when the READ RPC succeeded.
    // Its exact first cause is preserved while the physical lease remains held.
    const bool holdable_outcome = outcome.error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
                                  || outcome.error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
    if (!success && lease_query_timeout_ms_ > 0 && (read_result_unconfirmed || holdable_outcome)) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (done_)
                return;
            beginLeaseHold();
            done_          = true;
            success_       = false;
            error_code_    = error_code;
            error_message_ = error_message;
        }
        const int64_t hold_until_ms = lease_hold_until_ms_.load(std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING("[PD-DIAG] %s, retaining Decode target blocks until physical completion, unique_key=%s, "
                            "fail_stop_ms=%ld, tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                            readOutcomeHoldReason(error_code),
                            uniqueKey().c_str(),
                            hold_until_ms,
                            tp_sync_result_->totalCostTimeUs(),
                            server_call_result_->totalCostTimeUs());
        done_cv_.notify_all();
        notify();
        if (collector_) {
            collector_->success                  = false;
            collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_)
            return;
        done_       = true;
        success_    = success;
        error_code_ = error_code;
        error_message_.assign(error_message);
        done_cv_.notify_all();
        notify();
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
        return !cancel_confirmed_.load(std::memory_order_acquire);
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
    checkDone();
    cancel_requested_.store(true, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_ && !lease_hold_pending_.load(std::memory_order_acquire)) {
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
                if (collector_) {
                    collector_->success            = false;
                    collector_->total_cost_time_us = currentTimeUs() - collector_->start_time_us;
                }
            }
            done_cv_.notify_all();
            notify();
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

    // A failed client-side READ result does not prove that every worker handler
    // has stopped. A D-timeout lease hold always broadcasts CANCEL_READ, even
    // if READ itself completed, so the cancel-done barrier cannot be skipped.
    const bool read_confirmed_success = tp_sync_result_->done() && tp_sync_result_->success();
    const bool lease_stop_confirmation_required = lease_hold_pending_.load(std::memory_order_acquire);
    if ((lease_stop_confirmation_required || !read_confirmed_success) && tp_broadcast_client
        && !cancel_confirmed_.load(std::memory_order_acquire)) {
        bool expected = false;
        if (tp_cancel_broadcast_triggered_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            std::shared_ptr<P2PBroadcastClient::Result> cancel_result;
            try {
                cancel_result = tp_broadcast_client->cancel(
                    unique_key, P2PConnectorBroadcastType::CANCEL_READ, request_deadline_ms_, 0, 0);
            } catch (...) {
                // This invocation owns the CAS; allow a retry without releasing
                // the target lease if RPC creation throws.
                control_retry_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
                notify();
                throw;
            }
            if (!cancel_result) {
                control_retry_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
            } else {
                std::lock_guard<std::mutex> lock(state_mutex_);
                cancel_result_ = std::move(cancel_result);
                cancel_result_->setDoneCallback(completionCallback());
            }
        }
    }
    notify();
}

void P2PConnectorAsyncReadContext::checkCancelDone() {
    std::shared_ptr<P2PBroadcastClient::Result> cancel_result;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cancel_result = cancel_result_;
    }
    if (!cancel_result) {
        return;
    }
    if (!cancel_result->done()) {
        cancel_result->checkDone();
    }
    if (!cancel_result->done()) {
        return;
    }
    if (cancel_result->success()) {
        cancel_confirmed_.store(true, std::memory_order_release);
        RTP_LLM_LOG_DEBUG("CANCEL_READ confirmed by all ranks, unique_key=%s", uniqueKey().c_str());
        return;
    }

    RTP_LLM_LOG_WARNING("CANCEL_READ broadcast failed; retrying while lease is held, unique_key=%s",
                        uniqueKey().c_str());
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (cancel_result_ == cancel_result) {
            cancel_result_.reset();
            control_retry_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
            tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
        }
    }
}

void P2PConnectorAsyncReadContext::failStopIfLeaseUnconfirmed() {
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return;
    }
    const int64_t now           = currentTimeMs();
    const int64_t hold_until_ms = lease_hold_until_ms_.load(std::memory_order_relaxed);
    if (hold_until_ms <= 0 || now < hold_until_ms) {
        return;
    }
    RTP_LLM_LOG_ERROR("failStopIfLeaseUnconfirmed: D+query timeout reached without a valid all-rank stopped result; "
                      "aborting rank 0 without releasing Decode target resources, unique_key=%s retries=%d "
                      "fail_stop_ms=%ld",
                      uniqueKey().c_str(),
                      lease_poll_retry_count_.load(std::memory_order_relaxed),
                      hold_until_ms);
    std::abort();
}

void P2PConnectorAsyncReadContext::pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    if (!calls_ready_.load(std::memory_order_acquire)) {
        return;
    }
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return;
    }
    if (!cancel_confirmed_.load(std::memory_order_acquire)) {
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
    const auto    query_start_us    = currentTimeUs();
    auto          result            = tp_broadcast_client->queryLeaseStatus(unique_key, poll_timeout_ms);
    if (collector_) {
        collector_->lease_query_time_us =
            std::max<int64_t>(0, collector_->lease_query_time_us) + currentTimeUs() - query_start_us;
    }
    const int64_t after_poll_ms     = currentTimeMs();
    if (hold_until_ms > 0 && after_poll_ms >= hold_until_ms) {
        failStopIfLeaseUnconfirmed();
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
        const auto hold_start_us = lease_hold_start_us_.exchange(0);
        if (collector_ && hold_start_us > 0) {
            collector_->lease_hold_time_us = currentTimeUs() - hold_start_us;
        }
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

std::function<void()> P2PConnectorAsyncReadContext::completionCallback() const {
    std::weak_ptr<P2PNotification> weak = std::atomic_load(&notification_);
    return [weak] {
        if (auto notification = weak.lock()) {
            notification->notify();
        }
    };
}

void P2PConnectorAsyncReadContext::notify() const {
    if (auto notification = std::atomic_load(&notification_)) {
        notification->notify();
    }
}

void P2PConnectorAsyncReadContext::setNotification(const std::shared_ptr<P2PNotification>& notification) {
    std::atomic_store(&notification_, notification);
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Registration and setCallResults serialize. Already-completed RPCs notify
    // inline, closing both completion-before-registration orderings.
    if (tp_sync_result_) {
        tp_sync_result_->setDoneCallback(completionCallback());
    }
    if (server_call_result_) {
        server_call_result_->setDoneCallback(completionCallback());
    }
    if (cancel_result_) {
        cancel_result_->setDoneCallback(completionCallback());
    }
}

bool P2PConnectorAsyncReadContext::needsControl() const {
    return (needCancel() && !tp_cancel_broadcast_triggered_.load() && currentTimeMs() >= control_retry_ms_.load())
           || needLeasePoll();
}

int64_t P2PConnectorAsyncReadContext::nextWakeupMs(bool allow_control) const {
    auto next = std::numeric_limits<int64_t>::max();
    if (!done() && transfer_deadline_ms_ > 0) {
        next = transfer_deadline_ms_;
    }
    if (resourceHoldPending()) {
        next = std::min(next, lease_hold_until_ms_.load());
        if (allow_control && cancel_confirmed_.load()) {
            next = std::min(next, lease_poll_next_ms_.load());
        }
    }
    if (allow_control && needCancel() && !tp_cancel_broadcast_triggered_.load()) {
        next = std::min(next, control_retry_ms_.load());
    }
    return next;
}

void P2PConnectorAsyncReadContext::runControl(const std::shared_ptr<P2PBroadcastClient>& client) {
    try {
        if (needCancel()) {
            cancel(client);
        }
        pollLeaseIfNeeded(client);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR(
            "P2P control RPC failed, retaining lease, unique_key=%s error=%s", uniqueKey().c_str(), e.what());
        control_retry_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
        lease_poll_next_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
    } catch (...) {
        RTP_LLM_LOG_ERROR("P2P control RPC failed, retaining lease, unique_key=%s", uniqueKey().c_str());
        control_retry_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
        lease_poll_next_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs);
    }
    notify();
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

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&               metrics_reporter,
                                               const std::shared_ptr<P2PBroadcastClient>&        tp_broadcast_client,
                                               const std::shared_ptr<autil::LockFreeThreadPool>& control_pool) {
    if (!control_pool) {
        return false;
    }
    metrics_reporter_    = metrics_reporter;
    tp_broadcast_client_ = tp_broadcast_client;
    control_pool_        = control_pool;
    stopping_.store(false);
    async_read_check_thread_ = std::thread([this] {
        while (!stopping_.load()) {
            const auto generation = notification_->generation();
            checkOnce();
            int64_t next = std::numeric_limits<int64_t>::max();
            {
                std::lock_guard<std::mutex> lock(async_contexts_mutex_);
                const bool                  allow_control = !control_busy_->load();
                for (const auto& context : async_contexts_) {
                    next = std::min(next, context->nextWakeupMs(false));
                    if (allow_control) {
                        next = std::min(next, std::max(control_submit_retry_ms_, context->nextWakeupMs(true)));
                    }
                }
            }
            if (!stopping_.load()) {
                notification_->waitUntil(generation, next);
            }
        }
    });
    return true;
}

void P2PConnectorAsyncReadContextChecker::stop() {
    stopping_.store(true);
    notification_->notify();
    if (async_read_check_thread_.joinable()) {
        async_read_check_thread_.join();
    }
}

void P2PConnectorAsyncReadContextChecker::addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (!context) {
        return;
    }
    context->setNotification(notification_);
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    async_contexts_.push_back(context);
    notification_->notify();
}

size_t P2PConnectorAsyncReadContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    return async_contexts_.size();
}

void P2PConnectorAsyncReadContextChecker::checkOnce() {
    int64_t start_time_us = currentTimeUs();

    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> snapshot;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        snapshot = async_contexts_;
    }
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> controls;
    for (const auto& context : snapshot) {
        context->checkDone();
        context->expireTransferDeadlineIfNeeded();
        context->checkCancelDone();
        context->failStopIfLeaseUnconfirmed();
        if (context->needsControl()) {
            controls.push_back(context);
        }
    }
    // Connection acquisition and lease queries can block. Keep them off both
    // the CQ consumer and deadline checker, with one job on the existing pool.
    if (!controls.empty() && control_pool_ && currentTimeMs() >= control_submit_retry_ms_
        && !control_busy_->exchange(true)) {
        auto                           context = controls[lease_poll_cursor_++ % controls.size()];
        std::weak_ptr<P2PNotification> weak    = notification_;
        const auto                     busy    = control_busy_;
        auto                           ret     = control_pool_->pushTask(
            [context, client = tp_broadcast_client_, busy, weak] {
                context->runControl(client);
                busy->store(false);
                if (auto notification = weak.lock()) {
                    notification->notify();
                }
            },
            false);
        if (ret != autil::ThreadPoolBase::ERROR_NONE) {
            busy->store(false);
            // Queue saturation is retried on a bounded timer, not a busy loop.
            control_submit_retry_ms_ = currentTimeMs() + kLeasePollInitialIntervalMs;
        }
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

}  // namespace rtp_llm
