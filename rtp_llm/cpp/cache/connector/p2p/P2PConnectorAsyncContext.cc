#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <set>

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

void P2PConnectorAsyncReadContext::onDone(DoneCallback callback) {
    if (!callback) return;
    ErrorInfo error;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_) {
            done_callbacks_.push_back(std::move(callback));
            return;
        }
        error = ErrorInfo(error_code_, error_message_);
    }
    callback(std::move(error));
}

void P2PConnectorAsyncReadContext::dispatchDoneCallbacks() {
    std::vector<DoneCallback> callbacks;
    ErrorInfo error;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_) return;
        callbacks.swap(done_callbacks_);
        error = ErrorInfo(error_code_, error_message_);
    }
    for (auto& callback : callbacks) callback(error);
}

bool P2PConnectorAsyncReadContext::setCallResults(
    const std::shared_ptr<P2PBroadcastClient::Result>& tp_sync_result,
    const std::shared_ptr<DecodeLoadHelper::Result>&  server_call_result) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        tp_sync_result_     = tp_sync_result;
        server_call_result_ = server_call_result;
        kickoff_state_      = KickoffState::CALLS_READY;
        registerReadCompletionCallbacksLocked();
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
    DoneNotifier done_notifier{*this};
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
        collector_->decode_schedule_success            = false;
        collector_->decode_schedule_total_cost_time_us = currentTimeUs() - collector_->decode_schedule_start_time_us;
    }
    done_cv_.notify_all();
    notify();
}

void P2PConnectorAsyncReadContext::markKickoffUnconfirmed(const ErrorInfo& error_info) {
    DoneNotifier done_notifier{*this};
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_) {
            done_ = true;
            success_ = false;
            error_code_ = error_info.code();
            error_message_ = error_info.ToString();
        }
        // StartLoad/READ may already be in flight even if no result was returned.
        kickoff_state_ = KickoffState::CALLS_READY;
        if (!lease_hold_pending_.load(std::memory_order_acquire)) {
            beginLeaseHold();
        }
    }
    cancel_requested_.store(true, std::memory_order_release);
    calls_ready_.store(true, std::memory_order_release);
    done_cv_.notify_all();
    notify();
}

void P2PConnectorAsyncReadContext::beginLeaseHold() {
    if (no_transfer_) {
        return;
    }
    const int64_t now_ms      = P2PNotification::nowMs();
    lease_all_ranks_stopped_.store(false, std::memory_order_relaxed);
    lease_poll_interval_ms_.store(kLeasePollInitialIntervalMs, std::memory_order_relaxed);
    lease_poll_next_ms_.store(now_ms + kLeasePollInitialIntervalMs, std::memory_order_relaxed);
    lease_poll_retry_count_.store(0, std::memory_order_relaxed);
    lease_hold_start_us_.store(P2PNotification::nowUs(), std::memory_order_relaxed);
    lease_hold_pending_.store(true, std::memory_order_release);
}

bool P2PConnectorAsyncReadContext::expireTransferDeadlineIfNeeded() {
    DoneNotifier done_notifier{*this};
    checkDone();
    if (transfer_stop_at_ms_ <= 0 || P2PNotification::nowMs() < transfer_stop_at_ms_) {
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
        error_message_ = "observer_role=DECODE stage=transfer_deadline selected_path=decode_deadline key=" + uniqueKey()
                         + " transfer_deadline_ms=" + std::to_string(transfer_deadline_ms_) + " request_deadline_ms="
                         + std::to_string(request_deadline_ms_) + ": P2P transfer deadline exceeded";
        if (kickoff_state_ != KickoffState::QUEUED) {
            beginLeaseHold();
        }
    }
    cancel_requested_.store(true, std::memory_order_release);
    RTP_LLM_LOG_WARNING("P2P Decode terminal, stage=transfer_deadline unique_key=%s error_code=%d error_name=%s "
                        "transfer_deadline_ms=%ld request_deadline_ms=%ld; physical completion may still be pending",
                        uniqueKey().c_str(), static_cast<int>(ErrorCode::GENERATE_TIMEOUT),
                        ErrorCodeToString(ErrorCode::GENERATE_TIMEOUT).c_str(), transfer_deadline_ms_, request_deadline_ms_);
    if (collector_) {
        collector_->decode_schedule_success            = false;
        collector_->decode_schedule_total_cost_time_us = currentTimeUs() - collector_->decode_schedule_start_time_us;
    }
    done_cv_.notify_all();
    notify();
    return true;
}

// Only the checker merges outcomes; CQ callbacks publish results and notify.
void P2PConnectorAsyncReadContext::checkDone() {
    DoneNotifier done_notifier{*this};
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
        const auto  read_observed    = tp_sync_result_->firstError();
        const auto  prefill_observed = server_call_result_->firstError();
        const char* selected_path    = first.order == read_observed.order ? "decode_read" : "prefill_start_load";
        RTP_LLM_LOG_WARNING("P2P error selection, key=%s selected_path=%s selected_code=%d selected_order=%llu "
                            "observed_after_selection_read_code=%d read_order=%llu read_error=%s "
                            "observed_after_selection_prefill_code=%d prefill_order=%llu prefill_error=%s",
                            uniqueKey().c_str(),
                            selected_path,
                            static_cast<int>(first.error.code()),
                            static_cast<unsigned long long>(first.order),
                            static_cast<int>(read_observed.error.code()),
                            static_cast<unsigned long long>(read_observed.order),
                            read_observed.error.ToString().c_str(),
                            static_cast<int>(prefill_observed.error.code()),
                            static_cast<unsigned long long>(prefill_observed.order),
                            prefill_observed.error.ToString().c_str());
        const std::string message =
            "observer_role=DECODE stage=p2p_read_merge selected_path=" + std::string(selected_path)
            + " key=" + uniqueKey() + " selected_order=" + std::to_string(first.order) + ": " + first.error.ToString();
        applyMergedReadOutcome({false, first.error.code(), message});
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
            outcome.error_message = "observer_role=DECODE stage=p2p_read_merge selected_path=decode_read key="
                                    + uniqueKey() + ": " + tp_sync_result_->errorMessage();
        } else if (server_call_result_->done() && !server_call_result_->success()) {
            outcome.error_code    = server_call_result_->error_code;
            outcome.error_message = "observer_role=DECODE stage=p2p_read_merge selected_path=prefill_start_load key="
                                    + uniqueKey() + ": " + server_call_result_->error_message;
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
        RTP_LLM_LOG_WARNING("[PD-DIAG] %s, retaining Decode target blocks until physical completion, unique_key=%s, "
                            "tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                            readOutcomeHoldReason(error_code),
                            uniqueKey().c_str(),
                            tp_sync_result_->totalCostTimeUs(),
                            server_call_result_->totalCostTimeUs());
        done_cv_.notify_all();
        notify();
        if (collector_) {
            collector_->decode_schedule_success            = false;
            collector_->decode_schedule_total_cost_time_us = currentTimeUs() - collector_->decode_schedule_start_time_us;
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
                     currentTimeUs() - collector_->decode_schedule_start_time_us,
                     tp_sync_result_->totalCostTimeUs(),
                     server_call_result_->totalCostTimeUs());
    collector_->decode_schedule_success            = success_;
    collector_->decode_schedule_total_cost_time_us = currentTimeUs() - collector_->decode_schedule_start_time_us;
}

ErrorInfo P2PConnectorAsyncReadContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ErrorInfo(error_code_, error_message_);
}

bool P2PConnectorAsyncReadContext::needCancel() const {
    if (!calls_ready_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return kickoff_state_ == KickoffState::STARTING
               && cancel_requested_.load(std::memory_order_acquire)
               && !cancel_confirmed_.load(std::memory_order_acquire);
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
    try {
        if (tp_sync_result_->done() && !tp_sync_result_->success()) {
            return true;
        }
        if (server_call_result_->done() && !server_call_result_->success()) {
            return true;
        }
    } catch (...) {
        // A fatal RPC outcome still requires cancellation. The checker also calls
        // this from its wakeup calculation, outside checkOnce's exception boundary.
        return true;
    }
    return false;
}

void P2PConnectorAsyncReadContext::cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    DoneNotifier done_notifier{*this};
    checkDone();
    cancel_requested_.store(true, std::memory_order_release);

    std::shared_ptr<DecodeLoadHelper::Result> server_call;
    std::shared_ptr<P2PBroadcastClient::Result> read_call;
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
                error_message_ = "observer_role=DECODE stage=kickoff selected_path=decode_cancel key=" + uniqueKey()
                                 + ": P2P async read cancelled before kickoff";
                if (collector_) {
                    collector_->decode_schedule_success            = false;
                    collector_->decode_schedule_total_cost_time_us = currentTimeUs() - collector_->decode_schedule_start_time_us;
                }
            }
            done_cv_.notify_all();
            notify();
            return;
        }
        // STARTING may be blocked while creating RPCs. Install the worker
        // cancellation fence independently, but keep local completion/lease
        // release gated by kickoff publication and physical stop confirmation.
        server_call = server_call_result_;
        read_call = tp_sync_result_;
    }

    std::string unique_key = uniqueKey();

    if (server_call && !server_call->done()) {
        server_call->cancel();
    }

    // A failed client-side READ result does not prove that every worker handler
    // has stopped. A D-timeout lease hold always broadcasts CANCEL_READ, even
    // if READ itself completed, so the cancel-done barrier cannot be skipped.
    const bool read_confirmed_success = read_call && read_call->done() && read_call->success();
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
                control_retry_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
                notify();
                throw;
            }
            if (!cancel_result) {
                control_retry_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
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
            control_retry_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
            tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
        }
    }
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
    const int64_t now           = P2PNotification::nowMs();
    if (lease_all_ranks_stopped_.load(std::memory_order_acquire) || !tp_broadcast_client) {
        return;
    }
    if (now < lease_poll_next_ms_.load(std::memory_order_relaxed)) {
        return;
    }

    const std::string unique_key = uniqueKey();
    const int         retry      = lease_poll_retry_count_.fetch_add(1, std::memory_order_relaxed);

    const int64_t poll_timeout_ms   = std::min(kLeasePollRpcTimeoutMs, lease_query_timeout_ms_);
    const auto    query_start_us    = P2PNotification::nowUs();
    auto          result            = tp_broadcast_client->queryLeaseStatus(unique_key, poll_timeout_ms);
    if (collector_) {
        collector_->decode_schedule_lease_query_time_us =
            std::max<int64_t>(0, collector_->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed)) + P2PNotification::nowUs() - query_start_us;
    }
    const int64_t after_poll_ms     = P2PNotification::nowMs();
    if (!result.success) {
        RTP_LLM_LOG_WARNING("pollLeaseIfNeeded: QUERY_LEASE_STATUS broadcast failed, unique_key=%s retry=%d",
                            unique_key.c_str(),
                            retry);
        // Backoff: double interval up to max.
        const int64_t interval =
            std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
        lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
        lease_poll_next_ms_.store(after_poll_ms + interval, std::memory_order_relaxed);
        return;
    }

    if (result.allStopped()) {
        const auto hold_start_us = lease_hold_start_us_.exchange(0);
        if (collector_ && hold_start_us > 0) {
            collector_->decode_schedule_lease_hold_time_us = P2PNotification::nowUs() - hold_start_us;
        }
        RTP_LLM_LOG_DEBUG("pollLeaseIfNeeded: all ranks stopped, unique_key=%s retry=%d", unique_key.c_str(), retry);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            resource_.reset();
        }
        lease_all_ranks_stopped_.store(true, std::memory_order_release);
        lease_hold_pending_.store(false, std::memory_order_release);
        lease_poll_next_ms_.store(0, std::memory_order_relaxed);
        return;
    }

    // Not yet stopped — continue polling with backoff.
    const int64_t interval =
        std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
    lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
    lease_poll_next_ms_.store(after_poll_ms + interval, std::memory_order_relaxed);
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
    registerReadCompletionCallbacksLocked();
    if (cancel_result_) {
        cancel_result_->setDoneCallback(completionCallback());
    }
}

void P2PConnectorAsyncReadContext::registerReadCompletionCallbacksLocked() {
    // Both registration orders must preserve timing and notification together.
    // Callers hold state_mutex_; callbacks keep only weak references to RPC results.
    const auto notify_done = completionCallback();
    if (tp_sync_result_) {
        tp_sync_result_->setDoneCallback([notify_done,
                                          collector   = collector_,
                                          no_transfer = no_transfer_,
                                          weak_result = std::weak_ptr<P2PBroadcastClient::Result>(tp_sync_result_)] {
            if (auto result = weak_result.lock(); result && collector && !no_transfer && result->done()) {
                collector->decode_schedule_tp_sync_cost_time_us = result->totalCostTimeUs();
            }
            notify_done();
        });
    }
    if (server_call_result_) {
        server_call_result_->setDoneCallback(
            [notify_done,
             collector   = collector_,
             weak_result = std::weak_ptr<DecodeLoadHelper::Result>(server_call_result_)] {
                if (auto result = weak_result.lock(); result && collector) {
                    collector->decode_schedule_server_call_cost_time_us = result->totalCostTimeUs();
                }
                notify_done();
            });
    }
}

bool P2PConnectorAsyncReadContext::needsControl() const {
    return (needCancel() && !tp_cancel_broadcast_triggered_.load() && P2PNotification::nowMs() >= control_retry_ms_.load())
           || needLeasePoll();
}

int64_t P2PConnectorAsyncReadContext::nextWakeupMs(bool allow_control) const {
    auto next = std::numeric_limits<int64_t>::max();
    if (!done() && transfer_stop_at_ms_ > 0) {
        next = transfer_stop_at_ms_;
    }
    if (resourceHoldPending()) {
        // Resource accounting must keep running even while a control job is blocked.
        next = std::min(next, P2PNotification::nowMs() + 1000);
        if (allow_control && calls_ready_.load(std::memory_order_acquire) && cancel_confirmed_.load()) {
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
        control_retry_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
        lease_poll_next_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
    } catch (...) {
        RTP_LLM_LOG_ERROR("P2P control RPC failed, retaining lease, unique_key=%s", uniqueKey().c_str());
        control_retry_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
        lease_poll_next_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
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
                                               const std::shared_ptr<autil::LockFreeThreadPool>& control_pool,
                                               const P2PConnectorSchedulerConfig&               config) {
    if (!control_pool || !config.topology || config.p2p_kv_leak_exit_threshold_percent <= 0
        || config.p2p_kv_leak_exit_threshold_percent > 100) {
        RTP_LLM_LOG_ERROR("P2P lease checker requires topology and a KV leak exit threshold in (0, 100]");
        return false;
    }
    group_block_bytes_.clear();
    total_kv_bytes_ = 0;
    for (const auto& group : config.topology->groups()) {
        const size_t block_bytes = (group.kvBlockStrideBytes() + group.kvScaleStrideBytes())
                                   * config.topology->layerIdsForGroup(group.tag).size();
        group_block_bytes_.emplace(group.tag, block_bytes);
        total_kv_bytes_ += block_bytes * group.block_num;
    }
    if (total_kv_bytes_ == 0) {
        RTP_LLM_LOG_ERROR("P2P lease checker requires nonzero total KV capacity");
        return false;
    }
    kv_leak_exit_threshold_percent_ = config.p2p_kv_leak_exit_threshold_percent;
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
    // 停机后 checkOnce 不再摘除上下文，这里对仍在跟踪的采样补一次上报，
    // 保证每个 async read 上下文恰好上报一次。
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> remaining;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        remaining.swap(async_contexts_);
    }
    for (const auto& context : remaining) {
        reportDecodeScheduleMetrics(context);
    }
}

void P2PConnectorAsyncReadContextChecker::reportDecodeScheduleMetrics(
    const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (!metrics_reporter_ || !context) {
        return;
    }
    auto collector = context->metricsCollector();
    if (!collector) {
        return;
    }
    int64_t unset_total_us = -1;
    collector->decode_schedule_total_cost_time_us.compare_exchange_strong(
        unset_total_us, currentTimeUs() - collector->decode_schedule_start_time_us);
    metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
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

void P2PConnectorAsyncReadContextChecker::checkRetainedResources(
    const std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>>& contexts) {
    // Count each physical (group, block) once, including shared prefix blocks.
    // Groups can have different block sizes and layer counts, so compare bytes.
    std::map<std::string, std::set<BlockIdxType>> held_blocks;
    size_t                              held_contexts  = 0;
    int64_t                             oldest_hold_ms = 0;
    const int64_t                       now_us         = P2PNotification::nowUs();
    for (const auto& context : contexts) {
        std::lock_guard<std::mutex> lock(context->state_mutex_);
        if (!context->resourceHoldPending() || !context->resource_) {
            continue;
        }
        ++held_contexts;
        const auto hold_start_us = context->lease_hold_start_us_.load();
        if (hold_start_us > 0) {
            oldest_hold_ms = std::max(oldest_hold_ms, (now_us - hold_start_us) / 1000);
        }
        const auto& groups = context->resource_->groupBlockIds();
        RTP_LLM_CHECK_WITH_INFO(groups.size() == group_block_bytes_.size(), "P2P retained KV topology mismatch");
        for (const auto& [tag, block_bytes] : group_block_bytes_) {
            for (const auto block : groups.blocks(tag)) {
                if (!isNullBlockIdx(block)) {
                    held_blocks[tag].insert(block);
                }
            }
        }
    }
    size_t held_bytes = 0;
    for (const auto& [tag, blocks] : held_blocks) {
        held_bytes += blocks.size() * group_block_bytes_.at(tag);
    }
    const double held_percent = total_kv_bytes_ > 0 ? 100.0 * held_bytes / total_kv_bytes_ : 0.0;
    if (metrics_reporter_) {
        P2PConnectorMetricsCollector collector;
        collector.decode_scheduler_retained_kv_bytes      = held_bytes;
        collector.decode_scheduler_retained_kv_percent    = held_percent;
        collector.decode_scheduler_retained_context_count = held_contexts;
        collector.decode_scheduler_oldest_kv_hold_ms      = oldest_hold_ms;
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &collector);
    }
    const int64_t now_ms = now_us / 1000;
    if (held_contexts > 0 && (last_leak_log_ms_ == 0 || now_ms - last_leak_log_ms_ >= 30 * 1000)) {
        RTP_LLM_LOG_WARNING("P2P retained KV: contexts=%zu held_bytes=%zu total_bytes=%zu held_percent=%.2f "
                            "exit_threshold_percent=%d oldest_hold_ms=%ld",
                            held_contexts,
                            held_bytes,
                            total_kv_bytes_,
                            held_percent,
                            kv_leak_exit_threshold_percent_,
                            oldest_hold_ms);
        last_leak_log_ms_ = now_ms;
    } else if (held_contexts == 0 && last_leak_log_ms_ != 0) {
        RTP_LLM_LOG_INFO("P2P retained KV: all held resources released after physical stop confirmation");
        last_leak_log_ms_ = 0;
    }
    // Exact strict comparison without overflowing size_t or rounding percentages.
    const size_t threshold_bytes = (total_kv_bytes_ / 100) * kv_leak_exit_threshold_percent_
                                   + (total_kv_bytes_ % 100) * kv_leak_exit_threshold_percent_ / 100;
    if (held_bytes > threshold_bytes) {
        RTP_LLM_LOG_ERROR("P2P retained KV exceeded exit threshold; aborting rank 0 with KV still held: "
                          "held_bytes=%zu total_bytes=%zu held_percent=%.2f exit_threshold_percent=%d",
                          held_bytes,
                          total_kv_bytes_,
                          held_percent,
                          kv_leak_exit_threshold_percent_);
        std::abort();
    }
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
        // Consuming a failed RPC completion can throw. Keep it inside the checker
        // thread: the context stays tracked, so the transfer deadline and lease
        // watchdog still converge instead of the process terminating.
        try {
            context->checkDone();
            context->expireTransferDeadlineIfNeeded();
            context->checkCancelDone();
            if (context->needsControl()) {
                controls.push_back(context);
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR(
                "P2PConnectorAsyncReadContextChecker checkOnce: async read context threw, unique_key: %s, error: %s",
                context->uniqueKey().c_str(),
                e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR(
                "P2PConnectorAsyncReadContextChecker checkOnce: async read context threw, unique_key: %s",
                context->uniqueKey().c_str());
        }
    }
    checkRetainedResources(snapshot);
    // Connection acquisition and lease queries can block. Keep them off both
    // the CQ consumer and deadline checker, with one job on the existing pool.
    if (!controls.empty() && control_pool_ && P2PNotification::nowMs() >= control_submit_retry_ms_
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
            control_submit_retry_ms_ = P2PNotification::nowMs() + kLeasePollInitialIntervalMs;
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
                reportDecodeScheduleMetrics(*it);
                it = async_contexts_.erase(it);
                continue;
            }
            ++it;
        }
        inflight_after = async_contexts_.size();
    }

    for (const auto& async_context : failed_contexts) {
        auto error = async_context->errorInfo();
        RTP_LLM_LOG_WARNING("P2P Decode async read failed, unique_key=%s error_code=%d error_name=%s error=%s",
                            async_context->uniqueKey().c_str(),
                            static_cast<int>(error.code()),
                            ErrorCodeToString(error.code()).c_str(),
                            error.ToString().c_str());
    }

    if (metrics_reporter_) {
        auto collector = std::make_shared<P2PConnectorMetricsCollector>();
        collector->decode_scheduler_check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->decode_scheduler_inflight_context_count  = inflight_after;
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
    }
}

P2PConnectorAsyncWriteContext::P2PConnectorAsyncWriteContext(std::string                         unique_key,
                                                             int64_t                             deadline_ms,
                                                             std::shared_ptr<P2PBroadcastClient> client,
                                                             int64_t                             control_timeout_ms,
                                                             int64_t                             lease_query_timeout_ms,
                                                             std::function<void()>               on_released,
                                                             kmonitor::MetricsReporterPtr        metrics_reporter):
    unique_key_(std::move(unique_key)),
    deadline_ms_(deadline_ms),
    client_(std::move(client)),
    control_timeout_ms_(control_timeout_ms),
    lease_query_timeout_ms_(lease_query_timeout_ms),
    metrics_reporter_(std::move(metrics_reporter)),
    on_released_(std::move(on_released)) {}

bool P2PConnectorAsyncWriteContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool P2PConnectorAsyncWriteContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && error_.ok();
}

void P2PConnectorAsyncWriteContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return done_; });
}

void P2PConnectorAsyncWriteContext::onDone(DoneCallback callback) {
    if (!callback)
        return;
    ErrorInfo error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!done_) {
            done_callbacks_.push_back(std::move(callback));
            return;
        }
        error = error_;
    }
    callback(std::move(error));
}

void P2PConnectorAsyncWriteContext::dispatchDoneCallbacks() {
    std::vector<DoneCallback> callbacks;
    ErrorInfo                 error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!done_)
            return;
        callbacks.swap(done_callbacks_);
        error = error_;
    }
    for (auto& callback : callbacks)
        callback(error);
}

void P2PConnectorAsyncWriteContext::setCallResults(std::shared_ptr<P2PBroadcastClient::Result> result) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_result_      = std::move(result);
    if (!start_result_) {
        failLocked(
            ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback START broadcast failed"));
    }
    registerWriteCompletionCallbacksLocked();
}

bool P2PConnectorAsyncWriteContext::beginKickoff() {
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> lock(mutex_);
    if (kickoff_started_ || released_)
        return false;
    if (cancelled_ || currentTimeMs() >= deadline_ms_) {
        tryFinishLocked(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired or cancelled before start"));
        return false;
    }
    kickoff_started_ = true;
    return true;
}

void P2PConnectorAsyncWriteContext::checkDone() {
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_)
        return;
    const auto first = FirstError::earlier(start_result_ ? start_result_->firstError() : FirstError::Snapshot{},
                                           peer_result_ ? peer_result_->firstError() : FirstError::Snapshot{});
    if (first.error.hasError())
        failLocked(first.error);
    if (start_result_ && start_result_->done()) {
        // WRITE RPC completion proves local send completion, but an RDMA notification
        // failure still needs receiver stop confirmation before releasing the source.
        if (start_result_->success() || start_result_->rpcSuccess())
            transfer_stopped_ = true;
        if (!start_result_->success())
            failLocked(ErrorInfo(start_result_->errorCode(), start_result_->errorMessage()));
    }
    checkPeerResultLocked();
    if (transfer_stopped_ || (peer_pending_ && peer_stopped_ && !calls_ready_ && !ready_submitting_)) {
        tryFinishLocked(error_);
        if (released_)
            return;
    }
}

void P2PConnectorAsyncWriteContext::expireTransferDeadlineIfNeeded() {
    checkDone();
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> lock(mutex_);
    if (!released_ && currentTimeMs() >= deadline_ms_) {
        failLocked(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback transfer deadline exceeded"));
    }
}

ErrorInfo P2PConnectorAsyncWriteContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void P2PConnectorAsyncWriteContext::cancel(const ErrorInfo& error) {
    DoneNotifier                done_notifier{*this};
    checkDone();
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_)
        return;
    failLocked(error);
    done_ = true;
    cv_.notify_all();
    if (!kickoff_started_)
        tryFinishLocked(error_);
}

void P2PConnectorAsyncWriteContext::checkCancelDone() {
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_)
        return;
    bool completed = false;
    if (peer_cancel_result_ && peer_cancel_result_->done()) {
        if (peer_cancel_result_->success() && peer_cancel_result_->responses().size() == 1)
            peer_stopped_ = peer_stopped_ || peer_cancel_result_->responses().front().stopped();
        peer_cancel_result_.reset();
        completed = true;
    }
    if (control_result_ && control_result_->done()) {
        bool stopped = control_result_->success();
        for (const auto& response : control_result_->responses())
            stopped = stopped && response.has_p2p_response();
        transfer_stopped_ = transfer_stopped_ || stopped;
        control_result_.reset();
        completed = true;
    }
    if (completed && !control_result_ && !peer_cancel_result_) {
        control_next_ms_     = currentTimeMs() + control_interval_ms_;
        control_interval_ms_ = std::min(control_interval_ms_ * 2, kLeasePollMaxIntervalMs);
    }
    if (transfer_stopped_ || cancelled_)
        tryFinishLocked(error_);
}

void P2PConnectorAsyncWriteContext::failStopIfLeaseUnconfirmed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || currentTimeMs() < holdUntilMsLocked())
        return;
    RTP_LLM_LOG_ERROR("failStopIfLeaseUnconfirmed: D+query timeout reached without a valid all-rank stopped result; "
                      "aborting rank 0 without releasing writeback resources, unique_key=%s fail_stop_ms=%ld",
                      unique_key_.c_str(),
                      holdUntilMsLocked());
    std::abort();
}

bool P2PConnectorAsyncWriteContext::resourceHoldPending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !released_;
}

std::function<void()> P2PConnectorAsyncWriteContext::completionCallback() const {
    std::weak_ptr<P2PNotification> weak = std::atomic_load(&notification_);
    return [weak] {
        if (auto notification = weak.lock())
            notification->notify();
    };
}

void P2PConnectorAsyncWriteContext::notify() const {
    if (auto notification = std::atomic_load(&notification_))
        notification->notify();
}

void P2PConnectorAsyncWriteContext::setNotification(const std::shared_ptr<P2PNotification>& notification) {
    std::atomic_store(&notification_, notification);
    std::lock_guard<std::mutex> lock(mutex_);
    registerWriteCompletionCallbacksLocked();
}

void P2PConnectorAsyncWriteContext::registerWriteCompletionCallbacksLocked() {
    const auto notify_done = completionCallback();
    if (start_result_)
        start_result_->setDoneCallback(notify_done);
    if (peer_result_)
        peer_result_->setProgressCallback(notify_done);
    notify_done();
}

int64_t P2PConnectorAsyncWriteContext::nextWakeupMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  hold_until = holdUntilMsLocked();
    if (released_)
        return hold_until;
    if (!cancelled_)
        return deadline_ms_;
    auto next = hold_until;
    if (!control_submitting_ && !control_result_ && !peer_cancel_result_) {
        const bool peer_control   = peer_pending_ && !peer_stopped_ && peer_cancel_;
        const bool worker_control = calls_ready_ && !transfer_stopped_ && client_;
        if (peer_control || worker_control)
            next = std::min(next, control_next_ms_);
    }
    return next;
}

void P2PConnectorAsyncWriteContext::runControl(const std::shared_ptr<autil::ThreadPool>& control_pool) {
    bool       worker_control;
    PeerCancel peer_cancel;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (released_ || !cancelled_ || control_submitting_ || control_result_ || peer_cancel_result_
            || currentTimeMs() < control_next_ms_)
            return;
        if (peer_pending_ && !peer_stopped_)
            peer_cancel = peer_cancel_;
        worker_control = calls_ready_ && !transfer_stopped_ && client_;
        if (!peer_cancel && !worker_control)
            return;
        control_submitting_ = true;
    }
    auto submit = [self = shared_from_this(), peer_cancel = std::move(peer_cancel), worker_control] {
        int64_t timeout;
        int64_t deadline;
        {
            std::lock_guard<std::mutex> lock(self->mutex_);
            deadline = self->deadline_ms_;
            timeout  = std::min(self->holdUntilMsLocked() - currentTimeMs(), self->control_timeout_ms_);
        }
        std::shared_ptr<DecodeWriteHelper::CancelResult>       peer_result;
        std::shared_ptr<P2PBroadcastClient::TpBroadcastResult> worker_result;
        try {
            if (worker_control && timeout > 0) {
                worker_result = self->client_->controlWriteAsync(self->unique_key_, CANCEL_WRITE, deadline, timeout);
            }
        } catch (...) {}
        try {
            if (peer_cancel)
                peer_result = peer_cancel();
        } catch (...) {}
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->peer_cancel_result_ = std::move(peer_result);
        self->control_result_     = std::move(worker_result);
        self->control_submitting_ = false;
        self->control_next_ms_    = currentTimeMs() + self->control_interval_ms_;
        if (!self->peer_cancel_result_ && !self->control_result_)
            self->control_interval_ms_ = std::min(self->control_interval_ms_ * 2, kLeasePollMaxIntervalMs);
        const auto notify_done    = self->completionCallback();
        if (self->peer_cancel_result_)
            self->peer_cancel_result_->setProgressCallback(notify_done);
        if (self->control_result_)
            self->control_result_->setProgressCallback(notify_done);
        notify_done();
    };
    if (control_pool && control_pool->pushTask(submit, false, false) == autil::ThreadPoolBase::ERROR_NONE)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    control_submitting_ = false;
    control_next_ms_     = currentTimeMs() + control_interval_ms_;
    control_interval_ms_ = std::min(control_interval_ms_ * 2, kLeasePollMaxIntervalMs);
    notify();
}

bool P2PConnectorAsyncWriteContext::beginWorkerCall() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_ || released_)
        return false;
    calls_ready_       = true;
    metrics_.submitted = true;
    return true;
}

void P2PConnectorAsyncWriteContext::finishWithoutTransfer(const ErrorInfo& error) {
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> lock(mutex_);
    if (!calls_ready_ && !ready_submitting_) {
        tryFinishLocked(error.ok() && currentTimeMs() >= deadline_ms_ ?
                            ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired before completion") :
                            error);
    }
}

int64_t P2PConnectorAsyncWriteContext::deadlineMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deadline_ms_;
}

void P2PConnectorAsyncWriteContext::setPlannedBytes(int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.planned_bytes = bytes;
}

void P2PConnectorAsyncWriteContext::setReadyHandler(ReadyHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_handler_ = std::move(handler);
}

ErrorInfo P2PConnectorAsyncWriteContext::acceptWriteReady(const P2PConnectorWriteReadyRequestPB& request) {
    ReadyHandler handler;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancelled_ || released_ || currentTimeMs() >= deadline_ms_) {
            return ErrorInfo(ErrorCode::CANCELLED, "writeback is no longer accepting ready notifications");
        }
        if (!ready_handler_ || !peer_pending_ || request.deadline_ms() <= currentTimeMs()
            || request.deadline_ms() > deadline_ms_) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid writeback ready deadline or phase");
        }
        const auto encoded = request.SerializeAsString();
        if (!ready_request_.empty()) {
            return ready_request_ == encoded ? ErrorInfo::OkStatus() :
                                               ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback ready range changed");
        }
        ready_request_    = encoded;
        deadline_ms_      = std::min(deadline_ms_, request.deadline_ms());
        ready_submitting_ = true;
        handler           = ready_handler_;
    }
    notify();
    ErrorInfo error;
    try {
        error = handler(request);
    } catch (const std::exception& e) {
        error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what());
    } catch (...) {
        error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback ready handler failed");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_submitting_ = false;
    }
    if (error.hasError())
        cancel(error);
    notify();
    return error;
}

bool P2PConnectorAsyncWriteContext::beginPeerCall(PeerCancel cancel) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_ || released_)
        return false;
    peer_pending_ = true;
    peer_cancel_  = std::move(cancel);
    return true;
}

void P2PConnectorAsyncWriteContext::setPeerCallResult(std::shared_ptr<DecodeWriteHelper::Result> result) {
    std::lock_guard<std::mutex> lock(mutex_);
    peer_result_ = std::move(result);
    if (!peer_result_) {
        failLocked(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "StartWrite submission failed"));
    }
    registerWriteCompletionCallbacksLocked();
}

void P2PConnectorAsyncWriteContext::failLocked(const ErrorInfo& error) {
    if (error_.ok())
        error_ = error;
    if (!cancelled_) {
        cancelled_ = true;
        done_      = true;
        cv_.notify_all();
        notify();
    }
}

void P2PConnectorAsyncWriteContext::checkPeerResultLocked() {
    if (!peer_pending_ || peer_stopped_ || !peer_result_ || !peer_result_->done())
        return;
    if (peer_result_->rpcSuccess() && peer_result_->responses().size() == 1) {
        const auto response = peer_result_->responses().front();
        peer_stopped_       = response.stopped();
        if (response.error_code() != ErrorCodePB::NONE_ERROR) {
            failLocked(ErrorInfo(transRPCErrorCode(static_cast<ErrorCodePB>(response.error_code())),
                                 response.error_message()));
        } else if (!peer_stopped_ || (ready_request_.empty() && response.accepted_block_count() != 0)) {
            failLocked(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                 "StartWrite returned before receive stopped or without ready"));
        }
    } else {
        failLocked(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "StartWrite RPC failed"));
    }
    peer_result_.reset();
}

void P2PConnectorAsyncWriteContext::tryFinishLocked(const ErrorInfo& error) {
    if (released_)
        return;
    if (error_.ok())
        error_ = error;
    if (!calls_ready_)
        transfer_stopped_ = true;
    if (!transfer_stopped_ || ready_submitting_)
        return;
    // Before any sender submission, cancellation also fences late WriteReady callbacks.
    if (peer_pending_ && !peer_stopped_ && (calls_ready_ || error_.ok()))
        return;
    finishLocked(error_);
}

void P2PConnectorAsyncWriteContext::finishLocked(const ErrorInfo& error) {
    if (released_)
        return;
    if (error_.ok())
        error_ = error;
    done_     = true;
    released_ = true;
    if (on_released_) {
        on_released_();
        on_released_ = {};
    }
    ready_handler_ = {};
    peer_cancel_   = {};
    peer_result_.reset();
    peer_cancel_result_.reset();
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
    notify();
}

int64_t P2PConnectorAsyncWriteContext::holdUntilMsLocked() const {
    return lease_query_timeout_ms_ > std::numeric_limits<int64_t>::max() - deadline_ms_ ?
               std::numeric_limits<int64_t>::max() :
               deadline_ms_ + lease_query_timeout_ms_;
}

bool P2PConnectorAsyncWriteContext::retentionExpired() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_ && currentTimeMs() >= holdUntilMsLocked();
}

P2PConnectorAsyncWriteContextChecker::~P2PConnectorAsyncWriteContextChecker() {
    stop();
}

bool P2PConnectorAsyncWriteContextChecker::init() {
    if (thread_.joinable())
        return false;
    control_pool_ = std::make_shared<autil::ThreadPool>(4, 1024, nullptr, "P2PWriteControl");
    if (!control_pool_->start())
        return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = true;
    }
    stopping_.store(false);
    thread_ = std::thread([this] {
        while (!stopping_.load()) {
            const auto generation = notification_->generation();
            checkOnce();
            if (!stopping_.load())
                notification_->waitUntil(generation, nextWakeupMs());
        }
    });
    return true;
}

bool P2PConnectorAsyncWriteContextChecker::addContext(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context) {
    if (!context)
        return false;
    context->setNotification(notification_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_)
            return false;
        const auto it = contexts_.find(context->uniqueKey());
        if (it != contexts_.end() && it->second->retentionExpired())
            contexts_.erase(it);
        if (!contexts_.emplace(context->uniqueKey(), context).second)
            return false;
    }
    notification_->notify();
    return true;
}

void P2PConnectorAsyncWriteContextChecker::cancelAll() {
    std::vector<std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = false;
        for (const auto& [key, context] : contexts_)
            contexts.push_back(context);
    }
    for (const auto& context : contexts)
        context->cancel();
    notification_->notify();
}

void P2PConnectorAsyncWriteContextChecker::stop() {
    cancelAll();
    stopping_.store(true);
    notification_->notify();
    if (thread_.joinable())
        thread_.join();
    while (inflightContextCount() != 0) {
        const auto generation = notification_->generation();
        checkOnce();
        if (inflightContextCount() != 0)
            notification_->waitUntil(generation, nextWakeupMs());
    }
    if (control_pool_)
        control_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
}

size_t P2PConnectorAsyncWriteContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(
        contexts_.begin(), contexts_.end(), [](const auto& item) { return item.second->resourceHoldPending(); });
}

std::shared_ptr<P2PConnectorAsyncWriteContext>
P2PConnectorAsyncWriteContextChecker::findContext(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = contexts_.find(key);
    return it == contexts_.end() ? nullptr : it->second;
}

int64_t P2PConnectorAsyncWriteContextChecker::nextWakeupMs() const {
    auto                        next = std::numeric_limits<int64_t>::max();
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, context] : contexts_)
        next = std::min(next, context->nextWakeupMs());
    if (next == std::numeric_limits<int64_t>::max())
        return next;
    const auto delay_ms = std::max<int64_t>(0, next - currentTimeMs());
    const auto now_ms   = P2PNotification::nowMs();
    return now_ms + std::min(delay_ms, std::numeric_limits<int64_t>::max() - now_ms);
}

void P2PConnectorAsyncWriteContextChecker::checkOnce() {
    std::vector<std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [key, context] : contexts_)
            contexts.push_back(context);
    }
    for (const auto& context : contexts) {
        context->checkDone();
        context->expireTransferDeadlineIfNeeded();
        context->checkCancelDone();
        context->failStopIfLeaseUnconfirmed();
        context->runControl(control_pool_);
        if (context->retentionExpired()) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  it = contexts_.find(context->uniqueKey());
            if (it != contexts_.end() && it->second == context)
                contexts_.erase(it);
        }
    }
}

}  // namespace rtp_llm
