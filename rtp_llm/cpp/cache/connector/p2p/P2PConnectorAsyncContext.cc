#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <set>

namespace rtp_llm {

namespace {
constexpr int64_t kLeasePollInitialIntervalMs = 10;
constexpr int64_t kLeasePollMaxIntervalMs     = 100;
constexpr int64_t kLeasePollRpcTimeoutMs      = 500;
constexpr int64_t kCancelRetryMaxBaseMs       = 800;
constexpr int64_t kCancelWarningIntervalMs    = 30 * 1000;

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
    if (!callback)
        return;
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
    std::vector<DoneCallback>          callbacks;
    std::vector<std::function<void()>> release_callbacks;
    ErrorInfo                          error;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_)
            return;
        callbacks.swap(done_callbacks_);
        if (!resourceHoldPending()) {
            release_callbacks.swap(target_release_callbacks_);
        }
        error = ErrorInfo(error_code_, error_message_);
    }
    for (auto& callback : callbacks)
        callback(error);
    for (auto& callback : release_callbacks)
        callback();
}

void P2PConnectorAsyncReadContext::onTargetSafeToRelease(std::function<void()> callback) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_ || resourceHoldPending()) {
            target_release_callbacks_.push_back(std::move(callback));
            return;
        }
    }
    callback();
}

bool P2PConnectorAsyncReadContext::setCallResults(const std::shared_ptr<P2PBroadcastClient::Result>& tp_sync_result,
                                                  const std::shared_ptr<DecodeLoadHelper::Result>& server_call_result) {
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
            if (resourceHoldPending()) {
                releaseLeaseHoldLocked(/*all_ranks_stopped=*/false, P2PNotification::nowUs());
            }
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
            done_          = true;
            success_       = false;
            error_code_    = error_info.code();
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
    if (no_transfer_ || lease_released_ || resourceHoldPending()) {
        return;
    }
    const int64_t now_ms = P2PNotification::nowMs();
    lease_all_ranks_stopped_.store(false, std::memory_order_relaxed);
    lease_poll_interval_ms_.store(kLeasePollInitialIntervalMs, std::memory_order_relaxed);
    lease_poll_next_ms_.store(now_ms + kLeasePollInitialIntervalMs, std::memory_order_relaxed);
    lease_poll_retry_count_.store(0, std::memory_order_relaxed);
    lease_query_state_.store(LeaseQueryState::Pending, std::memory_order_relaxed);
    lease_hold_start_us_.store(P2PNotification::nowUs(), std::memory_order_relaxed);
    lease_hold_pending_.store(true, std::memory_order_release);
}

void P2PConnectorAsyncReadContext::releaseLeaseHoldLocked(bool all_ranks_stopped, int64_t now_us) {
    const auto hold_start_us = lease_hold_start_us_.exchange(0);
    if (collector_ && hold_start_us > 0) {
        collector_->decode_schedule_lease_hold_time_us = now_us - hold_start_us;
    }
    resource_.reset();
    lease_released_ = true;
    lease_all_ranks_stopped_.store(all_ranks_stopped, std::memory_order_release);
    lease_poll_next_ms_.store(0, std::memory_order_relaxed);
    lease_hold_pending_.store(false, std::memory_order_release);
}

bool P2PConnectorAsyncReadContext::expireTransferDeadlineIfNeeded() {
    DoneNotifier done_notifier{*this};
    if (transfer_stop_at_ms_ <= 0) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // Before kickoff, also honor the wall-clock deadline used by StartLoad.
        // Converting the deadline to the steady clock rounds each clock separately.
        const bool expired_before_kickoff =
            kickoff_state_ == KickoffState::QUEUED && currentTimeMs() >= transfer_deadline_ms_;
        if (done_ || (!expired_before_kickoff && P2PNotification::nowMs() < transfer_stop_at_ms_)) {
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
                        uniqueKey().c_str(),
                        static_cast<int>(ErrorCode::GENERATE_TIMEOUT),
                        ErrorCodeToString(ErrorCode::GENERATE_TIMEOUT).c_str(),
                        transfer_deadline_ms_,
                        request_deadline_ms_);
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
    DoneNotifier                done_notifier{*this};
    std::lock_guard<std::mutex> check_lock(check_mutex_);
    // Observe call publication before reading the terminal state. An unconfirmed
    // kickoff publishes done without RPC results, then releases calls_ready_.
    if (!calls_ready_.load(std::memory_order_acquire)) {
        return;
    }
    if (done()) {
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
                                  || outcome.error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED
                                  || outcome.error_code == ErrorCode::GENERATE_TIMEOUT
                                  || outcome.error_code == ErrorCode::CANCELLED;
    if (!success && !no_transfer_ && (holdable_outcome || (lease_query_timeout_ms_ > 0 && read_result_unconfirmed))) {
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
            collector_->decode_schedule_success = false;
            collector_->decode_schedule_total_cost_time_us =
                currentTimeUs() - collector_->decode_schedule_start_time_us;
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_)
            return;
        if (success) {
            const auto first_token = server_call_result_->firstTokenTiming();
            const auto kv_complete = no_transfer_ ? std::nullopt : tp_sync_result_->kvPhysicalCompletionTimeUs();
            if (first_token.received_us > 0 && (kv_complete || no_transfer_)) {
                handoff_ready_time_us_ = std::max(first_token.received_us, kv_complete.value_or(0));
            }
            if (collector_ && first_token.produced_us) {
                collector_->prefill_first_token_to_decode_receive_us =
                    first_token.received_us - *first_token.produced_us;
                if (kv_complete) {
                    collector_->prefill_first_token_to_decode_kv_complete_us = *kv_complete - *first_token.produced_us;
                }
            }
        }
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
        return kickoff_state_ == KickoffState::STARTING && cancel_requested_.load(std::memory_order_acquire)
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

    std::shared_ptr<DecodeLoadHelper::Result>   server_call;
    std::shared_ptr<P2PBroadcastClient::Result> read_call;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_ && !lease_hold_pending_.load(std::memory_order_acquire)) {
            return;
        }
        if (!done_) {
            if (kickoff_state_ != KickoffState::QUEUED) {
                beginLeaseHold();
            }
            done_          = true;
            success_       = false;
            error_code_    = ErrorCode::CANCELLED;
            error_message_ = "observer_role=DECODE stage=kickoff selected_path=decode_cancel key=" + uniqueKey()
                             + ": P2P async read cancelled";
            if (collector_) {
                collector_->decode_schedule_success = false;
                collector_->decode_schedule_total_cost_time_us =
                    currentTimeUs() - collector_->decode_schedule_start_time_us;
            }
            done_cv_.notify_all();
            notify();
        }
        if (kickoff_state_ == KickoffState::QUEUED) {
            return;
        }
        // STARTING may be blocked while creating RPCs. Install the worker
        // cancellation fence independently, but keep local completion/lease
        // release gated by kickoff publication and physical stop confirmation.
        server_call = server_call_result_;
        read_call   = tp_sync_result_;
    }

    std::string unique_key = uniqueKey();

    if (server_call && !server_call->done()) {
        server_call->cancel();
    }

    // A failed client-side READ result does not prove that every worker handler
    // has stopped. A D-timeout lease hold always broadcasts CANCEL_READ, even
    // if READ itself completed, so the cancel-done barrier cannot be skipped.
    const bool read_confirmed_success           = read_call && read_call->done() && read_call->success();
    const bool lease_stop_confirmation_required = lease_hold_pending_.load(std::memory_order_acquire);
    if ((lease_stop_confirmation_required || !read_confirmed_success) && tp_broadcast_client
        && !cancel_confirmed_.load(std::memory_order_acquire)
        && P2PNotification::nowMs() >= control_retry_ms_.load(std::memory_order_acquire)) {
        bool expected = false;
        if (tp_cancel_broadcast_triggered_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            if (P2PNotification::nowMs() < control_retry_ms_.load(std::memory_order_acquire)
                || cancel_confirmed_.load(std::memory_order_acquire)) {
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
                notify();
                return;
            }
            std::shared_ptr<P2PBroadcastClient::Result> cancel_result;
            try {
                cancel_result = tp_broadcast_client->cancel(unique_key,
                                                            P2PConnectorBroadcastType::CANCEL_READ,
                                                            request_deadline_ms_,
                                                            read_call ? read_call->requestId().value_or(0) : 0,
                                                            0);
            } catch (const std::exception& error) {
                scheduleCancelRetry(error.what());
                notify();
                return;
            } catch (...) {
                scheduleCancelRetry("broadcast submission threw");
                notify();
                return;
            }
            if (!cancel_result) {
                scheduleCancelRetry("broadcast submission failed");
            } else {
                std::lock_guard<std::mutex> lock(state_mutex_);
                cancel_result_ = std::move(cancel_result);
                cancel_result_->setDoneCallback(completionCallback());
            }
        }
    }
    notify();
}

void P2PConnectorAsyncReadContext::scheduleCancelRetry(const char* reason) {
    // Only the owner of the pending broadcast may advance the backoff.
    const int64_t now_ms = P2PNotification::nowMs();
    const auto    jitter = std::hash<const void*>{}(this) % (cancel_retry_interval_ms_ / 4 + 1);
    control_retry_ms_.store(now_ms + cancel_retry_interval_ms_ + static_cast<int64_t>(jitter),
                            std::memory_order_release);
    cancel_retry_interval_ms_ = std::min(cancel_retry_interval_ms_ * 2, kCancelRetryMaxBaseMs);
    if (now_ms >= next_cancel_warning_ms_) {
        next_cancel_warning_ms_ = now_ms + kCancelWarningIntervalMs;
        RTP_LLM_LOG_WARNING(
            "CANCEL_READ failed; retrying while lease is held, unique_key=%s reason=%s", unique_key_.c_str(), reason);
    }
    tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
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

    const auto error = cancel_result->firstError().error.ToString();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (cancel_result_ == cancel_result) {
            cancel_result_.reset();
            scheduleCancelRetry(error.c_str());
        }
    }
}

void P2PConnectorAsyncReadContext::pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    if (lease_query_timeout_ms_ <= 0 || !calls_ready_.load(std::memory_order_acquire)) {
        return;
    }
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return;
    }
    if (!cancel_confirmed_.load(std::memory_order_acquire)) {
        return;
    }
    const int64_t now = P2PNotification::nowMs();
    if (lease_all_ranks_stopped_.load(std::memory_order_acquire) || !tp_broadcast_client) {
        return;
    }
    if (now < lease_poll_next_ms_.load(std::memory_order_relaxed)) {
        return;
    }

    const std::string unique_key = uniqueKey();
    const int         retry      = lease_poll_retry_count_.fetch_add(1, std::memory_order_relaxed);

    const int64_t poll_timeout_ms = std::min(kLeasePollRpcTimeoutMs, lease_query_timeout_ms_);
    const auto    query_start_us  = P2PNotification::nowUs();
    auto          result          = tp_broadcast_client->queryLeaseStatus(unique_key, poll_timeout_ms);
    if (collector_) {
        collector_->decode_schedule_lease_query_time_us =
            std::max<int64_t>(0, collector_->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed))
            + P2PNotification::nowUs() - query_start_us;
    }
    const int64_t after_poll_ms = P2PNotification::nowMs();
    if (!result.success) {
        lease_query_state_.store(LeaseQueryState::Failed, std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING("pollLeaseIfNeeded: QUERY_LEASE_STATUS failed, unique_key=%s retry=%d reason=%s",
                            unique_key.c_str(),
                            retry,
                            result.failure.c_str());
        // Backoff: double interval up to max.
        const int64_t interval =
            std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
        lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
        lease_poll_next_ms_.store(after_poll_ms + interval, std::memory_order_relaxed);
        return;
    }

    if (result.allStopped()) {
        DoneNotifier                done_notifier{*this};
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (resourceHoldPending()) {
            RTP_LLM_LOG_DEBUG(
                "pollLeaseIfNeeded: all ranks stopped, unique_key=%s retry=%d", unique_key.c_str(), retry);
            releaseLeaseHoldLocked(/*all_ranks_stopped=*/true, P2PNotification::nowUs());
        }
        return;
    }

    // Not yet stopped — continue polling with backoff.
    lease_query_state_.store(LeaseQueryState::WaitingForStop, std::memory_order_relaxed);
    const int64_t interval =
        std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
    lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
    lease_poll_next_ms_.store(after_poll_ms + interval, std::memory_order_relaxed);
}

std::function<void()> P2PConnectorAsyncReadContext::completionCallback() const {
    const auto callback = std::atomic_load(&wakeup_callback_);
    return callback ? *callback : std::function<void()>([] {});
}

void P2PConnectorAsyncReadContext::notify() const {
    if (const auto callback = std::atomic_load(&wakeup_callback_)) {
        (*callback)();
    }
}

void P2PConnectorAsyncReadContext::setWakeupCallback(std::function<void()> callback) {
    std::atomic_store(&wakeup_callback_, std::make_shared<const std::function<void()>>(std::move(callback)));
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
    return (needCancel() && !tp_cancel_broadcast_triggered_.load()
            && P2PNotification::nowMs() >= control_retry_ms_.load())
           || needLeasePoll();
}

int64_t P2PConnectorAsyncReadContext::nextWakeupMs(bool allow_control) const {
    auto next = std::numeric_limits<int64_t>::max();
    if (!done() && transfer_stop_at_ms_ > 0) {
        next = transfer_stop_at_ms_;
    }
    if (resourceHoldPending()) {
        if (lease_query_timeout_ms_ > 0 && allow_control && calls_ready_.load(std::memory_order_acquire)
            && cancel_confirmed_.load()) {
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
        lease_poll_next_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
    } catch (...) {
        RTP_LLM_LOG_ERROR("P2P control RPC failed, retaining lease, unique_key=%s", uniqueKey().c_str());
        lease_poll_next_ms_.store(P2PNotification::nowMs() + kLeasePollInitialIntervalMs);
    }
    notify();
}

/*----------------------------------------------- P2PConnectorAsyncReadContextChecker
 * -------------------------------------------------*/
P2PConnectorAsyncReadContextChecker::~P2PConnectorAsyncReadContextChecker() {
    stop();
}

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&               metrics_reporter,
                                               const std::shared_ptr<P2PBroadcastClient>&        tp_broadcast_client,
                                               const std::shared_ptr<autil::LockFreeThreadPool>& control_pool,
                                               const P2PConnectorSchedulerConfig&                config) {
    if (!control_pool || !config.topology) {
        RTP_LLM_LOG_ERROR("P2P lease checker requires a control pool and topology");
        return false;
    }
    group_block_bytes_.clear();
    total_kv_bytes_ = 0;
    for (const auto& group : config.topology->groups()) {
        const auto   physical    = config.physical_block_bytes_by_group.find(group.tag);
        const size_t block_bytes = physical != config.physical_block_bytes_by_group.end() ?
                                       physical->second :
                                       config.topology->blockSizeBytesForGroup(group.tag);
        group_block_bytes_.emplace(group.tag, block_bytes);
        total_kv_bytes_ += block_bytes * group.block_num;
    }
    if (total_kv_bytes_ == 0) {
        RTP_LLM_LOG_ERROR("P2P lease checker requires nonzero total KV capacity");
        return false;
    }
    metrics_reporter_    = metrics_reporter;
    tp_broadcast_client_ = tp_broadcast_client;
    control_pool_        = control_pool;
    control_capacity_    = control_pool_->getThreadNum();
    return true;
}

void P2PConnectorAsyncReadContextChecker::stop() {
    std::unordered_map<const P2PConnectorAsyncReadContext*, TrackedRead> remaining;
    {
        std::lock_guard<std::mutex> processing_lock(processing_mutex_);
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        stopping_ = true;
        remaining.swap(async_contexts_);
        if (cancel_retained_check_) {
            cancel_retained_check_();
            cancel_retained_check_ = {};
        }
        retained_contexts_.clear();
    }
    for (const auto& entry : remaining) {
        const auto& read = entry.second;
        if (read.cancel_wakeup) {
            read.cancel_wakeup();
        }
        reportDecodeScheduleMetrics(read.context);
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
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        if (stopping_ || !async_contexts_.try_emplace(context.get(), TrackedRead{context, {}}).second) {
            return;
        }
    }
    const std::weak_ptr<P2PConnectorAsyncReadContext> weak_context = context;
    context->setWakeupCallback([weak_checker = weak_from_this(), weak_context] {
        auto checker = weak_checker.lock();
        auto read    = weak_context.lock();
        if (checker && read) {
            checker->scheduleRead(read, P2PNotification::nowMs());
        }
    });
    scheduleRead(context, P2PNotification::nowMs());
}

size_t P2PConnectorAsyncReadContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    return async_contexts_.size();
}

void P2PConnectorAsyncReadContextChecker::scheduleRead(const std::shared_ptr<P2PConnectorAsyncReadContext>& context,
                                                       int64_t                                              wakeup_ms) {
    if (wakeup_ms == std::numeric_limits<int64_t>::max()) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    auto                        it = async_contexts_.find(context.get());
    if (it == async_contexts_.end() || it->second.wakeup_ms <= wakeup_ms) {
        return;
    }
    auto& read = it->second;
    if (read.cancel_wakeup) {
        read.cancel_wakeup();
    }
    read.wakeup_ms        = wakeup_ms;
    const auto generation = read.generation = ++next_generation_;
    read.cancel_wakeup                      = RpcCompletionQueue::instance().scheduleAfter(
        std::max<int64_t>(0, wakeup_ms - P2PNotification::nowMs()),
        [weak_checker = weak_from_this(), key = context.get(), generation](bool ok) {
            if (!ok) {
                return;
            }
            auto checker = weak_checker.lock();
            if (checker) {
                checker->advanceRead(key, generation);
            }
        });
}

void P2PConnectorAsyncReadContextChecker::checkRetainedResources(
    const std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>>& contexts) {
    // Count each physical (group, block) once, including shared prefix blocks.
    // Groups can have different block sizes and layer counts, so compare bytes.
    std::map<std::string, std::set<BlockIdxType>> held_blocks;
    size_t                                        held_contexts   = 0;
    size_t                                        awaiting_cancel = 0;
    size_t                                        pending_query   = 0;
    size_t                                        failed_query    = 0;
    size_t                                        awaiting_stop   = 0;
    int64_t                                       oldest_hold_ms  = 0;
    const int64_t                                 now_us          = P2PNotification::nowUs();
    for (const auto& context : contexts) {
        std::lock_guard<std::mutex> lock(context->state_mutex_);
        if (!context->resourceHoldPending() || !context->resource_) {
            continue;
        }
        ++held_contexts;
        if (!context->cancel_confirmed_.load(std::memory_order_acquire)) {
            ++awaiting_cancel;
        } else {
            switch (context->lease_query_state_.load(std::memory_order_relaxed)) {
                case P2PConnectorAsyncReadContext::LeaseQueryState::Pending:
                    ++pending_query;
                    break;
                case P2PConnectorAsyncReadContext::LeaseQueryState::Failed:
                    ++failed_query;
                    break;
                case P2PConnectorAsyncReadContext::LeaseQueryState::WaitingForStop:
                    ++awaiting_stop;
                    break;
            }
        }
        const auto hold_start_us = context->lease_hold_start_us_.load();
        if (hold_start_us > 0) {
            oldest_hold_ms = std::max(oldest_hold_ms, (now_us - hold_start_us) / 1000);
        }
        const auto& groups = context->resource_->groupBlockIds();
        RTP_LLM_CHECK_WITH_INFO(groups.size() == group_block_bytes_.size(), "P2P retained KV topology mismatch");
        for (const auto& [tag, block_bytes] : group_block_bytes_) {
            for (const auto block : groups.blocks(tag)) {
                if (!isNullBlockIdx(block) && block > 0) {
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
                            "oldest_hold_ms=%ld awaiting_cancel=%zu pending_query=%zu "
                            "failed_query=%zu awaiting_stop=%zu",
                            held_contexts,
                            held_bytes,
                            total_kv_bytes_,
                            held_percent,
                            oldest_hold_ms,
                            awaiting_cancel,
                            pending_query,
                            failed_query,
                            awaiting_stop);
        last_leak_log_ms_ = now_ms;
    } else if (held_contexts == 0 && last_leak_log_ms_ != 0) {
        RTP_LLM_LOG_INFO("P2P retained KV: all held resources released");
        last_leak_log_ms_ = 0;
    }
}

void P2PConnectorAsyncReadContextChecker::advanceRead(const P2PConnectorAsyncReadContext* key, uint64_t generation) {
    const auto                                    start_us = currentTimeUs();
    std::lock_guard<std::mutex>                   processing_lock(processing_mutex_);
    std::shared_ptr<P2PConnectorAsyncReadContext> context;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        auto                        it = async_contexts_.find(key);
        if (it == async_contexts_.end() || it->second.generation != generation) {
            return;
        }
        it->second.cancel_wakeup = {};
        it->second.wakeup_ms     = std::numeric_limits<int64_t>::max();
        context                  = it->second.context;
    }

    // The CQ advances the request's state. Any control operation that can
    // wait for worker RPCs runs on the existing control pool.
    try {
        context->checkDone();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("P2P read completion failed, key=%s error=%s", context->uniqueKey().c_str(), error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("P2P read completion failed, key=%s", context->uniqueKey().c_str());
    }
    try {
        context->expireTransferDeadlineIfNeeded();
        context->checkCancelDone();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("P2P read deadline or cancellation processing failed, key=%s error=%s",
                          context->uniqueKey().c_str(),
                          error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("P2P read deadline or cancellation processing failed, key=%s", context->uniqueKey().c_str());
    }

    int64_t retry_ms = std::numeric_limits<int64_t>::max();
    if (!context->control_active_.load() && context->needsControl()) {
        if (control_pool_ && control_inflight_->load() < control_capacity_) {
            const auto inflight = control_inflight_;
            inflight->fetch_add(1);
            context->control_active_.store(true);
            const auto ret = control_pool_->pushTask(
                [context, client = tp_broadcast_client_, inflight] {
                    context->runControl(client);
                    inflight->fetch_sub(1);
                    context->control_active_.store(false);
                    context->notify();
                },
                /*isBlocked=*/false,
                /*executeWhenFail=*/false);
            if (ret != autil::ThreadPoolBase::ERROR_NONE) {
                inflight->fetch_sub(1);
                context->control_active_.store(false);
            }
        }
        if (control_pool_ && !context->control_active_.load()) {
            retry_ms = P2PNotification::nowMs() + kLeasePollInitialIntervalMs;
        }
    }

    // Observe terminal publication before checking its associated lease.
    const bool done = context->done();
    const bool held = context->resourceHoldPending();
    if (held) {
        retained_contexts_.try_emplace(context.get(), context);
        scheduleRetainedCheck(0);
    } else if (retained_contexts_.erase(context.get()) != 0) {
        scheduleRetainedCheck(0);
    }

    if (done && !held && !context->control_active_.load()) {
        {
            std::lock_guard<std::mutex> lock(async_contexts_mutex_);
            auto                        it = async_contexts_.find(context.get());
            if (it->second.cancel_wakeup) {
                it->second.cancel_wakeup();
            }
            async_contexts_.erase(it);
        }
        reportDecodeScheduleMetrics(context);
        if (!context->success()) {
            RTP_LLM_LOG_WARNING("P2P Decode async read failed, unique_key=%s error=%s",
                                context->uniqueKey().c_str(),
                                context->errorInfo().ToString().c_str());
        }
    } else {
        const bool allow_control =
            control_pool_ && !context->control_active_.load() && retry_ms == std::numeric_limits<int64_t>::max();
        scheduleRead(context, std::min(retry_ms, context->nextWakeupMs(allow_control)));
    }

    if (metrics_reporter_) {
        P2PConnectorMetricsCollector collector;
        collector.decode_scheduler_check_once_cost_time_us = currentTimeUs() - start_us;
        collector.decode_scheduler_inflight_context_count  = inflightContextCount();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &collector);
    }
}

void P2PConnectorAsyncReadContextChecker::scheduleRetainedCheck(int64_t delay_ms) {
    if (cancel_retained_check_) {
        return;
    }
    cancel_retained_check_ =
        RpcCompletionQueue::instance().scheduleAfter(delay_ms, [weak_checker = weak_from_this()](bool ok) {
            if (auto checker = weak_checker.lock(); ok && checker) {
                checker->runRetainedCheck();
            }
        });
}

void P2PConnectorAsyncReadContextChecker::runRetainedCheck() {
    std::lock_guard<std::mutex> processing_lock(processing_mutex_);
    if (!cancel_retained_check_) {
        return;  // Stopped after the alarm was queued.
    }
    cancel_retained_check_ = {};
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> contexts;
    contexts.reserve(retained_contexts_.size());
    for (const auto& entry : retained_contexts_) {
        contexts.push_back(entry.second);
    }
    checkRetainedResources(contexts);
    if (!contexts.empty()) {
        scheduleRetainedCheck(1000);
    }
}

}  // namespace rtp_llm
