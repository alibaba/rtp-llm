#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyTaskGuard.h"

#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyDeadline.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

MemoryCopyTaskGuard::MemoryCopyTaskGuard(std::shared_ptr<MemoryAsyncContext>              context,
                                         std::unique_ptr<RemoteLoadLeaseRetainer::Ticket> ticket):
    context_(std::move(context)), ticket_(std::move(ticket)) {}

MemoryCopyTaskGuard::~MemoryCopyTaskGuard() {
    if (state_.load() == State::PENDING) {
        cancelBeforeDispatch();
    } else if (state_.load() == State::ENTERED) {
        abandon();
    }
}

bool MemoryCopyTaskGuard::enterBeforeDeadline(int64_t operation_deadline_unix_ms,
                                              int64_t retention_timeout_ms,
                                              int64_t safety_window_ms,
                                              int64_t now_unix_ms) {
    State expected = State::PENDING;
    if (!state_.compare_exchange_strong(expected, State::ENTERED)) {
        return false;
    }
    const auto admission = MemoryCopyDeadline::evaluateCopy(
        operation_deadline_unix_ms, retention_timeout_ms, safety_window_ms, now_unix_ms);
    if (!admission) {
        finish(false);
        return false;
    }
    return true;
}

bool MemoryCopyTaskGuard::markStarted() {
    if (ticket_ != nullptr && ticket_->markStarted()) {
        return true;
    }
    finish(false);
    return false;
}

bool MemoryCopyTaskGuard::finish(bool success) {
    State expected = State::ENTERED;
    if (!state_.compare_exchange_strong(expected, State::TERMINAL)) {
        return false;
    }
    const bool retired = ticket_ != nullptr && ticket_->complete();
    ticket_.reset();
    if (!retired) {
        RTP_LLM_LOG_ERROR("failed to retire a memory copy lease");
    }
    if (context_) {
        context_->complete(success && retired);
    }
    return retired;
}

void MemoryCopyTaskGuard::abandon() {
    State expected = State::ENTERED;
    if (!state_.compare_exchange_strong(expected, State::TERMINAL)) {
        return;
    }
    ticket_.reset();
    if (context_) {
        context_->complete(false);
    }
}

void MemoryCopyTaskGuard::cancelBeforeDispatch() noexcept {
    State expected = State::PENDING;
    if (!state_.compare_exchange_strong(expected, State::TERMINAL)) {
        return;
    }
    const bool retired = ticket_ != nullptr && ticket_->complete();
    ticket_.reset();
    if (!retired) {
        RTP_LLM_LOG_ERROR("failed to retire an undispatched memory copy lease");
    }
    if (context_) {
        context_->complete(false);
    }
}

}  // namespace rtp_llm
