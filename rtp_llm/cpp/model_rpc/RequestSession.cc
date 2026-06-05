#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// ========================== RequestOutputBuffer ==========================

RequestOutputBuffer::RequestOutputBuffer(size_t max_bytes): max_bytes_(max_bytes) {}

PushResult RequestOutputBuffer::push(GenerateOutputsPB&& output) {
    size_t output_bytes = output.ByteSizeLong();
    std::unique_lock<std::mutex> lock(mu_);

    if (closed_) {
        return PushResult::CLOSED;
    }
    if (max_bytes_ > 0 && bytes_ + output_bytes > max_bytes_) {
        return PushResult::BUDGET_EXCEEDED;
    }

    bytes_ += output_bytes;
    outputs_.push_back(std::move(output));
    lock.unlock();
    cv_.notify_all();
    return PushResult::OK;
}

PopResult RequestOutputBuffer::popLive(uint64_t lease_id, int64_t wait_timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);

    if (next_live_seq_ >= outputs_.size() && !closed_) {
        auto timeout = std::chrono::milliseconds(wait_timeout_ms);
        cv_.wait_for(lock, timeout, [this] {
            return next_live_seq_ < outputs_.size() || closed_;
        });
    }

    if (next_live_seq_ >= outputs_.size()) {
        if (closed_) {
            return PopResult{PopResult::CLOSED, {}};
        }
        return PopResult{PopResult::WAIT_TIMEOUT, {}};
    }

    std::vector<GenerateOutputsPB> batch;
    while (next_live_seq_ < outputs_.size()) {
        batch.push_back(outputs_[next_live_seq_]);
        ++next_live_seq_;
    }
    return PopResult{PopResult::OUTPUT, std::move(batch)};
}

std::shared_ptr<const FrozenSnapshot> RequestOutputBuffer::freeze(int64_t request_id,
                                                                   int64_t session_epoch,
                                                                   TerminalInfo terminal) {
    std::lock_guard<std::mutex> lock(mu_);
    if (snapshot_) {
        return snapshot_;
    }
    auto snap = std::make_shared<FrozenSnapshot>();
    snap->request_id = request_id;
    snap->session_epoch = session_epoch;
    snap->terminal = std::move(terminal);
    snap->outputs = outputs_;
    snapshot_ = snap;
    return snapshot_;
}

std::shared_ptr<const FrozenSnapshot> RequestOutputBuffer::snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return snapshot_;
}

void RequestOutputBuffer::close() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
    }
    cv_.notify_all();
}

void RequestOutputBuffer::dropSnapshot() {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_.reset();
    outputs_.clear();
    outputs_.shrink_to_fit();
    bytes_ = 0;
}

size_t RequestOutputBuffer::bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bytes_;
}

size_t RequestOutputBuffer::pendingLiveCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return outputs_.size() - next_live_seq_;
}

bool RequestOutputBuffer::isClosed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
}

// ========================== SessionWriter ==========================

bool SessionWriter::Write(const GenerateOutputsPB& outputs, grpc::WriteOptions /*options*/) {
    return session_->pushOutput(GenerateOutputsPB(outputs)) == PushResult::OK;
}

// ========================== RequestSession ==========================

RequestSession::RequestSession(const SessionCreateOptions& options, int64_t session_epoch):
    output_buffer_(options.max_buffer_bytes),
    request_id_(options.request_id),
    session_epoch_(session_epoch),
    payload_hash_(options.payload_hash),
    batch_id_(options.batch_id),
    batch_index_(options.batch_index),
    create_time_us_(options.create_time_us),
    attach_deadline_us_(options.attach_deadline_us),
    payload_ttl_us_(options.payload_ttl_us),
    tombstone_ttl_us_(options.tombstone_ttl_us) {}

bool RequestSession::isTerminal() const {
    auto p = phase_.load();
    return p == Phase::FINALIZING || p == Phase::TERMINAL;
}

bool RequestSession::hasConsumer() const {
    std::lock_guard<std::mutex> lock(lease_mu_);
    return lease_active_;
}

bool RequestSession::bindStream(std::shared_ptr<GenerateStream> stream) {
    std::lock_guard<std::mutex> lock(resource_mu_);
    if (isTerminal()) {
        return false;
    }
    stream_ = std::move(stream);
    return true;
}

void RequestSession::setCancelState(std::shared_ptr<AsyncProducerCancelState> cancel_state) {
    std::lock_guard<std::mutex> lock(resource_mu_);
    cancel_state_ = std::move(cancel_state);
}

std::shared_ptr<GenerateStream> RequestSession::getStream() const {
    std::lock_guard<std::mutex> lock(resource_mu_);
    return stream_;
}

PushResult RequestSession::pushOutput(GenerateOutputsPB&& output) {
    if (isTerminal()) {
        return PushResult::CLOSED;
    }
    return output_buffer_.push(std::move(output));
}

LeaseResult RequestSession::acquireLiveLease() {
    std::lock_guard<std::mutex> lock(lease_mu_);
    if (lease_active_) {
        return LeaseResult{AttachState::ALREADY_ATTACHED, 0};
    }
    lease_active_ = true;
    lease_id_ = next_lease_id_++;
    return LeaseResult{AttachState::LIVE, lease_id_};
}

PopResult RequestSession::popLive(uint64_t lease_id, int64_t wait_timeout_ms) {
    {
        std::lock_guard<std::mutex> lock(lease_mu_);
        if (!lease_active_ || lease_id_ != lease_id) {
            return PopResult{PopResult::CLOSED, {}};
        }
    }
    return output_buffer_.popLive(lease_id, wait_timeout_ms);
}

void RequestSession::releaseLiveLease(uint64_t lease_id) {
    std::lock_guard<std::mutex> lock(lease_mu_);
    if (lease_active_ && lease_id_ == lease_id) {
        lease_active_ = false;
    }
}

bool RequestSession::cancel(CancelReason reason, int64_t now_us) {
    TerminalReason tr;
    switch (reason) {
        case CancelReason::ATTACH_DEADLINE:
        case CancelReason::EXECUTION_TIMEOUT:
        case CancelReason::SLO_DEADLINE:
            tr = TerminalReason::TIMEOUT;
            break;
        default:
            tr = TerminalReason::CANCELLED;
            break;
    }
    return finalizeTerminal(tr, grpc::Status(grpc::StatusCode::CANCELLED, "cancelled"), now_us);
}

bool RequestSession::finalizeTerminal(TerminalReason reason, grpc::Status status, int64_t now_us) {
    Phase expected = Phase::LIVE;
    if (!phase_.compare_exchange_strong(expected, Phase::FINALIZING)) {
        return false;
    }

    TerminalInfo info;
    info.reason = reason;
    info.status = std::move(status);
    info.terminal_time_us = now_us;
    info.payload_expire_time_us = now_us + payload_ttl_us_;
    info.tombstone_expire_time_us = now_us + payload_ttl_us_ + tombstone_ttl_us_;

    output_buffer_.freeze(request_id_, session_epoch_, info);
    output_buffer_.close();

    {
        std::lock_guard<std::mutex> lock(terminal_mu_);
        terminal_ = info;
    }

    {
        std::lock_guard<std::mutex> lock(resource_mu_);
        if (stream_) {
            if (reason == TerminalReason::CANCELLED || reason == TerminalReason::TIMEOUT) {
                stream_->reportError(ErrorCode::CANCELLED, "session terminated");
            }
        }
        if (cancel_state_) {
            cancel_state_->cancelled.store(true);
            std::shared_ptr<grpc::ClientContext> client_ctx;
            {
                std::lock_guard<std::mutex> state_lock(cancel_state_->mu);
                client_ctx = cancel_state_->client_context.lock();
            }
            if (client_ctx) {
                client_ctx->TryCancel();
            }
        }
    }

    phase_.store(Phase::TERMINAL);
    return true;
}

LookupResult RequestSession::buildLookup(int64_t now_us) const {
    auto p = phase_.load();
    if (p == Phase::LIVE) {
        return LookupResult{AttachState::LIVE,
                            nullptr, nullptr, std::nullopt};
    }
    if (p == Phase::FINALIZING) {
        return LookupResult{AttachState::LIVE,
                            nullptr, nullptr, std::nullopt};
    }
    // TERMINAL
    std::lock_guard<std::mutex> lock(terminal_mu_);
    if (now_us < terminal_.payload_expire_time_us) {
        return LookupResult{AttachState::FINISHED_IN_TTL,
                            nullptr,
                            output_buffer_.snapshot(),
                            terminal_};
    }
    return LookupResult{AttachState::GONE,
                        nullptr, nullptr, terminal_};
}

bool RequestSession::payloadExpired(int64_t now_us) const {
    if (!isTerminal()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(terminal_mu_);
    return now_us >= terminal_.payload_expire_time_us;
}

TerminalInfo RequestSession::terminalInfo() const {
    std::lock_guard<std::mutex> lock(terminal_mu_);
    return terminal_;
}

std::shared_ptr<const FrozenSnapshot> RequestSession::snapshot() const {
    return output_buffer_.snapshot();
}

}  // namespace rtp_llm
