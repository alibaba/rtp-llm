#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "autil/TimeUtility.h"

namespace rtp_llm {

// ---- BoundedRelay ----

BoundedRelay::BoundedRelay(size_t cap): cap_(cap) {}

bool BoundedRelay::push(GenerateOutputsPB output) {
    std::unique_lock<std::mutex> lock(mu_);
    // 1s 轮询而非无限阻塞：close() 的 notify_all 可能在 wait_for 返回后、
    // 下一轮条件检查前到达，最坏情况 cancel 后多等 1s——可接受的延迟换取
    // 不依赖 reaper 及时到达的鲁棒性
    while (queue_.size() >= cap_ && !closed_) {
        cv_.wait_for(lock, std::chrono::seconds(1));
    }
    if (closed_) {
        return false;
    }
    queue_.push_back(std::move(output));
    lock.unlock();
    cv_.notify_all();
    return true;
}

bool BoundedRelay::tryPop(GenerateOutputsPB* out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) {
        return false;
    }
    *out = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return true;
}

size_t BoundedRelay::drainTo(std::vector<GenerateOutputsPB>* out) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t count = queue_.size();
    out->reserve(out->size() + count);
    for (auto& item : queue_) {
        out->push_back(std::move(item));
    }
    queue_.clear();
    if (count > 0) {
        cv_.notify_all();
    }
    return count;
}

bool BoundedRelay::waitForData(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || closed_; });
}

void BoundedRelay::close() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
    }
    cv_.notify_all();
}

bool BoundedRelay::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.empty();
}

size_t BoundedRelay::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

bool BoundedRelay::isClosed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
}

// ---- RequestSession ----

RequestSession::RequestSession(int64_t request_id, int64_t batch_id, int64_t admitted_at_us, bool is_pd):
    request_id_(request_id), batch_id_(batch_id), is_pd_(is_pd),
    relay_(is_pd ? 1000 : 0), admitted_at_us_(admitted_at_us) {}

void RequestSession::cancel(CancelReason reason) {
    std::lock_guard<std::mutex> lock(mu_);
    auto current = state_.load();
    if (current == SessionState::FINISHED || current == SessionState::CANCELLED || current == SessionState::ERROR) {
        return;
    }
    cancel_reason_.store(reason);
    state_.store(SessionState::CANCELLED);
    finished_at_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());

    if (stream_) {
        stream_->reportError(ErrorCode::CANCELLED, "session cancelled");
    }
    if (cancel_state_) {
        // store 在 cancel_state_->mu 之外是安全的：refreshAsyncProducerCancelState
        // 在释放 mu 后才读 cancelled，cancel() 在获取 mu 前已设标志，
        // 保证 TryCancel 至少被调用一次（见时间线分析 A/B/C）
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
    relay_.close();
}

bool RequestSession::acquireLease() {
    bool expected = false;
    return consumer_taken_.compare_exchange_strong(expected, true);
}

void RequestSession::releaseLease() {
    consumer_taken_.store(false);
}

SessionState RequestSession::deriveState() {
    auto current = state_.load();
    if (current == SessionState::FINISHED || current == SessionState::CANCELLED || current == SessionState::ERROR) {
        return current;
    }
    if (!stream_) {
        return current;
    }
    auto stream_status = stream_->getStatus();
    if (stream_status == StreamState::RUNNING) {
        state_.store(SessionState::RUNNING);
        return SessionState::RUNNING;
    }
    if (stream_status == StreamState::FINISHED) {
        if (trySetTerminal(SessionState::FINISHED)) {
            return SessionState::FINISHED;
        }
        return state_.load();
    }
    return current;
}

bool RequestSession::isTerminal() const {
    auto s = state_.load();
    return s == SessionState::FINISHED || s == SessionState::CANCELLED || s == SessionState::ERROR;
}

std::shared_ptr<GenerateStream> RequestSession::getStream() const {
    return stream_;
}

void RequestSession::setStream(std::shared_ptr<GenerateStream> stream) {
    stream_ = std::move(stream);
}

void RequestSession::setCancelState(std::shared_ptr<AsyncProducerCancelState> cancel_state) {
    cancel_state_ = std::move(cancel_state);
}

void RequestSession::markFinished() {
    trySetTerminal(SessionState::FINISHED);
    relay_.close();
}

void RequestSession::markError(const std::string& error_msg) {
    trySetTerminal(SessionState::ERROR);
    relay_.close();
}

bool RequestSession::trySetTerminal(SessionState target) {
    std::lock_guard<std::mutex> lock(mu_);
    auto current = state_.load();
    if (current == SessionState::FINISHED || current == SessionState::CANCELLED || current == SessionState::ERROR) {
        return false;
    }
    state_.store(target);
    finished_at_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());
    return true;
}

}  // namespace rtp_llm
