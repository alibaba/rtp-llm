#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <utility>
#include <vector>

namespace rtp_llm {

void ResponseBufferEntry::installCancelProducer(std::function<void()> producer) {
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!cancelled.load()) {
            cancel_producer = std::move(producer);
            return;
        }
    }
    producer();
}

bool ResponseBufferEntry::write(const GenerateOutputsPB& outputs) {
    bool   overflow   = false;
    size_t queue_size = 0;
    {
        std::lock_guard<std::mutex> lock(mu);
        RTP_LLM_CHECK_WITH_INFO(
            !done.load(), "write response after producer finished for request_id=%ld", outputs.request_id());
        if (cancelled.load()) {
            return false;
        }
        queue_size = queue.size();
        if (queue_size >= kMaxQueueSize) {
            overflow = true;
        } else {
            queue.push_back(outputs);
            last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
        }
    }
    if (overflow) {
        RTP_LLM_LOG_ERROR("response queue overflow for request_id=%ld, queue_size=%zu, cancelling request",
                          outputs.request_id(),
                          queue_size);
        cancel();
        return false;
    }
    cv.notify_all();
    return true;
}

ResponseBufferEntry::DrainResult ResponseBufferEntry::waitAndDrain(std::chrono::milliseconds timeout) {
    DrainResult                  result;
    std::unique_lock<std::mutex> lock(mu);
    cv.wait_for(lock, timeout, [this] {
        return !queue.empty() || done.load() || cancelled.load() || error_status.has_value();
    });
    result.outputs.swap(queue);
    if (cancelled.load()) {
        result.terminal        = true;
        result.terminal_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    } else if (error_status.has_value()) {
        result.terminal        = true;
        result.terminal_status = *error_status;
    } else if (done.load()) {
        result.terminal = true;
    }
    return result;
}

void ResponseBufferEntry::finish(const grpc::Status& status) {
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!status.ok()) {
            error_status = status;
        }
        done.store(true);
        last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
        cancel_producer  = nullptr;
    }
    cv.notify_all();
}

void ResponseBufferEntry::cancel() {
    std::function<void()> producer;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (done.load()) {
            return;
        }
        cancelled.store(true);
        last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
        producer         = std::move(cancel_producer);
    }
    if (producer) {
        producer();
    }
    cv.notify_all();
}

bool ResponseBufferEntry::producerDone() const {
    return done.load();
}

bool ResponseBufferEntry::isCancelled() const {
    return cancelled.load();
}

bool ResponseBufferEntry::discardIfProducerDoneAndIdle(int64_t now_us, int64_t ttl_us) {
    std::lock_guard<std::mutex> lock(mu);
    if (!done.load() || (now_us - last_activity_us) < ttl_us) {
        return false;
    }
    std::deque<GenerateOutputsPB>().swap(queue);
    cancel_producer = nullptr;
    return true;
}

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::reserve(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it != map_.end()) {
        return nullptr;
    }
    auto entry              = std::make_shared<ResponseBufferEntry>();
    entry->last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
    map_.emplace(request_id, Record{entry, State::PENDING, false});
    return entry;
}

void ResponseBufferRegistry::publish(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    RTP_LLM_CHECK_WITH_INFO(it != map_.end(), "publish missing response entry for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(
        it->second.entry == expected_entry, "publish response entry identity mismatch for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(it->second.state == State::PENDING,
                            "invalid response registry transition: publish request_id=%ld state=%d expected=PENDING",
                            request_id,
                            static_cast<int>(it->second.state));
    it->second.state = State::READY;
}

ResponseBufferRegistry::ClaimResult ResponseBufferRegistry::claim(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it == map_.end() || it->second.state == State::PENDING) {
        return {ClaimStatus::NOT_FOUND, nullptr};
    }
    if (it->second.state == State::FETCH_CLAIMED) {
        return {ClaimStatus::ALREADY_CLAIMED, nullptr};
    }
    it->second.state = State::FETCH_CLAIMED;
    return {ClaimStatus::SUCCESS, it->second.entry};
}

void ResponseBufferRegistry::finish(int64_t                                     request_id,
                                    const std::shared_ptr<ResponseBufferEntry>& expected_entry,
                                    const grpc::Status&                         status) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    RTP_LLM_CHECK_WITH_INFO(it != map_.end(), "finish missing response entry for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(
        it->second.entry == expected_entry, "finish response entry identity mismatch for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(
        !expected_entry->producerDone(), "duplicate producer finish for request_id=%ld", request_id);
    expected_entry->finish(status);
    if (it->second.state == State::FETCH_CLAIMED && it->second.fetch_released) {
        map_.erase(it);
    }
}

void ResponseBufferRegistry::releaseClaim(int64_t                                     request_id,
                                          const std::shared_ptr<ResponseBufferEntry>& expected_entry) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    RTP_LLM_CHECK_WITH_INFO(it != map_.end(), "release missing response entry for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(
        it->second.entry == expected_entry, "release response entry identity mismatch for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(it->second.state == State::FETCH_CLAIMED,
                            "invalid response registry transition: release request_id=%ld state=%d "
                            "expected=FETCH_CLAIMED",
                            request_id,
                            static_cast<int>(it->second.state));
    RTP_LLM_CHECK_WITH_INFO(!it->second.fetch_released, "duplicate fetch release for request_id=%ld", request_id);
    if (expected_entry->producerDone()) {
        map_.erase(it);
    } else {
        it->second.fetch_released = true;
    }
}

void ResponseBufferRegistry::abort(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    RTP_LLM_CHECK_WITH_INFO(it != map_.end(), "abort missing response entry for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(
        it->second.entry == expected_entry, "abort response entry identity mismatch for request_id=%ld", request_id);
    RTP_LLM_CHECK_WITH_INFO(it->second.state == State::PENDING,
                            "invalid response registry transition: abort request_id=%ld state=%d expected=PENDING",
                            request_id,
                            static_cast<int>(it->second.state));
    map_.erase(it);
}

void ResponseBufferRegistry::cancelAll() {
    std::vector<std::shared_ptr<ResponseBufferEntry>> entries;

    {
        std::lock_guard<std::mutex> lock(mu_);
        entries.reserve(map_.size());
        for (const auto& kv : map_) {
            entries.push_back(kv.second.entry);
        }
    }

    for (const auto& entry : entries) {
        entry->cancel();
    }
}

size_t ResponseBufferRegistry::gc(std::chrono::microseconds ttl) {
    const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    const int64_t ttl_us = ttl.count();
    size_t        swept  = 0;

    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = map_.begin(); it != map_.end();) {
        const auto& record = it->second;
        if (record.state != State::READY) {
            ++it;
            continue;
        }
        if (record.entry->discardIfProducerDoneAndIdle(now_us, ttl_us)) {
            it = map_.erase(it);
            ++swept;
        } else {
            ++it;
        }
    }
    return swept;
}

size_t ResponseBufferRegistry::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return map_.size();
}

bool ResponseBufferWriter::Write(const GenerateOutputsPB& outputs, grpc::WriteOptions /*options*/) {
    return entry_ && entry_->write(outputs);
}

}  // namespace rtp_llm
