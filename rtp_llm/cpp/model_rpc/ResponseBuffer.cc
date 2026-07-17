#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <utility>
#include <vector>

namespace rtp_llm {

size_t ResponseBufferEntry::kMaxQueueBytes = 512 * 1024 * 1024;  // 512MB per entry

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
        queue_size             = queue.size();
        const size_t msg_bytes = outputs.ByteSizeLong();
        if (queue_size >= kMaxQueueSize || queue_bytes_ + msg_bytes >= kMaxQueueBytes) {
            overflow = true;
        } else {
            queue.push_back(outputs);
            queue_bytes_ += msg_bytes;
            last_activity_us.store(autil::TimeUtility::currentTimeInMicroSeconds());
        }
    }
    if (overflow) {
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
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
    queue_bytes_ = 0;
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
        last_activity_us.store(autil::TimeUtility::currentTimeInMicroSeconds());
        cancel_producer = nullptr;
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
        last_activity_us.store(autil::TimeUtility::currentTimeInMicroSeconds());
        producer = std::move(cancel_producer);
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

size_t ResponseBufferEntry::droppedCount() const {
    return dropped_count_.load(std::memory_order_relaxed);
}

bool ResponseBufferEntry::discardIfProducerDoneAndIdle(int64_t now_us, int64_t ttl_us) {
    std::lock_guard<std::mutex> lock(mu);
    if (!done.load() || (now_us - last_activity_us.load()) < ttl_us) {
        return false;
    }
    std::deque<GenerateOutputsPB>().swap(queue);
    queue_bytes_    = 0;
    cancel_producer = nullptr;
    return true;
}

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::reserve(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it != map_.end()) {
        return nullptr;
    }
    auto entry = std::make_shared<ResponseBufferEntry>();
    entry->last_activity_us.store(autil::TimeUtility::currentTimeInMicroSeconds());
    map_.emplace(request_id, Record{entry, State::PENDING, false});
    return entry;
}

void ResponseBufferRegistry::publish(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it == map_.end()) {
        RTP_LLM_LOG_WARNING("publish: response entry not found for request_id=%ld, already GC'd", request_id);
        return;
    }
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
    if (it == map_.end()) {
        RTP_LLM_LOG_WARNING("finish: response entry not found for request_id=%ld, already GC'd", request_id);
        return;
    }
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
    if (it == map_.end()) {
        RTP_LLM_LOG_WARNING("releaseClaim: response entry not found for request_id=%ld, already GC'd", request_id);
        return;
    }
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
    if (it == map_.end()) {
        RTP_LLM_LOG_WARNING("abort: response entry not found for request_id=%ld, already GC'd", request_id);
        return;
    }
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
    const int64_t now_us         = autil::TimeUtility::currentTimeInMicroSeconds();
    const int64_t ttl_us         = ttl.count();
    const int64_t pending_ttl_us = 5 * 60 * 1000 * 1000LL;  // 5 minutes for orphaned PENDING entries
    size_t        swept          = 0;

    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = map_.begin(); it != map_.end();) {
        const auto&   record  = it->second;
        const auto&   entry   = record.entry;
        const int64_t idle_us = now_us - entry->last_activity_us.load();

        if (record.state == State::PENDING) {
            // Orphaned PENDING entry whose producer never published.
            // Cancel first (in case a producer started late), then erase.
            if (idle_us >= pending_ttl_us) {
                entry->cancel();
                it = map_.erase(it);
                ++swept;
            } else {
                ++it;
            }
            continue;
        }

        if (record.state == State::FETCH_CLAIMED) {
            // Fetch in progress but idle past TTL — cancel but don't erase.
            // The FetchResponse path will call releaseClaim() which performs the erase.
            if (idle_us >= ttl_us) {
                entry->cancel();
            }
            ++it;
            continue;
        }

        // State::READY
        if (entry->producerDone()) {
            // Producer finished — sweep if idle past TTL.
            if (entry->discardIfProducerDoneAndIdle(now_us, ttl_us)) {
                it = map_.erase(it);
                ++swept;
            } else {
                ++it;
            }
        } else {
            // READY but producer still running and idle past TTL — cancel but don't erase.
            // The producer's finish() path will erase when it completes.
            if (idle_us >= ttl_us) {
                entry->cancel();
            }
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
