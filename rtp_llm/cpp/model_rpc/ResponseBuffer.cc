#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

#include "autil/TimeUtility.h"

#include <utility>
#include <vector>

namespace rtp_llm {

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::createOrGet(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it != map_.end()) {
        return it->second;
    }
    auto entry              = std::make_shared<ResponseBufferEntry>();
    entry->last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
    map_.emplace(request_id, entry);
    return entry;
}

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::reserve(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it != map_.end()) {
        return nullptr;
    }
    auto entry              = std::make_shared<ResponseBufferEntry>();
    entry->last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
    map_.emplace(request_id, entry);
    return entry;
}

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::get(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = map_.find(request_id);
    if (it == map_.end()) {
        return nullptr;
    }
    return it->second;
}

void ResponseBufferRegistry::erase(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    map_.erase(request_id);
}

void ResponseBufferRegistry::cancelAll() {
    const int64_t                                     now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::vector<std::shared_ptr<ResponseBufferEntry>> entries;
    std::vector<std::function<void()>>                cancel_producers;

    {
        std::lock_guard<std::mutex> lock(mu_);
        entries.reserve(map_.size());
        for (const auto& kv : map_) {
            entries.push_back(kv.second);
        }
    }

    cancel_producers.reserve(entries.size());
    for (const auto& entry : entries) {
        std::function<void()> cancel_producer;
        {
            std::lock_guard<std::mutex> entry_lock(entry->mu);
            entry->cancelled.store(true);
            entry->last_activity_us = now_us;
            cancel_producer         = entry->cancel_producer;
            entry->cancel_producer  = nullptr;
        }
        if (cancel_producer) {
            cancel_producers.push_back(std::move(cancel_producer));
        }
        entry->cv.notify_all();
    }

    for (const auto& cancel_producer : cancel_producers) {
        cancel_producer();
    }
}

size_t ResponseBufferRegistry::gc(std::chrono::microseconds ttl) {
    const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    const int64_t ttl_us = ttl.count();
    size_t        swept  = 0;

    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = map_.begin(); it != map_.end();) {
        const auto& entry = it->second;
        bool        terminal = false;
        bool        idle     = false;
        {
            std::lock_guard<std::mutex> entry_lock(entry->mu);
            terminal = entry->done.load() || entry->cancelled.load() || entry->error_status.has_value();
            idle     = (now_us - entry->last_activity_us) >= ttl_us;
            if (terminal && idle) {
                std::deque<GenerateOutputsPB>().swap(entry->queue);
                entry->cancel_producer = nullptr;
            }
        }
        if (terminal && idle) {
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
    if (!entry_) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(entry_->mu);
        if (entry_->cancelled.load()) {
            return false;
        }
        entry_->queue.push_back(outputs);
        entry_->last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
    }
    entry_->cv.notify_all();
    return true;
}

}  // namespace rtp_llm
