#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

#include "autil/TimeUtility.h"

namespace rtp_llm {

std::shared_ptr<ResponseBufferEntry> ResponseBufferRegistry::create(int64_t request_id) {
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

size_t ResponseBufferRegistry::gc(std::chrono::microseconds ttl) {
    const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    const int64_t ttl_us = ttl.count();
    size_t        swept  = 0;

    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = map_.begin(); it != map_.end();) {
        const auto& entry    = it->second;
        bool        terminal = entry->done.load() || entry->cancelled.load() || entry->error_status.has_value();
        bool        idle     = (now_us - entry->last_activity_us) >= ttl_us;
        bool        drained  = false;
        {
            std::lock_guard<std::mutex> elock(entry->mu);
            drained = entry->queue.empty();
        }
        if (terminal && drained && idle) {
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
    if (entry_->cancelled.load()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(entry_->mu);
        entry_->queue.push_back(outputs);
        entry_->last_activity_us = autil::TimeUtility::currentTimeInMicroSeconds();
    }
    entry_->cv.notify_all();
    return true;
}

}  // namespace rtp_llm
