#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace rtp_llm {

// A response can be queued before the executor finishes timing its step.
// Only the RPC consumer waits; completing or cancelling never blocks execution.
class PendingFrontendSpTpotSample {
public:
    ~PendingFrontendSpTpotSample() {
        if (!completed_) {
            promise_.set_value(std::nullopt);
        }
    }
    std::future<std::optional<double>> future() {
        return promise_.get_future();
    }
    void complete(double value) {
        promise_.set_value(value);
        completed_ = true;
    }

private:
    std::promise<std::optional<double>> promise_;
    bool                                completed_ = false;
};

// Shared by all queued output frames, including coalesced metric-only frames.
// Each engine sample is drained exactly once, regardless of output chunking.
class FrontendSpTpotSamples {
public:
    std::shared_ptr<PendingFrontendSpTpotSample> begin() {
        auto                        pending = std::make_shared<PendingFrontendSpTpotSample>();
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.emplace_back(++sequence_, pending->future());
        return pending;
    }
    std::vector<std::pair<uint64_t, double>> take() {
        std::vector<std::pair<uint64_t, std::future<std::optional<double>>>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending.swap(pending_);
        }
        std::vector<std::pair<uint64_t, double>> result;
        for (auto& item : pending) {
            auto value = item.second.get();
            if (value.has_value()) {
                result.emplace_back(item.first, *value);
            }
        }
        return result;
    }

private:
    std::mutex                                                           mutex_;
    uint64_t                                                             sequence_ = 0;
    std::vector<std::pair<uint64_t, std::future<std::optional<double>>>> pending_;
};
}  // namespace rtp_llm
