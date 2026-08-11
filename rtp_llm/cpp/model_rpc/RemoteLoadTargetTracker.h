#pragma once

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

class RemoteLoadTargetTracker {
public:
    explicit RemoteLoadTargetTracker(std::vector<std::string> targets):
        targets_(std::move(targets)), started_(targets_.size()) {
        for (auto& started : started_) {
            started.store(false, std::memory_order_relaxed);
        }
    }

    bool markStarted(size_t target_index) noexcept {
        if (target_index >= started_.size()) {
            return false;
        }
        started_[target_index].store(true, std::memory_order_release);
        return true;
    }

    std::vector<std::string> startedTargets() const {
        std::vector<std::string> result;
        result.reserve(targets_.size());
        for (size_t index = 0; index < targets_.size(); ++index) {
            if (started_[index].load(std::memory_order_acquire)) {
                result.push_back(targets_[index]);
            }
        }
        return result;
    }

private:
    const std::vector<std::string> targets_;
    std::vector<std::atomic<bool>> started_;
};

}  // namespace rtp_llm
