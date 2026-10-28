#pragma once
#include <mutex>
#include <string>
#include <unordered_map>

namespace rtp_llm {
class TransferTaskRegistry {
public:
    enum class State { NOT_FOUND, RUNNING, FINISHED };

    bool start(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return !task_id.empty() && tasks_.emplace(task_id, State::RUNNING).second;
    }

    // Call only after all I/O has stopped accessing the task's blocks.
    bool finish(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(task_id);
        if (it == tasks_.end()) {
            return false;
        }
        it->second = State::FINISHED;
        return true;
    }

    // Idempotent: a lost cleanup reply can be retried, but running work is retained.
    bool forget(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(task_id);
        if (it == tasks_.end()) return true;
        if (it->second != State::FINISHED) return false;
        tasks_.erase(it);
        return true;
    }

    State query(const std::string& task_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tasks_.find(task_id);
        return it == tasks_.end() ? State::NOT_FOUND : it->second;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, State> tasks_;
};
}  // namespace rtp_llm
