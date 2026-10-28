#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace rtp_llm {
class KVCMOperationRegistry {
public:
    enum class State { NOT_FOUND, RUNNING, FINISHED };

    bool start(const std::string& operation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return !operation_id.empty() && operations_.emplace(operation_id, State::RUNNING).second;
    }

    bool finish(const std::string& operation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = operations_.find(operation_id);
        if (it == operations_.end()) return false;
        it->second = State::FINISHED;
        return true;
    }

    State query(const std::string& operation_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = operations_.find(operation_id);
        return it == operations_.end() ? State::NOT_FOUND : it->second;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, State> operations_;
};
}  // namespace rtp_llm
