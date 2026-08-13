#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rtp_llm {

class MemoryCopyFence {
private:
    struct Entry;
    struct State;

public:
    class OperationGuard {
    public:
        ~OperationGuard();

        OperationGuard(const OperationGuard&)            = delete;
        OperationGuard& operator=(const OperationGuard&) = delete;

    private:
        friend class MemoryCopyFence;
        OperationGuard(std::shared_ptr<State> state, std::shared_ptr<Entry> entry);

    private:
        std::shared_ptr<State> state_;
        std::shared_ptr<Entry> entry_;
    };

    using Operation = std::shared_ptr<OperationGuard>;

    struct BeginResult {
        Operation   operation;
        std::string error;

        explicit operator bool() const {
            return operation != nullptr;
        }
    };

    MemoryCopyFence();
    ~MemoryCopyFence() = default;

    BeginResult begin(const std::string& operation_id, std::chrono::milliseconds retention);
    BeginResult beginBeforeDeadline(const std::string&         operation_id,
                                    std::chrono::milliseconds retention,
                                    int64_t                   operation_deadline_unix_ms);

    bool sealAndWait(const std::string&         operation_id,
                     std::chrono::milliseconds wait_timeout,
                     std::chrono::milliseconds retention);

    bool stopAndWait(std::chrono::milliseconds wait_timeout);

    size_t entryCountForTest() const;
    size_t pruneCandidateChecksForTest() const;
    void   pruneExpiredAtForTest(std::chrono::steady_clock::time_point now);
    void   withStateLockForTest(const std::function<void()>& callback);

private:
    BeginResult beginLocked(const std::string&                    operation_id,
                            std::chrono::milliseconds             retention,
                            std::chrono::steady_clock::time_point now);

private:
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
