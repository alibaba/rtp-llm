#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

namespace rtp_llm {

// 用于 memory connector match
class MemoryAsyncMatchContext: public AsyncMatchContext {
public:
    explicit MemoryAsyncMatchContext(size_t matched_block_count): matched_block_count_(matched_block_count) {}
    ~MemoryAsyncMatchContext() override = default;

public:
    void   waitDone() override;
    bool   done() const override;
    bool   success() const override;
    size_t matchedBlockCount() const override;

private:
    size_t matched_block_count_{0};
};

// 用于 memory connector read/write
class MemoryAsyncContext: public AsyncContext {
public:
    explicit MemoryAsyncContext(const std::function<void(bool)>& done_callback): done_callback_(done_callback) {}
    ~MemoryAsyncContext() override = default;

public:
    void waitDone() override;
    bool done() const override;
    bool success() const override;
    void complete(bool success);

private:
    std::function<void(bool)>                                               done_callback_;
    std::atomic<bool>                                                       already_done_{false};
    std::atomic<bool>                                                       completion_success_{false};
    mutable std::mutex                                                      completion_mutex_;
    std::condition_variable                                                 completion_cv_;
    bool                                                                    completion_started_{false};
};

}  // namespace rtp_llm
