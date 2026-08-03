#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {

// 用于 memory connector match
class MemoryAsyncMatchContext: public AsyncMatchContext {
public:
    explicit MemoryAsyncMatchContext(size_t matched_token_count,
                                     size_t planned_start_token = 0,
                                     size_t planned_token_count = 0):
        matched_token_count_(matched_token_count),
        planned_start_token_(planned_start_token),
        planned_token_count_(planned_token_count) {}
    ~MemoryAsyncMatchContext() override = default;

public:
    void   waitDone() override;
    bool   done() const override;
    bool   success() const override;
    size_t matchedTokenCount() const override;
    size_t plannedStartToken() const;
    size_t plannedTokenCount() const;

private:
    size_t matched_token_count_{0};
    size_t planned_start_token_{0};
    size_t planned_token_count_{0};
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
    void setBroadcastResult(const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result);
    void markFailed(const std::string& reason);

private:
    bool successLocked() const;

private:
    mutable std::mutex                                                      mutex_;
    std::condition_variable                                                 cv_;
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> broadcast_result_;
    std::function<void(bool)>                                               done_callback_;
    bool                                                                    result_ready_{false};
    bool                                                                    finalizing_{false};
    bool                                                                    failed_{false};
    std::string                                                             failure_reason_;
    std::atomic<bool>                                                       already_done_{false};
};

}  // namespace rtp_llm
