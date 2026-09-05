#pragma once

#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"

namespace rtp_llm {

class TransferBatchAsyncContext final: public AsyncContext {
public:
    explicit TransferBatchAsyncContext(std::shared_ptr<void> completion_guard = nullptr):
        completion_guard_(std::move(completion_guard)) {}

    void      waitDone() override;
    void      onDone(DoneCallback callback) override;
    bool      done() const override;
    bool      success() const override;
    ErrorInfo errorInfo() const override;

    void complete(ErrorInfo error);

private:
    mutable std::mutex      mutex_;
    std::condition_variable done_cv_;
    bool                    done_{false};
    ErrorInfo               error_{ErrorInfo::OkStatus()};
    std::shared_ptr<void>   completion_guard_;
    std::vector<DoneCallback> callbacks_;
};

}  // namespace rtp_llm
