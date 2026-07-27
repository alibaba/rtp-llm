#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <mutex>

#include "rtp_llm/cpp/cache/AsyncContext.h"

namespace rtp_llm {

class LoadBackAsyncContext: public AsyncContext {
public:
    enum class State : int {
        PENDING          = 0,
        CANCEL_REQUESTED = 1,
        SUCCEEDED        = 2,
        FAILED           = 3,
        CANCELLED        = 4
    };

    explicit LoadBackAsyncContext(size_t pending_transfer_count);
    ~LoadBackAsyncContext() override = default;

    bool requestCancel();
    bool isRequestCanceled() const;
    bool completeOne(bool success);
    bool onTaskFail();
    void waitDone() override;
    bool done() const override;
    bool success() const override;

private:
    std::atomic<State>      state_{State::PENDING};
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    size_t                  remaining_transfer_count_;
    bool                    has_failure_{false};
};

}  // namespace rtp_llm
