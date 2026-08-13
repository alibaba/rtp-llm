#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadLeaseRetainer.h"

namespace rtp_llm {

class MemoryCopyTaskGuard {
public:
    MemoryCopyTaskGuard(std::shared_ptr<MemoryAsyncContext>              context,
                        std::unique_ptr<RemoteLoadLeaseRetainer::Ticket> ticket);
    ~MemoryCopyTaskGuard();

    MemoryCopyTaskGuard(const MemoryCopyTaskGuard&)            = delete;
    MemoryCopyTaskGuard& operator=(const MemoryCopyTaskGuard&) = delete;

    bool enterBeforeDeadline(int64_t operation_deadline_unix_ms,
                             int64_t retention_timeout_ms,
                             int64_t safety_window_ms,
                             int64_t now_unix_ms);
    bool markStarted();
    bool finish(bool success);
    void abandon();
    void cancelBeforeDispatch() noexcept;

private:
    enum class State : uint8_t {
        PENDING,
        ENTERED,
        TERMINAL,
    };

    std::shared_ptr<MemoryAsyncContext>              context_;
    std::unique_ptr<RemoteLoadLeaseRetainer::Ticket> ticket_;
    std::atomic<State>                               state_{State::PENDING};
};

}  // namespace rtp_llm
