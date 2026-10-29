#pragma once

#include <cstdint>

namespace rtp_llm::cuda_trace {

// Observation only. No CUDA calls, tensor retention or stream/event dependencies
// are introduced by these business markers. start() is explicitly invoked once
// per backend process; a disabled recorder does not load CUPTI or start threads.
void start(int rank) noexcept;
bool active() noexcept;
// Remains true after a writer failure while this process still owns a CUPTI
// subscription. Profiler admission must use this, not recorder health.
bool     ownsCupti() noexcept;
uint64_t newId() noexcept;

struct Context {
    uint64_t scope_id   = 0;
    int64_t  request_id = -1;
    uint64_t step       = 0;
};

Context capture() noexcept;

class Scope {
public:
    Scope(const char* name,
          int64_t     request_id = -1,
          uint64_t    step       = 0,
          uint64_t    value0     = 0,
          uint64_t    value1     = 0) noexcept;
    ~Scope() noexcept;
    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Context  previous_{};
    uint64_t id_            = 0;
    uint64_t active_handle_ = 0;
};

class ContextGuard {
public:
    explicit ContextGuard(Context context) noexcept;
    ~ContextGuard() noexcept;
    ContextGuard(const ContextGuard&)            = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

private:
    Context previous_{};
    bool    installed_ = false;
};

void mark(const char* name, uint64_t value0 = 0, uint64_t value1 = 0, int64_t request_id = -1) noexcept;

}  // namespace rtp_llm::cuda_trace
