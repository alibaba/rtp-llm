#pragma once

#include <condition_variable>
#include <mutex>
#include <stdexcept>

#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {
namespace detail {

// Only native background transfer scopes participate. Engine scopes can enclose
// start/stop, so enrolling every ProfilingScope would self-deadlock the owner.
struct BackgroundProfilingState {
    std::mutex              mutex;
    std::condition_variable changed;
    size_t                  active{0};
    bool                    transitioning{false};
};

inline BackgroundProfilingState& backgroundProfilingState() {
    static BackgroundProfilingState state;
    return state;
}

inline size_t& backgroundProfilingDepth() {
    static thread_local size_t depth{0};
    return depth;
}

class BackgroundProfilingLease {
public:
    BackgroundProfilingLease() {
        if (backgroundProfilingDepth()++ == 0) {
            auto&                        state = backgroundProfilingState();
            std::unique_lock<std::mutex> lock(state.mutex);
            state.changed.wait(lock, [&] { return !state.transitioning; });
            ++state.active;
        }
    }
    ~BackgroundProfilingLease() {
        if (--backgroundProfilingDepth() == 0) {
            auto&                       state = backgroundProfilingState();
            std::lock_guard<std::mutex> lock(state.mutex);
            --state.active;
            state.changed.notify_all();
        }
    }
    BackgroundProfilingLease(const BackgroundProfilingLease&)            = delete;
    BackgroundProfilingLease& operator=(const BackgroundProfilingLease&) = delete;
};

}  // namespace detail

// Close admission, wait for existing native RecordFunctions to end, then permit
// Kineto to install/remove/finalize its global callbacks. New transfer scopes
// resume after the transition, without retaining profiler TLS or stale contexts.
class BackgroundProfilingTransition {
public:
    BackgroundProfilingTransition() {
        if (detail::backgroundProfilingDepth() != 0) {
            throw std::logic_error("profiler transition inside a background transfer scope");
        }
        auto&                        state = detail::backgroundProfilingState();
        std::unique_lock<std::mutex> lock(state.mutex);
        state.changed.wait(lock, [&] { return !state.transitioning; });
        state.transitioning = true;
        state.changed.notify_all();
        state.changed.wait(lock, [&] { return state.active == 0; });
    }
    ~BackgroundProfilingTransition() {
        auto&                       state = detail::backgroundProfilingState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.transitioning = false;
        state.changed.notify_all();
    }
    BackgroundProfilingTransition(const BackgroundProfilingTransition&)            = delete;
    BackgroundProfilingTransition& operator=(const BackgroundProfilingTransition&) = delete;
};

class BackgroundProfilingScope {
public:
    explicit BackgroundProfilingScope(const char* name): scope_(name) {}

private:
    // Destruction order matters: RecordFunction ends before its lease releases.
    detail::BackgroundProfilingLease lease_;
    ProfilingScope                   scope_;
};

}  // namespace rtp_llm
