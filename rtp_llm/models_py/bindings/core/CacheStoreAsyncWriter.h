#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

#include "autil/ThreadPool.h"

namespace rtp_llm {

// Offloads writeCacheStore CPU-heavy work to a background thread pool so the
// main thread can keep launching CUDA kernels without stalling.
// Thread-safe: init / submit / waitAllDone / reset can be called from any thread.
// Lifecycle: init() -> submit()* -> waitAllDone() -> init() -> ...
// reset() is the recovery hook for a cycle abandoned before waitAllDone() (see .cc); init()
// calls it internally, so an abandoned RUNNING cycle self-heals instead of wedging.
class CacheStoreAsyncWriter {
public:
    explicit CacheStoreAsyncWriter(int device_id = -1);
    ~CacheStoreAsyncWriter();

    void init();
    void submit(std::function<void()> task);
    void waitAllDone();
    // Drain in-flight tasks and force IDLE, discarding any stored exception. Non-throwing and
    // idempotent (no-op when already IDLE), so it is safe to call from a RAII guard during stack
    // unwinding. Recovers a cycle that never reached waitAllDone().
    void reset() noexcept;

private:
    enum class State {
        IDLE,
        RUNNING
    };

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::mutex               state_mutex_;
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::mutex               exception_mutex_;
    std::exception_ptr       stored_exception_;
    State                    state_{State::IDLE};
    int                      device_id_{-1};
};

}  // namespace rtp_llm
