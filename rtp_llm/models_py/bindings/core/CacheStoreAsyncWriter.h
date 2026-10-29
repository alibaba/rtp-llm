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
// Thread-safe: init / submit / waitAllDone can be called from any thread.
// Lifecycle: init() -> submit()* -> waitAllDone() -> init() -> ...
class CacheStoreAsyncWriter {
public:
    explicit CacheStoreAsyncWriter(int device_id = -1);
    ~CacheStoreAsyncWriter();

    void init();
    void submit(std::function<void()> task);
    void waitAllDone();

    // Submit independent GPU post-processing before draining this step's writes.
    // An early return from submit still passes through the drain. On a submit
    // exception, drain before its buffers can be reused, preserving that error.
    template<typename F>
    auto completeAfter(F&& submit) -> decltype(submit()) {
        bool drain_started = false;
        try {
            auto result   = submit();
            drain_started = true;
            waitAllDone();
            return result;
        } catch (...) {
            if (!drain_started) {
                const auto original = std::current_exception();
                try {
                    waitAllDone();
                } catch (...) {
                    // submit's failure is primary; background task failures
                    // were logged by submit()'s worker wrapper already.
                }
                std::rethrow_exception(original);
            }
            throw;
        }
    }

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
