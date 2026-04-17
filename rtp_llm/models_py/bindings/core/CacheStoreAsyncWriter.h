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
    CacheStoreAsyncWriter();
    ~CacheStoreAsyncWriter();

    void init();
    void submit(std::function<void()> task);
    void waitAllDone();

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
};

}  // namespace rtp_llm
