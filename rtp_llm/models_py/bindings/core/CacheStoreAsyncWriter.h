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
    void trackExternalTask();
    void finishExternalTask(std::exception_ptr exception = nullptr);
    void waitAllDone();

private:
    enum class State {
        IDLE,
        RUNNING
    };

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::atomic<int64_t>     pending_external_count_{0};
    std::mutex               state_mutex_;
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::mutex               exception_mutex_;
    std::exception_ptr       stored_exception_;
    State                    state_{State::IDLE};
    // REBASE CONFLICT CONTEXT(6511f0467): new base pins worker threads to the
    // caller device via `device_id_`; source branch added external callback
    // tracking helpers. Keep both pieces.
    int device_id_{-1};

    void recordException(std::exception_ptr exception);
    void notifyIfDone();
};

}  // namespace rtp_llm
