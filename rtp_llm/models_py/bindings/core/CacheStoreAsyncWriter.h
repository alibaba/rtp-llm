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
// Thread-safe: lifecycle methods may be called from different threads, but a
// cycle must remain ordered as either
// init -> submit* -> waitAllDone, or
// init -> submit* -> finishSubmissions -> waitStoreCompletions.
class CacheStoreAsyncWriter {
public:
    using StoreCompletionCallback = std::function<void(std::exception_ptr)>;

    explicit CacheStoreAsyncWriter(int device_id = -1);
    ~CacheStoreAsyncWriter();

    void init(bool track_store_completions = false);
    void submit(std::function<void()> task);

    // Register cache-store publication work started by a submitted task.  The
    // returned once-only callback must be invoked when CacheStore::store's
    // callback fires, not merely when CacheStore::store accepts the task.
    StoreCompletionCallback registerStoreCompletion();

    // Finish only the writer tasks that register CacheStore work.  This lets
    // DSpARK overlap target/draft computation with actual cache publication.
    // A matching waitStoreCompletions() is required before init() can begin
    // another cycle.
    void finishSubmissions();

    // Wait for the CacheStore callbacks registered by the finished cycle.
    // Waiting is intentionally unbounded: returning while a retained store
    // task still aliases request KV memory would permit that memory to be
    // released/reused before the late publication runs.
    void waitStoreCompletions();

    // Legacy convenience path: finish submissions and publication together.
    void waitAllDone();

private:
    enum class State {
        IDLE,
        RUNNING
    };

    struct StoreCompletionState {
        std::atomic<int64_t>    pending_count{0};
        std::mutex              wait_mutex;
        std::condition_variable cv;
        std::mutex              exception_mutex;
        std::exception_ptr      stored_exception;
    };

    struct StoreCompletionToken {
        explicit StoreCompletionToken(std::shared_ptr<StoreCompletionState> state): state(std::move(state)) {}

        std::shared_ptr<StoreCompletionState> state;
        std::atomic<bool>                      completed{false};
    };

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::mutex               state_mutex_;
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::mutex               exception_mutex_;
    std::exception_ptr       stored_exception_;
    std::shared_ptr<StoreCompletionState> active_store_completion_state_;
    std::shared_ptr<StoreCompletionState> finished_store_completion_state_;
    State                    state_{State::IDLE};
    int                      device_id_{-1};
};

}  // namespace rtp_llm
