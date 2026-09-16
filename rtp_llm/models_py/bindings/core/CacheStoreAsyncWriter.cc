#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"

namespace rtp_llm {

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int device_id): device_id_(device_id) {
    constexpr size_t kThreadCount = 3;
    constexpr size_t kQueueSize   = 10000;
    auto pool = std::make_shared<autil::LockFreeThreadPool>(kThreadCount, kQueueSize, nullptr, "CacheStoreAsync");
    RTP_LLM_CHECK_WITH_INFO(pool->start(), "CacheStoreAsyncWriter: failed to start thread pool");
    thread_pool_ = std::move(pool);
}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    if (state_ == State::RUNNING) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed while RUNNING — "
                            "caller should call waitAllDone() before destruction");
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// Drain any in-flight tasks and force the writer back to IDLE, discarding any stored exception.
// Non-throwing and idempotent (a no-op when already IDLE), so it is safe to call from a RAII
// guard during stack unwinding.
//
// This is the recovery hook for a cycle that never reached waitAllDone(). PyWrappedModel runs
// init() ... waitAllDone() with no scope guard on the async-prepare path, and that prepare runs
// on a SEPARATE thread (MtpExecutor target_verify_prepare_runner_) from the main-thread forward()
// that calls waitAllDone(). A forward that throws (the pybind11 forward, fusedCopy, an output-size
// RTP_LLM_CHECK, or a 32K CUDA-OOM), or an async prepare orphaned by a cancel before its forward,
// can therefore leave state_==RUNNING. The old RTP_LLM_CHECK(state==IDLE) in init() turned that
// recoverable state into an unbounded "already RUNNING" retry storm (the engine-loop catch-all
// swallowed and retried forever, flooding the log). Draining here makes the writer reusable so
// subsequent requests succeed.
void CacheStoreAsyncWriter::reset() noexcept {
    try {
        bool was_running;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            was_running = (state_ == State::RUNNING);
        }
        if (!was_running) {
            return;  // IDLE: nothing to drain.
        }
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter::reset() draining a cycle that never reached "
                            "waitAllDone() (a forward likely threw, or an async prepare was "
                            "orphaned); recovering to IDLE so the next init() does not wedge.");
        {
            std::unique_lock<std::mutex> lock(wait_mutex_);
            wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            pending_count_.store(0, std::memory_order_relaxed);
            state_ = State::IDLE;
        }
        {
            std::lock_guard<std::mutex> ex_lock(exception_mutex_);
            stored_exception_ = nullptr;
        }
    } catch (...) {
        // reset() may run during stack unwinding (a forward() RAII guard); never propagate.
        // Best effort: force IDLE so the writer is at least reusable.
        std::lock_guard<std::mutex> lock(state_mutex_);
        pending_count_.store(0, std::memory_order_relaxed);
        state_ = State::IDLE;
    }
}

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle. If a prior cycle was
// abandoned (still RUNNING), recover it first via reset() rather than asserting (see reset()).
// The happy path is unchanged: reset() is a no-op when init() is entered at IDLE.
void CacheStoreAsyncWriter::init() {
    reset();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        pending_count_.store(0, std::memory_order_relaxed);
        state_ = State::RUNNING;
    }
    {
        std::lock_guard<std::mutex> ex_lock(exception_mutex_);
        stored_exception_ = nullptr;
    }
}

// Enqueue a task to the background thread pool. Must be in RUNNING state.
void CacheStoreAsyncWriter::submit(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter::submit() called when not RUNNING. "
                            "Call init() first.");

    pending_count_.fetch_add(1, std::memory_order_acq_rel);

    auto wrapped = [this, task = std::move(task)]() {
        pinThreadToDeviceOnce(device_id_);
        try {
            task();
        } catch (...) {
            {
                std::lock_guard<std::mutex> ex_lock(exception_mutex_);
                if (!stored_exception_) {
                    stored_exception_ = std::current_exception();
                }
            }
            RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter: background task threw an exception");
        }
        if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(wait_mutex_);
            wait_cv_.notify_all();
        }
    };

    auto rc = thread_pool_->pushTask(std::move(wrapped));
    if (rc != autil::ThreadPoolBase::ERROR_NONE) {
        pending_count_.fetch_sub(1, std::memory_order_acq_rel);
        RTP_LLM_CHECK_WITH_INFO(false,
                                "CacheStoreAsyncWriter: pushTask failed (rc=%d). "
                                "Queue full or thread pool in bad state.",
                                static_cast<int>(rc));
    }
}

// Block until all submitted tasks complete, then RUNNING -> IDLE.
// Re-throws the first stored exception after state transition.
void CacheStoreAsyncWriter::waitAllDone() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::waitAllDone() called when not RUNNING. "
                                "Call init() first.");
    }

    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_ = State::IDLE;
    }

    if (stored_exception_) {
        auto ex           = stored_exception_;
        stored_exception_ = nullptr;
        std::rethrow_exception(ex);
    }
}

}  // namespace rtp_llm
