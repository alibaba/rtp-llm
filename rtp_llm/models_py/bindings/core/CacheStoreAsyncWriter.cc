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

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle.
//
// SELF-HEAL a stale RUNNING state instead of asserting. The caller (PyWrappedModel) runs
// init() ... waitAllDone() with no scope guard between them, so a forward that throws in that
// region (the pybind11 python forward, fusedCopy, or an output-size RTP_LLM_CHECK) skips
// waitAllDone() and leaves state_==RUNNING. The old RTP_LLM_CHECK then made EVERY subsequent
// init() throw -> the engine-loop catch-all swallowed and retried -> an unbounded "already
// RUNNING" retry loop that floods the log with no recovery. Draining the abandoned cycle's
// in-flight tasks (the same wait_cv_/pending_count_ mechanism waitAllDone uses) and resetting
// makes the writer recoverable so subsequent requests succeed. The happy path is unchanged:
// init() is normally entered with state_==IDLE and pending_count_==0, so the drain wait returns
// immediately.
void CacheStoreAsyncWriter::init() {
    bool stale = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        stale = (state_ == State::RUNNING);
    }
    if (stale) {
        RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter::init() found stale RUNNING state (a prior cycle "
                          "did not reach waitAllDone(), likely an escaped forward exception); "
                          "draining in-flight tasks and resetting to avoid an 'already RUNNING' "
                          "retry storm.");
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }
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
