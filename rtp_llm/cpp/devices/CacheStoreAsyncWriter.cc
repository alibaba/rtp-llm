#include "rtp_llm/cpp/devices/CacheStoreAsyncWriter.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

CacheStoreAsyncWriter::CacheStoreAsyncWriter() {
    constexpr size_t kThreadCount = 30;
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

// IDLE -> RUNNING.  Resets bookkeeping for a new forward-pass cycle.
void CacheStoreAsyncWriter::init() {
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() called while already RUNNING. "
                            "Must call waitAllDone() before re-initializing.");

    pending_count_.store(0, std::memory_order_relaxed);
    stored_exception_ = nullptr;
    state_            = State::RUNNING;
}

void CacheStoreAsyncWriter::submit(std::function<void()> task) {
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter::submit() called when not RUNNING. "
                            "Call init() first.");

    pending_count_.fetch_add(1, std::memory_order_acq_rel);

    auto wrapped = [this, task = std::move(task)]() {
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
// Re-throws the first stored exception after state transition so that
// the writer is always left in a clean IDLE state regardless of errors.
void CacheStoreAsyncWriter::waitAllDone() {
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter::waitAllDone() called when not RUNNING. "
                            "Call init() first.");

    // pending_count_ starts at 0 in init().  If no tasks were submitted this
    // cycle the condition is immediately true and we return without blocking.
    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    // Transition to IDLE *before* rethrowing so the writer is always in a
    // clean state after waitAllDone() returns (even on exception).
    state_ = State::IDLE;

    if (stored_exception_) {
        auto ex           = stored_exception_;
        stored_exception_ = nullptr;
        std::rethrow_exception(ex);
    }
}

}  // namespace rtp_llm
