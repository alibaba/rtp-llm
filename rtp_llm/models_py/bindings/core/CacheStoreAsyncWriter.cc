#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"

#include <chrono>

namespace rtp_llm {

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int device_id): device_id_(device_id) {
    constexpr size_t kThreadCount = 3;
    constexpr size_t kQueueSize   = 10000;
    auto pool = std::make_shared<autil::LockFreeThreadPool>(kThreadCount, kQueueSize, nullptr, "CacheStoreAsync");
    RTP_LLM_CHECK_WITH_INFO(pool->start(), "CacheStoreAsyncWriter: failed to start thread pool");
    thread_pool_ = std::move(pool);
}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    if (state_ == State::RUNNING || finished_store_completion_state_) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed with unfinished cache-store work — "
                            "caller should drain submissions and publication before destruction");
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle.
void CacheStoreAsyncWriter::init(bool track_store_completions) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() called while already RUNNING. "
                            "Must call waitAllDone() before re-initializing.");
    RTP_LLM_CHECK_WITH_INFO(finished_store_completion_state_ == nullptr,
                            "CacheStoreAsyncWriter::init() called before waitStoreCompletions() drained the previous "
                            "cycle");
    pending_count_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> ex_lock(exception_mutex_);
        stored_exception_ = nullptr;
    }
    active_store_completion_state_ =
        track_store_completions ? std::make_shared<StoreCompletionState>() : nullptr;
    state_ = State::RUNNING;
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

CacheStoreAsyncWriter::StoreCompletionCallback CacheStoreAsyncWriter::registerStoreCompletion() {
    std::shared_ptr<StoreCompletionState> completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::registerStoreCompletion() called when not RUNNING. "
                                "Call init() first.");
        completion_state = active_store_completion_state_;
    }
    RTP_LLM_CHECK_WITH_INFO(completion_state != nullptr,
                            "CacheStoreAsyncWriter: missing store completion state while RUNNING");
    completion_state->pending_count.fetch_add(1, std::memory_order_acq_rel);

    // CacheStore's callback contract is once-only. Keep the success path
    // lock-free; only failures contend on the exception mutex, and only the
    // final completion wakes the waiter.
    return [completion_state](std::exception_ptr exception) {
        if (exception) {
            std::lock_guard<std::mutex> lock(completion_state->exception_mutex);
            if (!completion_state->stored_exception) {
                completion_state->stored_exception = exception;
            }
        }
        const auto previous = completion_state->pending_count.fetch_sub(1, std::memory_order_acq_rel);
        if (previous <= 0) {
            RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter: store completion counter underflow");
            return;
        }
        if (previous == 1) {
            std::lock_guard<std::mutex> lock(completion_state->wait_mutex);
            completion_state->cv.notify_one();
        }
    };
}

void CacheStoreAsyncWriter::finishSubmissions() {
    std::shared_ptr<StoreCompletionState> completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::finishSubmissions() called when not RUNNING. "
                                "Call init() first.");
        completion_state = active_store_completion_state_;
    }
    RTP_LLM_CHECK_WITH_INFO(completion_state != nullptr,
                            "CacheStoreAsyncWriter::finishSubmissions() requires tracked store completions");

    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    std::exception_ptr worker_exception;
    {
        std::lock_guard<std::mutex> ex_lock(exception_mutex_);
        worker_exception  = stored_exception_;
        stored_exception_ = nullptr;
    }
    if (worker_exception && completion_state) {
        std::lock_guard<std::mutex> lock(completion_state->exception_mutex);
        if (!completion_state->stored_exception) {
            completion_state->stored_exception = worker_exception;
        }
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter changed state while finishing submissions");
        state_                           = State::IDLE;
        active_store_completion_state_.reset();
        finished_store_completion_state_ = std::move(completion_state);
    }
}

void CacheStoreAsyncWriter::waitStoreCompletions() {
    std::shared_ptr<StoreCompletionState> completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE && finished_store_completion_state_ != nullptr,
                                "CacheStoreAsyncWriter::waitStoreCompletions() called without a finished cycle");
        completion_state = finished_store_completion_state_;
    }

    std::exception_ptr store_exception;
    {
        std::unique_lock<std::mutex> lock(completion_state->wait_mutex);
        auto all_store_completions_done = [&completion_state]() {
            return completion_state->pending_count.load(std::memory_order_acquire) == 0;
        };
        while (!all_store_completions_done()) {
            if (completion_state->cv.wait_for(lock, std::chrono::seconds(10), all_store_completions_done)) {
                break;
            }
            RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter: still waiting for %ld cache-store publication callback(s)",
                                completion_state->pending_count.load(std::memory_order_acquire));
        }
    }
    {
        std::lock_guard<std::mutex> lock(completion_state->exception_mutex);
        store_exception = completion_state->stored_exception;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(finished_store_completion_state_ == completion_state,
                                "CacheStoreAsyncWriter completion cycle changed while waiting");
        finished_store_completion_state_.reset();
    }

    if (store_exception) {
        std::rethrow_exception(store_exception);
    }
}

void CacheStoreAsyncWriter::waitAllDone() {
    bool has_tracked_completions = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::waitAllDone() called when not RUNNING. Call init() first.");
        has_tracked_completions = active_store_completion_state_ != nullptr;
    }
    if (has_tracked_completions) {
        finishSubmissions();
        waitStoreCompletions();
        return;
    }

    // Preserve the legacy hot path exactly: no completion-state allocation or
    // callback bookkeeping for target/Eagle/MTP writes.
    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_ = State::IDLE;
    }
    std::exception_ptr worker_exception;
    {
        std::lock_guard<std::mutex> ex_lock(exception_mutex_);
        worker_exception  = stored_exception_;
        stored_exception_ = nullptr;
    }
    if (worker_exception) {
        std::rethrow_exception(worker_exception);
    }
}

}  // namespace rtp_llm
