#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int device_id): device_id_(device_id) {}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    try {
        drainIfActive();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter destruction drain failed: %s", e.what());
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// IDLE -> RUNNING. Lazily starts the workers on first real use so constructing
// the model at an SCR checkpoint does not create idle cache-store threads.
void CacheStoreAsyncWriter::init(std::shared_ptr<void> publication_lease) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() called while already RUNNING. "
                            "Must call waitAllDone() before re-initializing.");

    if (!thread_pool_) {
        constexpr size_t kThreadCount = 3;
        constexpr size_t kQueueSize   = 10000;
        auto pool = std::make_shared<autil::LockFreeThreadPool>(kThreadCount, kQueueSize, nullptr, "CacheStoreAsync");
        RTP_LLM_CHECK_WITH_INFO(pool->start(), "CacheStoreAsyncWriter: failed to start thread pool");
        thread_pool_ = std::move(pool);
    }

    publications_ = std::make_shared<Publications>();
    publications_->lease = std::move(publication_lease);
    pending_count_.store(0, std::memory_order_relaxed);
    stored_exception_ = nullptr;
    state_            = State::RUNNING;
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
    RTP_LLM_PROFILE_SCOPE("cache_store_async_writer.waitAllDone");
    seal();

    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    bool publication_failed;
    {
        std::unique_lock<std::mutex> lock(publications_->mutex);
        publications_->cv.wait(lock, [this]() { return publications_->pending == 0; });
        publication_failed = publications_->failed;
        publications_->lease.reset();
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
    RTP_LLM_CHECK_WITH_INFO(!publication_failed, "cache store publication failed");
}

void CacheStoreAsyncWriter::seal() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ != State::IDLE, "cannot seal an idle cache store cycle");
    state_ = State::SEALED;
}

void CacheStoreAsyncWriter::drainIfActive() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_ == State::IDLE) {
            return;
        }
    }
    waitAllDone();
}

std::function<void(bool)> CacheStoreAsyncWriter::registerCompletion() {
    // Workers may register publications after sealing, but before their task
    // count reaches zero. Drain waits for both boundaries in that order.
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ != State::IDLE, "cannot register publication outside a write cycle");
    auto publications = publications_;
    auto completed = std::make_shared<std::atomic<bool>>(false);
    std::function<void(bool)> completion = [publications, completed](bool success) {
        if (completed->exchange(true)) {
            return;
        }
        std::lock_guard<std::mutex> lock(publications->mutex);
        publications->failed |= !success;
        --publications->pending;
        publications->cv.notify_all();
    };
    {
        std::lock_guard<std::mutex> lock(publications->mutex);
        ++publications->pending;
    }
    return completion;
}

}  // namespace rtp_llm
