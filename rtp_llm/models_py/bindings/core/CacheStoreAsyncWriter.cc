#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <memory>
#include <utility>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheStoreAsyncWriter::PendingTaskGuard::PendingTaskGuard(CacheStoreAsyncWriter& writer): writer_(writer) {}

CacheStoreAsyncWriter::PendingTaskGuard::~PendingTaskGuard() {
    writer_.completePendingTask();
}

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int                             device_id,
                                             std::shared_ptr<KVCacheManager> cache_manager,
                                             size_t                          cache_model_id,
                                             std::optional<int>              mtp_cache_config_index,
                                             int                             forward_cp_rank,
                                             int                             forward_cp_size):
    device_id_(device_id),
    cache_manager_(std::move(cache_manager)),
    cache_model_id_(cache_model_id),
    cp_rank_(forward_cp_rank),
    cp_size_(forward_cp_size) {
    RTP_LLM_CHECK_WITH_INFO(cp_size_ > 0 && cp_rank_ >= 0 && cp_rank_ < cp_size_,
                            "CacheStoreAsyncWriter: invalid forward CP topology rank=%d size=%d",
                            cp_rank_,
                            cp_size_);
    if (cache_manager_) {
        const CacheConfig* selected_config = &cache_manager_->cacheConfig();
        if (mtp_cache_config_index.has_value()) {
            selected_config = &cache_manager_->getMTPModuleCacheConfig(*mtp_cache_config_index);
        }
        cache_config_ = std::shared_ptr<const CacheConfig>(cache_manager_, selected_config);
    }

    constexpr size_t kThreadCount = 3;
    constexpr size_t kQueueSize   = 10000;
    auto pool = std::make_shared<autil::LockFreeThreadPool>(kThreadCount, kQueueSize, nullptr, "CacheStoreAsync");
    RTP_LLM_CHECK_WITH_INFO(pool->start(), "CacheStoreAsyncWriter: failed to start thread pool");
    thread_pool_ = std::move(pool);
}

void CacheStoreAsyncWriter::completePendingTask() {
    if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

void CacheStoreAsyncWriter::storeCurrentException() {
    std::lock_guard<std::mutex> ex_lock(exception_mutex_);
    if (!stored_exception_) {
        stored_exception_ = std::current_exception();
    }
}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    if (state_ == State::RUNNING) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed while RUNNING - "
                            "caller should call waitAllDone() before destruction");
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle.
void CacheStoreAsyncWriter::init() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() called while already RUNNING. "
                            "Must call waitAllDone() before re-initializing.");
    pending_count_.store(0, std::memory_order_relaxed);
    stored_exception_   = nullptr;
    active_cache_store_ = cache_manager_ ? cache_manager_->getCacheStore() : nullptr;
    state_              = State::RUNNING;
}

// Enqueue a task to the background thread pool. Must be in RUNNING state.
void CacheStoreAsyncWriter::submit(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter::submit() called when not RUNNING. "
                            "Call init() first.");

    pending_count_.fetch_add(1, std::memory_order_acq_rel);
    auto wrapped = [this, task = std::move(task)]() {
        PendingTaskGuard pending_task_guard(*this);
        try {
            setCurrentThreadDeviceIfNeeded(device_id_);
            task();
        } catch (...) {
            storeCurrentException();
            RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter: background task threw an exception");
        }
    };

    const auto rc = thread_pool_->pushTask(std::move(wrapped));
    if (rc != autil::ThreadPoolBase::ERROR_NONE) {
        completePendingTask();
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
        active_cache_store_.reset();
        state_ = State::IDLE;
    }

    if (stored_exception_) {
        auto ex           = stored_exception_;
        stored_exception_ = nullptr;
        std::rethrow_exception(ex);
    }
}

void CacheStoreAsyncWriter::write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                                  const torch_ext::LayerKVCache&       layer_kv) {
    if (!active_cache_store_ || !cache_config_) {
        return;
    }

    // Capture tensors by value so their underlying storage stays alive in the background thread.
    // A torch::Tensor copy only increments the reference count.
    auto captured_cache_store_inputs = cache_store_inputs;
    auto captured_layer_kv           = layer_kv;
    auto cache_config                = cache_config_;
    auto cache_store                 = active_cache_store_;
    // Create the event on the main thread to avoid event-record contention in worker threads.
    auto event = runtimeCreateEvent();
    auto run   = [captured_cache_store_inputs,
                captured_layer_kv,
                cache_config,
                cache_store,
                cache_model_id = cache_model_id_,
                cp_rank        = cp_rank_,
                cp_size        = cp_size_,
                event          = std::move(event)]() mutable {
        runtimeWriteCacheStore(captured_cache_store_inputs,
                               captured_layer_kv,
                               *cache_config,
                               cache_store,
                               cache_model_id,
                               cp_rank,
                               cp_size,
                               std::move(event));
    };
    submit(std::move(run));
}

}  // namespace rtp_llm
