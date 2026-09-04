#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "autil/EnvUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr int64_t kDefaultStoreCompletionTimeoutMs = 30000;

std::chrono::milliseconds
resolveStoreCompletionTimeout(std::optional<std::chrono::milliseconds> configured_timeout) {
    const auto timeout_ms = configured_timeout.has_value() ? configured_timeout->count() :
                                                             autil::EnvUtil::getEnv(
                                                                 "RTP_LLM_CACHE_STORE_PUBLICATION_TIMEOUT_MS",
                                                                 kDefaultStoreCompletionTimeoutMs);
    RTP_LLM_CHECK_WITH_INFO(timeout_ms > 0,
                            "CacheStoreAsyncWriter publication timeout must be positive, got %ld ms",
                            timeout_ms);
    return std::chrono::milliseconds(timeout_ms);
}

}  // namespace

CacheStoreAsyncWriter::PendingTaskGuard::PendingTaskGuard(CacheStoreAsyncWriter& writer): writer_(writer) {}

CacheStoreAsyncWriter::PendingTaskGuard::~PendingTaskGuard() {
    writer_.completePendingTask();
}

CacheStoreAsyncWriter::CacheStoreAsyncWriter(
    int                                      device_id,
    std::shared_ptr<KVCacheManager>          cache_manager,
    size_t                                   cache_model_id,
    std::optional<int>                       mtp_cache_config_index,
    std::optional<std::chrono::milliseconds> store_completion_timeout):
    device_id_(device_id),
    store_completion_timeout_(resolveStoreCompletionTimeout(store_completion_timeout)),
    cache_manager_(std::move(cache_manager)),
    cache_model_id_(cache_model_id) {
    if (cache_manager_) {
        const CacheConfig* selected_config = &cache_manager_->cacheConfig();
        if (mtp_cache_config_index.has_value()) {
            selected_config = &cache_manager_->getMTPModuleCacheConfig(*mtp_cache_config_index);
        }
        cache_config_ = std::shared_ptr<const CacheConfig>(cache_manager_, selected_config);

        if (const auto cp_slot_mapper = cache_manager_->cpSlotMapper()) {
            cp_rank_ = cp_slot_mapper->cpRank();
            cp_size_ = cp_slot_mapper->cpSize();
        }
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

void CacheStoreAsyncWriter::terminateStoreCompletions(
    const std::shared_ptr<StoreCompletionState>& completion_state, std::exception_ptr exception) {
    if (!completion_state) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(completion_state->mutex);
        if (completion_state->terminal) {
            return;
        }
        completion_state->terminal      = true;
        completion_state->pending_count = 0;
        if (!completion_state->stored_exception) {
            completion_state->stored_exception = std::move(exception);
        }
    }
    completion_state->cv.notify_all();
}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    bool                                  unfinished = false;
    std::shared_ptr<StoreCompletionState> active_completion_state;
    std::shared_ptr<StoreCompletionState> finished_completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        unfinished                = state_ == State::RUNNING || finished_store_completion_state_ != nullptr;
        active_completion_state   = active_store_completion_state_;
        finished_completion_state = finished_store_completion_state_;
    }
    if (unfinished) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed with unfinished cache-store work - "
                            "cancelling outstanding publication callbacks");
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
    auto shutdown_exception =
        std::make_exception_ptr(std::runtime_error("cache-store publication cancelled during writer shutdown"));
    terminateStoreCompletions(active_completion_state, shutdown_exception);
    terminateStoreCompletions(finished_completion_state, shutdown_exception);
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

CacheStoreAsyncWriter::StoreCompletionCallback CacheStoreAsyncWriter::registerStoreCompletion() {
    std::shared_ptr<StoreCompletionState> completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::registerStoreCompletion() called when not RUNNING. "
                                "Call init() first.");
        completion_state = active_store_completion_state_;
    }
    return registerStoreCompletionOn(completion_state);
}

CacheStoreAsyncWriter::StoreCompletionCallback
CacheStoreAsyncWriter::registerStoreCompletionOn(const std::shared_ptr<StoreCompletionState>& completion_state) {
    RTP_LLM_CHECK_WITH_INFO(completion_state != nullptr,
                            "CacheStoreAsyncWriter: missing store completion state while RUNNING");

    auto completion_token = std::make_shared<StoreCompletionToken>(completion_state);
    StoreCompletionCallback completion = [completion_token](std::exception_ptr exception) {
        if (completion_token->completed.exchange(true, std::memory_order_acq_rel)) {
            RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter: duplicate store completion ignored");
            return;
        }
        const auto& completion_state = completion_token->state;
        bool        notify_waiter    = false;
        {
            std::lock_guard<std::mutex> lock(completion_state->mutex);
            if (completion_state->terminal) {
                RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter: late store completion ignored after cycle termination");
                return;
            }
            if (exception && !completion_state->stored_exception) {
                completion_state->stored_exception = std::move(exception);
            }
            RTP_LLM_CHECK_WITH_INFO(completion_state->pending_count > 0,
                                    "CacheStoreAsyncWriter: store completion counter underflow");
            --completion_state->pending_count;
            notify_waiter = completion_state->pending_count == 0;
        }
        if (notify_waiter) {
            completion_state->cv.notify_one();
        }
    };

    {
        std::lock_guard<std::mutex> lock(completion_state->mutex);
        if (!completion_state->terminal) {
            ++completion_state->pending_count;
        }
    }
    return completion;
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
    if (worker_exception) {
        terminateStoreCompletions(completion_state, worker_exception);
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter changed state while finishing submissions");
        active_cache_store_.reset();
        state_                            = State::IDLE;
        active_store_completion_state_.reset();
        finished_store_completion_state_ = std::move(completion_state);
    }
}

void CacheStoreAsyncWriter::waitStoreCompletions() {
    std::shared_ptr<StoreCompletionState> completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_ == State::IDLE && finished_store_completion_state_ == nullptr) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE && finished_store_completion_state_ != nullptr,
                                "CacheStoreAsyncWriter::waitStoreCompletions() called before submissions finished");
        completion_state = finished_store_completion_state_;
    }

    std::exception_ptr store_exception;
    {
        std::unique_lock<std::mutex> lock(completion_state->mutex);
        const auto completed = completion_state->cv.wait_for(lock, store_completion_timeout_, [&completion_state]() {
            return completion_state->pending_count == 0 || completion_state->terminal;
        });
        if (!completed) {
            const auto pending_count         = completion_state->pending_count;
            completion_state->terminal      = true;
            completion_state->pending_count = 0;
            if (!completion_state->stored_exception) {
                completion_state->stored_exception = std::make_exception_ptr(std::runtime_error(
                    "timed out after " + std::to_string(store_completion_timeout_.count()) + " ms waiting for "
                    + std::to_string(pending_count) + " cache-store publication callback(s)"));
            }
            RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter: timed out after %ld ms waiting for %ld publication callback(s)",
                              store_completion_timeout_.count(),
                              pending_count);
        }
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

void CacheStoreAsyncWriter::cancelStoreCompletions(std::exception_ptr exception) {
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(exception),
                            "CacheStoreAsyncWriter::cancelStoreCompletions() requires a failure exception");
    std::shared_ptr<StoreCompletionState> active_completion_state;
    std::shared_ptr<StoreCompletionState> finished_completion_state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        active_completion_state   = active_store_completion_state_;
        finished_completion_state = finished_store_completion_state_;
    }
    terminateStoreCompletions(active_completion_state, exception);
    terminateStoreCompletions(finished_completion_state, std::move(exception));
}

// Block until all submitted tasks and any tracked publication complete, then
// return to IDLE. Re-throws the first stored exception after draining.
void CacheStoreAsyncWriter::waitAllDone() {
    bool has_tracked_completions = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::waitAllDone() called when not RUNNING. "
                                "Call init() first.");
        has_tracked_completions = active_store_completion_state_ != nullptr;
    }
    if (has_tracked_completions) {
        finishSubmissions();
        waitStoreCompletions();
        return;
    }

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
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        active_cache_store_.reset();
        state_ = State::IDLE;
    }
    if (worker_exception) {
        std::rethrow_exception(worker_exception);
    }
}

void CacheStoreAsyncWriter::write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                                  const torch_ext::LayerKVCache&       layer_kv) {
    if (!active_cache_store_ || !cache_config_) {
        // Fail closed when publication is tracked: the executor waits on this
        // cycle and reduces the result across TP before dispatching decode. A
        // silent skip here would leave zero pending callbacks, so the wait would
        // report success while the KV transfer never happened.
        bool tracked = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            tracked = active_store_completion_state_ != nullptr;
        }
        RTP_LLM_CHECK_WITH_INFO(!tracked,
                                "CacheStoreAsyncWriter::write() has tracked publication but no cache store/config; "
                                "refusing to silently drop the KV transfer");
        return;
    }

    // Capture tensors by value so their underlying storage stays alive in the background thread.
    // A torch::Tensor copy only increments the reference count.
    auto captured_cache_store_inputs = cache_store_inputs;
    auto captured_layer_kv           = layer_kv;
    auto cache_config                = cache_config_;
    auto cache_store                 = active_cache_store_;
    CacheStoreCompletionRegistrar register_store_completion;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::write() called when not RUNNING. Call init() first.");
        if (auto completion_state = active_store_completion_state_) {
            register_store_completion = [completion_state]() {
                return registerStoreCompletionOn(completion_state);
            };
        }
    }
    // Create the event on the main thread to avoid event-record contention in worker threads.
    auto event = runtimeCreateEvent();
    auto run   = [captured_cache_store_inputs,
                captured_layer_kv,
                cache_config,
                cache_store,
                register_store_completion,
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
                               std::move(event),
                               register_store_completion);
    };
    submit(std::move(run));
}

}  // namespace rtp_llm
