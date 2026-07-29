#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <memory>
#include <utility>

#include "autil/EnvUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheStoreAsyncWriter::PendingTaskGuard::PendingTaskGuard(CacheStoreAsyncWriter& writer): writer_(writer) {}

CacheStoreAsyncWriter::PendingTaskGuard::~PendingTaskGuard() {
    writer_.completePendingTask();
}

const char* CacheStoreAsyncWriter::stateName(State state) {
    switch (state) {
        case State::IDLE:
            return "IDLE";
        case State::RUNNING:
            return "RUNNING";
        case State::DRAINING:
            return "DRAINING";
    }
    return "UNKNOWN";
}

std::shared_ptr<const CacheConfig>
CacheStoreAsyncWriter::selectCacheConfig(const std::shared_ptr<KVCacheManager>& cache_manager,
                                         const std::optional<int>&              mtp_cache_config_index) {
    if (!cache_manager) {
        return nullptr;
    }
    const CacheConfig* selected_config = &cache_manager->cacheConfig();
    if (mtp_cache_config_index.has_value()) {
        selected_config = &cache_manager->getMTPModuleCacheConfig(*mtp_cache_config_index);
    }
    return std::shared_ptr<const CacheConfig>(cache_manager, selected_config);
}

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int                             device_id,
                                             std::shared_ptr<KVCacheManager> cache_manager,
                                             size_t                          cache_model_id,
                                             std::optional<int>              mtp_cache_config_index):
    device_id_(device_id),
    cache_manager_(std::move(cache_manager)),
    cache_config_(selectCacheConfig(cache_manager_, mtp_cache_config_index)),
    cache_model_id_(cache_model_id) {
    if (cache_manager_) {
        if (const auto cp_slot_mapper = cache_manager_->cpSlotMapper()) {
            cp_rank_ = cp_slot_mapper->cpRank();
            cp_size_ = cp_slot_mapper->cpSize();
        }
    }

    // Env-tunable with safe defaults (same operational style as
    // CACHE_STORE_SKIP_WRITE_WHEN_UNREADY): queue-full aborts the forward, so
    // deployments with unusual layer/tag/batch products can widen the pool
    // without a rebuild. Read once per writer at construction.
    const size_t thread_count = autil::EnvUtil::getEnv("CACHE_STORE_WRITER_THREAD_NUM", static_cast<size_t>(3));
    const size_t queue_size   = autil::EnvUtil::getEnv("CACHE_STORE_WRITER_QUEUE_SIZE", static_cast<size_t>(10000));
    RTP_LLM_CHECK_WITH_INFO(thread_count > 0 && queue_size > 0,
                            "CacheStoreAsyncWriter: CACHE_STORE_WRITER_THREAD_NUM (%zu) and "
                            "CACHE_STORE_WRITER_QUEUE_SIZE (%zu) must both be positive",
                            thread_count,
                            queue_size);
    auto pool = std::make_shared<autil::LockFreeThreadPool>(thread_count, queue_size, nullptr, "CacheStoreAsync");
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
    State state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = state_;
    }
    if (state != State::IDLE) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed while %s - "
                            "caller should call waitAllDone() before destruction",
                            stateName(state));
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle.
// On failure the writer stays IDLE and init() can be called again once the
// CacheStore is injected (RemoteRpcServer::initCacheStore() ->
// KVCacheManager::setCacheStore()); no partially initialized state survives.
void CacheStoreAsyncWriter::init() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() requires IDLE state, got %s. "
                            "Must call waitAllDone() before re-initializing.",
                            stateName(state_));
    RTP_LLM_CHECK_WITH_INFO(cache_manager_ != nullptr,
                            "CacheStoreAsyncWriter::init() cannot start cache-store work because KVCacheManager is "
                            "unavailable (model_id=%zu, device_id=%d)",
                            cache_model_id_,
                            device_id_);
    RTP_LLM_CHECK_WITH_INFO(cache_config_ != nullptr,
                            "CacheStoreAsyncWriter::init() cannot start cache-store work because CacheConfig is "
                            "unavailable (model_id=%zu, device_id=%d)",
                            cache_model_id_,
                            device_id_);
    auto cache_store = cache_manager_->getCacheStore();
    // Operational rollback for the fail-fast-on-missing-CacheStore behavior:
    // CACHE_STORE_SKIP_WRITE_WHEN_UNREADY=1 restores the legacy WriteCacheStoreOp
    // semantics (WARN and silently skip this cycle's writes) for deployments whose
    // startup order cannot guarantee CacheStore injection before PD prefill traffic.
    // Read per cycle so flipping the env takes effect without code changes.
    skip_cycle_writes_ = false;
    if (cache_store == nullptr && autil::EnvUtil::getEnv("CACHE_STORE_SKIP_WRITE_WHEN_UNREADY", false)) {
        RTP_LLM_INTERVAL_LOG(60,
                             WARN,
                             "CacheStoreAsyncWriter: CacheStore not injected yet; skipping cache-store writes for "
                             "this forward cycle because CACHE_STORE_SKIP_WRITE_WHEN_UNREADY is set "
                             "(model_id=%zu, device_id=%d)",
                             cache_model_id_,
                             device_id_);
        skip_cycle_writes_ = true;
    } else {
        RTP_LLM_CHECK_WITH_INFO(cache_store != nullptr,
                                "CacheStoreAsyncWriter::init() cannot start cache-store work because CacheStore is "
                                "unavailable (model_id=%zu, device_id=%d). Ensure RemoteRpcServer::initCacheStore() "
                                "has injected the CacheStore before PD prefill, or set "
                                "CACHE_STORE_SKIP_WRITE_WHEN_UNREADY=1 to temporarily restore the legacy "
                                "skip-write behavior.",
                                cache_model_id_,
                                device_id_);
    }

    pending_count_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> exception_lock(exception_mutex_);
        stored_exception_ = nullptr;
    }
    active_cache_store_ = std::move(cache_store);
    ++cycle_id_;
    state_ = State::RUNNING;
}

// Enqueue a task to the background thread pool. Must be in RUNNING state.
void CacheStoreAsyncWriter::submit(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    enqueueLocked(std::move(task));
}

// state_mutex_ must be held by the caller. Admission and pending accounting are
// intentionally one critical section with the RUNNING-state check.
void CacheStoreAsyncWriter::enqueueLocked(std::function<void()> task) {
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter task submission requires RUNNING state, got %s. "
                            "Call init() first and do not submit after waitAllDone() starts.",
                            stateName(state_));

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

// Stop accepting new tasks, then block until all admitted tasks complete.
// Re-throws the first stored exception after state transition.
void CacheStoreAsyncWriter::waitAllDone() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::waitAllDone() requires RUNNING state, got %s. "
                                "Call init() first.",
                                stateName(state_));
        state_ = State::DRAINING;
    }

    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    std::exception_ptr ex;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::DRAINING,
                                "CacheStoreAsyncWriter::waitAllDone() expected DRAINING state, got %s",
                                stateName(state_));
        std::lock_guard<std::mutex> exception_lock(exception_mutex_);
        ex                = stored_exception_;
        stored_exception_ = nullptr;
        active_cache_store_.reset();
        state_ = State::IDLE;
    }
    if (ex) {
        std::rethrow_exception(ex);
    }
}

void CacheStoreAsyncWriter::write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                                  const torch_ext::LayerKVCache&       layer_kv) {
    {
        // Legacy-skip cycle (see init()): the rollback switch admitted this cycle
        // without a CacheStore, so drop the write silently. Non-RUNNING states fall
        // through to the regular error paths below.
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_ == State::RUNNING && skip_cycle_writes_) {
            return;
        }
    }
#if !USING_CUDA && !USING_ROCM
    (void)cache_store_inputs;
    (void)layer_kv;
    RTP_LLM_CHECK_WITH_INFO(false, "CacheStoreAsyncWriter::write() requires a CUDA or ROCm build");
#else
    // Capture tensors by value so their underlying storage stays alive in the background thread.
    // A torch::Tensor copy only increments the reference count.
    auto captured_cache_store_inputs = cache_store_inputs;
    auto captured_layer_kv           = layer_kv;

    std::shared_ptr<const CacheConfig> cache_config;
    std::shared_ptr<CacheStore>        cache_store;
    uint64_t                           cycle_id;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::write() requires an active RUNNING forward cycle, got %s "
                                "for tag=%s layer=%d",
                                stateName(state_),
                                layer_kv.tag.c_str(),
                                layer_kv.layer_id);
        RTP_LLM_CHECK_WITH_INFO(active_cache_store_ != nullptr && cache_config_ != nullptr,
                                "CacheStoreAsyncWriter::write() has no active CacheStore/CacheConfig snapshot "
                                "for tag=%s layer=%d (model_id=%zu, device_id=%d)",
                                layer_kv.tag.c_str(),
                                layer_kv.layer_id,
                                cache_model_id_,
                                device_id_);
        cache_config = cache_config_;
        cache_store  = active_cache_store_;
        cycle_id     = cycle_id_;
    }

    // Create the event on the caller thread without extending the writer state lock
    // across a device runtime call.
    auto event = runtimeCreateEvent();

    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING && cycle_id_ == cycle_id,
                            "CacheStoreAsyncWriter forward cycle changed while creating an event "
                            "for tag=%s layer=%d",
                            layer_kv.tag.c_str(),
                            layer_kv.layer_id);
    auto run = [captured_cache_store_inputs,
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
    enqueueLocked(std::move(run));
#endif
}

}  // namespace rtp_llm
