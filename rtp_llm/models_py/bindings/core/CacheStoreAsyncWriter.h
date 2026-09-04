#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "autil/ThreadPool.h"
#include "rtp_llm/models_py/bindings/CacheStoreWriter.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

class CacheConfig;
class CacheStore;
class KVCacheManager;

// Offloads writeCacheStore CPU-heavy work to a background thread pool so the
// main thread can keep launching CUDA kernels without stalling. A cycle is
// ordered as either init -> write* -> waitAllDone, or init -> write* ->
// finishSubmissions -> waitStoreCompletions.
class CacheStoreAsyncWriter: public CacheStoreWriter {
public:
    using StoreCompletionCallback = CacheStoreCompletionCallback;

    explicit CacheStoreAsyncWriter(
        int                                      device_id              = -1,
        std::shared_ptr<KVCacheManager>          cache_manager          = nullptr,
        size_t                                   cache_model_id         = 0,
        std::optional<int>                       mtp_cache_config_index = std::nullopt,
        std::optional<std::chrono::milliseconds> store_completion_timeout = std::nullopt);
    ~CacheStoreAsyncWriter() override;

    void init(bool track_store_completions = false);
    StoreCompletionCallback registerStoreCompletion();
    void                    finishSubmissions();
    void                    waitStoreCompletions();
    void                    cancelStoreCompletions(std::exception_ptr exception);
    void                    waitAllDone();
    void write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
               const torch_ext::LayerKVCache&       layer_kv) override;

private:
    void submit(std::function<void()> task);

    class PendingTaskGuard {
    public:
        explicit PendingTaskGuard(CacheStoreAsyncWriter& writer);
        ~PendingTaskGuard();

        PendingTaskGuard(const PendingTaskGuard&)            = delete;
        PendingTaskGuard& operator=(const PendingTaskGuard&) = delete;

    private:
        CacheStoreAsyncWriter& writer_;
    };

    void completePendingTask();
    void storeCurrentException();

    enum class State {
        IDLE,
        RUNNING
    };

    struct StoreCompletionState {
        int64_t                 pending_count{0};
        bool                    terminal{false};
        std::mutex              mutex;
        std::condition_variable cv;
        std::exception_ptr      stored_exception;
    };

    struct StoreCompletionToken {
        explicit StoreCompletionToken(std::shared_ptr<StoreCompletionState> state): state(std::move(state)) {}

        std::shared_ptr<StoreCompletionState> state;
        std::atomic<bool>                     completed{false};
    };

    static void terminateStoreCompletions(const std::shared_ptr<StoreCompletionState>& completion_state,
                                          std::exception_ptr                            exception);

    static StoreCompletionCallback
    registerStoreCompletionOn(const std::shared_ptr<StoreCompletionState>& completion_state);

    autil::ThreadPoolBasePtr                    thread_pool_;
    std::atomic<int64_t>                        pending_count_{0};
    std::mutex                                  state_mutex_;
    std::mutex                                  wait_mutex_;
    std::condition_variable                     wait_cv_;
    std::mutex                                  exception_mutex_;
    std::exception_ptr                          stored_exception_;
    std::shared_ptr<StoreCompletionState>        active_store_completion_state_;
    std::shared_ptr<StoreCompletionState>        finished_store_completion_state_;
    State                                       state_{State::IDLE};
    int                                         device_id_{-1};
    const std::chrono::milliseconds             store_completion_timeout_;

    std::shared_ptr<KVCacheManager>    cache_manager_;
    std::shared_ptr<const CacheConfig> cache_config_;
    size_t                             cache_model_id_{0};
    int                                cp_rank_{0};
    int                                cp_size_{1};

    // CacheStore can be injected after model construction, so resolve it once per forward.
    std::shared_ptr<CacheStore> active_cache_store_;
};

}  // namespace rtp_llm
