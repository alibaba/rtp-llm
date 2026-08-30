#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "autil/ThreadPool.h"
#include "rtp_llm/models_py/bindings/CacheStoreWriter.h"

namespace rtp_llm {

class CacheConfig;
class CacheStore;
class KVCacheManager;

// Offloads writeCacheStore CPU-heavy work to a background thread pool so the
// main thread can keep launching CUDA kernels without stalling.
// Lifecycle: init() -> write()* -> waitAllDone() -> init() -> ...
class CacheStoreAsyncWriter: public CacheStoreWriter {
public:
    explicit CacheStoreAsyncWriter(int                             device_id              = -1,
                                   std::shared_ptr<KVCacheManager> cache_manager          = nullptr,
                                   size_t                          cache_model_id         = 0,
                                   std::optional<int>              mtp_cache_config_index = std::nullopt,
                                   int                             forward_cp_rank        = 0,
                                   int                             forward_cp_size        = 1);
    ~CacheStoreAsyncWriter() override;

    void init();
    void waitAllDone();
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

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::mutex               state_mutex_;
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::mutex               exception_mutex_;
    std::exception_ptr       stored_exception_;
    State                    state_{State::IDLE};
    int                      device_id_{-1};

    std::shared_ptr<KVCacheManager>    cache_manager_;
    std::shared_ptr<const CacheConfig> cache_config_;
    size_t                             cache_model_id_{0};
    int                                cp_rank_{0};
    int                                cp_size_{1};

    // CacheStore can be injected after model construction, so resolve it once per forward.
    std::shared_ptr<CacheStore> active_cache_store_;
};

}  // namespace rtp_llm
