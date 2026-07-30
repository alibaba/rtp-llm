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
// write() is valid only during the RUNNING portion of an active forward cycle.
// Once waitAllDone() starts draining, new writes are rejected.
class CacheStoreAsyncWriter: public CacheStoreWriter {
public:
    explicit CacheStoreAsyncWriter(int                             device_id              = -1,
                                   std::shared_ptr<KVCacheManager> cache_manager          = nullptr,
                                   size_t                          cache_model_id         = 0,
                                   std::optional<int>              mtp_cache_config_index = std::nullopt);
    ~CacheStoreAsyncWriter() override;

    void init();
    void waitAllDone();
    void write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
               const torch_ext::LayerKVCache&       layer_kv) override;

private:
    // Test-only convenience; production goes through write(). Tests reach it
    // via -fno-access-control.
    void submit(std::function<void()> task);
    void enqueueLocked(std::function<void()> task);

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
    // True only for the cycle's first failure; later failures are dropped as before.
    bool storeCurrentException();

    enum class State {
        IDLE,
        RUNNING,
        DRAINING
    };

    static const char*                        stateName(State state);
    static std::shared_ptr<const CacheConfig> selectCacheConfig(const std::shared_ptr<KVCacheManager>& cache_manager,
                                                                const std::optional<int>& mtp_cache_config_index);

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::mutex               state_mutex_;
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::mutex               exception_mutex_;
    std::exception_ptr       stored_exception_;
    State                    state_{State::IDLE};
    uint64_t                 cycle_id_{0};
    // Guarded by state_mutex_. True only for a cycle admitted without a CacheStore
    // under the CACHE_STORE_SKIP_WRITE_WHEN_UNREADY rollback switch; write() then
    // no-ops for the whole cycle (degraded skip mode; see init()).
    bool skip_cycle_writes_{false};
    int  device_id_{-1};

    const std::shared_ptr<KVCacheManager> cache_manager_;
    // Selected once during construction and immutable for the writer lifetime.
    const std::shared_ptr<const CacheConfig> cache_config_;
    size_t                                   cache_model_id_{0};
    int                                      cp_rank_{0};
    int                                      cp_size_{1};

    // CacheStore can be injected after model construction, so resolve it once per forward.
    std::shared_ptr<CacheStore> active_cache_store_;
};

}  // namespace rtp_llm
