#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"

#include <memory>

namespace rtp_llm {

// MockCacheStore: A CacheStore implementation that mocks KV cache transmission.
// When CACHE_STORE_MOCK_MODE env var is set, store()/load() immediately return
// success without transferring actual KV data. The gRPC handshake, scheduling,
// cancel/timeout logic all remain real. The engine still does real prefill
// forward + decode generation.
class MockCacheStore: public CacheStore {
public:
    MockCacheStore()           = default;
    ~MockCacheStore() override = default;

public:
    // Core transfer methods (mock behavior)
    void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
               CacheStoreStoreDoneCallback                callback) override;

    void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&                         ip,
              uint32_t                                   port,
              uint32_t                                   rdma_port,
              uint32_t                                   timeout_ms      = 1000,
              int                                        partition_count = 1,
              int                                        partition_id    = 0) override;

    // Batch transfer methods (mock behavior)
    std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                const std::string&                                      ip,
                uint32_t                                                port,
                uint32_t                                                rdma_port,
                int64_t                                                 timeout_ms,
                LoadContext::CheckCancelFunc                            check_cancel_func,
                int                                                     partition_count = 1,
                int                                                     partition_id    = 0) override;

    std::shared_ptr<StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                 int64_t                                                 timeout_ms) override;

    // Remote store methods
    std::shared_ptr<RemoteStoreTask>
         submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&                    request,
                               const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                               RemoteStoreTask::CheckCancelFunc                              check_cancel_func) override;
    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task) override;

    // Buffer management methods
    bool                         regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) override;
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string& buffer_key) override;

    // Utility methods
    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override;
    void                               debugInfo() override;

public:
    // Check if CACHE_STORE_MOCK_MODE env var is enabled.
    // Uses static local to ensure env var is read only once.
    static bool isMockModeEnabled();

private:
    // Null MemoryUtil — mock mode does not register memory regions
    std::shared_ptr<MemoryUtil> null_memory_util_;
};

}  // namespace rtp_llm
