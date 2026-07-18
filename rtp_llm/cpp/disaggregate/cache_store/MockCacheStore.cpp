#include "rtp_llm/cpp/disaggregate/cache_store/MockCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <strings.h>

namespace rtp_llm {

// ─────────────────────────────────────────────────────────────────────────────
// MockRemoteStoreTask: a RemoteStoreTask that is already completed with success.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

class MockRemoteStoreTask: public RemoteStoreTask {
public:
    MockRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>& request, CheckCancelFunc check_cancel_func):
        RemoteStoreTask(request, check_cancel_func) {}

    void waitDone() override {}
    bool success() const override {
        return true;
    }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Helper: fill block buffers with zero so the decode engine sees valid KV
// cache memory and does not need to recompute the entire prefill.
//
// GPU memory  → cudaMemset (synchronous, default stream)
// CPU memory  → memset
// ─────────────────────────────────────────────────────────────────────────────
void fillBlockBuffersWithZero(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    if (!request_block_buffer) {
        return;
    }
    auto blocks = request_block_buffer->getBlocks();
    for (const auto& [key, block] : blocks) {
        if (!block || !block->addr || block->len == 0) {
            continue;
        }
        void* ptr = block->addr.get();
        if (block->gpu_mem) {
            cudaMemset(ptr, 0, block->len);
        } else {
            memset(ptr, 0, block->len);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment variable helper
// ─────────────────────────────────────────────────────────────────────────────

bool MockCacheStore::isMockModeEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv(kEnvMockMode.c_str());
        return value != nullptr
               && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
                   || strcasecmp(value, "yes") == 0);
    }();
    return enabled;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core transfer methods (mock behavior)
// ─────────────────────────────────────────────────────────────────────────────

void MockCacheStore::store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                           CacheStoreStoreDoneCallback                callback) {
    if (request_block_buffer == nullptr || !request_block_buffer->isValid()) {
        RTP_LLM_LOG_WARNING("mock cache store call store failed, request block is invalid");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    // Mock: fill block memory with zero so the data appears valid, then
    // immediately callback with success (no actual KV transmission).
    fillBlockBuffersWithZero(request_block_buffer);
    RTP_LLM_LOG_DEBUG("mock cache store store success, request id is %s, blocks count is %zu",
                      request_block_buffer->getRequestId().c_str(),
                      request_block_buffer->getBlocksCount());
    callback(true, CacheStoreErrorCode::None);
}

void MockCacheStore::load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                          CacheStoreLoadDoneCallback                 callback,
                          const std::string&                         ip,
                          uint32_t                                   port,
                          uint32_t                                   rdma_port,
                          uint32_t                                   timeout_ms,
                          int                                        partition_count,
                          int                                        partition_id) {
    if (request_block_buffer == nullptr || !request_block_buffer->isValid() || ip.empty()) {
        RTP_LLM_LOG_WARNING("mock cache store run load failed, invalid params");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    // Mock: fill block memory with zero so the decode engine sees valid KV
    // cache and does not need to recompute the entire prefill.
    fillBlockBuffersWithZero(request_block_buffer);
    RTP_LLM_LOG_DEBUG("mock cache store load success, request id is %s, blocks count is %zu, ip is %s",
                      request_block_buffer->getRequestId().c_str(),
                      request_block_buffer->getBlocksCount(),
                      ip.c_str());
    callback(true, CacheStoreErrorCode::None);
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch transfer methods (mock behavior)
//
// loadBuffers/storeBuffers create a LoadContext/StoreContext backed by this
// MockCacheStore. The context's doCall() invokes MockCacheStore::load()/store()
// which synchronously callbacks success, so the context is already completed
// by the time these methods return.
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<LoadContext>
MockCacheStore::loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                            const std::string&                                      ip,
                            uint32_t                                                port,
                            uint32_t                                                rdma_port,
                            int64_t                                                 timeout_ms,
                            LoadContext::CheckCancelFunc                            check_cancel_func,
                            int                                                     partition_count,
                            int                                                     partition_id) {
    if (request_block_buffers.empty() || ip.empty()) {
        return nullptr;
    }

    auto load_context = std::make_shared<LoadContext>(shared_from_this(), /*combine_load=*/false);
    load_context->load(
        request_block_buffers, ip, port, rdma_port, timeout_ms, check_cancel_func, partition_count, partition_id);
    return load_context;
}

std::shared_ptr<StoreContext>
MockCacheStore::storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                             int64_t                                                 timeout_ms) {
    if (request_block_buffers.empty()) {
        return nullptr;
    }

    auto store_context = std::make_shared<StoreContext>(shared_from_this());
    store_context->store(request_block_buffers, timeout_ms);
    return store_context;
}

// ─────────────────────────────────────────────────────────────────────────────
// Remote store methods
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<RemoteStoreTask>
MockCacheStore::submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&                    request,
                                      const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                      RemoteStoreTask::CheckCancelFunc                              check_cancel_func) {
    RTP_LLM_LOG_DEBUG("mock cache store submit remote store task, request id is %s, request is %s",
                      request->request_id.c_str(),
                      request->toString().c_str());
    return std::make_shared<MockRemoteStoreTask>(request, check_cancel_func);
}

void MockCacheStore::releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task) {
    // No-op: mock mode does not track remote store tasks
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer management methods
// ─────────────────────────────────────────────────────────────────────────────

bool MockCacheStore::regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) {
    return true;
}

std::shared_ptr<BlockBuffer> MockCacheStore::findUserBuffer(const std::string& buffer_key) {
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility methods
// ─────────────────────────────────────────────────────────────────────────────

const std::shared_ptr<MemoryUtil>& MockCacheStore::getMemoryUtil() const {
    return null_memory_util_;
}

void MockCacheStore::debugInfo() {
    RTP_LLM_LOG_INFO("MockCacheStore: mock mode enabled, no actual KV transmission");
}

}  // namespace rtp_llm
