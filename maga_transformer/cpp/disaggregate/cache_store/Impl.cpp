#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {
std::unique_ptr<MemoryUtilBase> createMemoryUtilImpl(bool rdma_mode) {
    if (rdma_mode) {
        throw std::runtime_error("rdma mode not supported");
    }
    return std::make_unique<NoRdmaMemoryUtilImpl>();
}

std::unique_ptr<MessagerClient> createMessagerClient(const std::shared_ptr<MemoryUtil>& memory_util) {
    return std::make_unique<MessagerClient>(memory_util);
}

std::unique_ptr<CacheStoreServiceImpl>
createCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                            const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                            const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter) {
    return std::make_unique<TcpCacheStoreServiceImpl>(memory_util, request_block_buffer_store, metrics_reporter);
}

std::unique_ptr<MessagerServer>
createMessagerServer(const std::shared_ptr<MemoryUtil>&                memory_util,
                     const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                     const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter) {
    return std::make_unique<MessagerServer>(memory_util, request_block_buffer_store, metrics_reporter);
}
}  // namespace rtp_llm