#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(bool rdma_mode) {
    if (rdma_mode) {
        throw std::runtime_error("rdma mode not supported");
    }
    return std::make_shared<NoRdmaMemoryUtilImpl>();
}

std::unique_ptr<MessagerClient> createMessagerClient(const std::shared_ptr<MemoryUtil>& memory_util) {
    return std::make_unique<MessagerClient>(memory_util);
}

std::unique_ptr<CacheStoreServiceImpl>
createCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                            const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                            const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                            const std::shared_ptr<arpc::TimerManager>&        timer_manager) {
    return std::make_unique<TcpCacheStoreServiceImpl>(
        memory_util, request_block_buffer_store, metrics_reporter, timer_manager);
}

std::unique_ptr<MessagerServer>
createMessagerServer(const std::shared_ptr<MemoryUtil>&                memory_util,
                     const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                     const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                     const std::shared_ptr<arpc::TimerManager>&        timer_manager) {
    return std::make_unique<MessagerServer>(memory_util, request_block_buffer_store, metrics_reporter, timer_manager);
}
}  // namespace rtp_llm