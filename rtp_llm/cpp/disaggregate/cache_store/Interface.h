#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MessagerClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MessagerServer.h"

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(bool rdma_mode);

std::unique_ptr<MessagerClient> createMessagerClient(const std::shared_ptr<MemoryUtil>& memory_util);

std::unique_ptr<CacheStoreServiceImpl>
createCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                            const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                            const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                            const std::shared_ptr<arpc::TimerManager>&        timer_manager);

std::unique_ptr<MessagerServer>
createMessagerServer(const std::shared_ptr<MemoryUtil>&                memory_util,
                     const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                     const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                     const std::shared_ptr<arpc::TimerManager>&        timer_manager);
}  // namespace rtp_llm
