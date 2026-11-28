#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(bool rdma_mode);
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(const CacheStoreInitParams& params);
std::shared_ptr<Messager>   createMessager(const CacheStoreInitParams&                     params,
                                           const std::shared_ptr<MemoryUtil>&              memory_util,
                                           const std::shared_ptr<RequestBlockBufferStore>& store,
                                           const kmonitor::MetricsReporterPtr&             metrics);

std::shared_ptr<Messager> createMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                                         const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                         const kmonitor::MetricsReporterPtr&             metrics_reporter);
}  // namespace rtp_llm
