#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(bool rdma_mode) {
    if (rdma_mode) {
        throw std::runtime_error("rdma mode not supported");
    }
    return std::make_shared<NoRdmaMemoryUtilImpl>();
}

std::shared_ptr<Messager> createMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                                         const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                         const kmonitor::MetricsReporterPtr&             metrics_reporter) {
    if (memory_util->isRdmaMode()) {
        throw std::runtime_error("rdma mode not supported");
    }
    return std::shared_ptr<Messager>(new TcpMessager(memory_util, request_block_buffer_store, metrics_reporter));
}

}  // namespace rtp_llm