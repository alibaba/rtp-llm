#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMemoryUtilImpl.h"

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(bool rdma_mode) {
    if (rdma_mode) {
        return std::make_shared<RdmaMemoryUtilImpl>();
    } else {
        return std::make_shared<NoRdmaMemoryUtilImpl>();
    }
}

std::shared_ptr<Messager> createMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                                         const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                         const kmonitor::MetricsReporterPtr&             metrics_reporter) {
    if (memory_util->isRdmaMode()) {
        return std::shared_ptr<Messager>(new RdmaMessager(memory_util, request_block_buffer_store, metrics_reporter));
    }
    return std::shared_ptr<Messager>(new TcpMessager(memory_util, request_block_buffer_store, metrics_reporter));
}

}  // namespace rtp_llm