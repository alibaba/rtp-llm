#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include <stdexcept>

#if USING_ASCEND
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMessager.h"
#endif

namespace rtp_llm {
std::shared_ptr<MemoryUtil> createMemoryUtilImpl(const CacheStoreInitParams& params) {
#if USING_ASCEND
    if (params.transport == "hixl") {
        return HixlMemoryUtil::create(
            params.hixl_engine_id,
            autil::EnvUtil::getEnv("ASCEND_LOCAL_COMM_RES", std::string(R"({"version":"1.3"})")));
    }
#endif
    throw std::invalid_argument("unsupported cache store transport: " + params.transport);
}

std::shared_ptr<Messager> createMessager(const CacheStoreInitParams&                     params,
                                         const std::shared_ptr<MemoryUtil>&              memory_util,
                                         const std::shared_ptr<RequestBlockBufferStore>& store,
                                         const kmonitor::MetricsReporterPtr&             metrics) {
    if (params.transport.empty()) {
        return createMessager(memory_util, store, metrics);
    }
#if USING_ASCEND
    if (params.transport == "hixl") {
        return std::make_shared<HixlMessager>(
            params, std::static_pointer_cast<HixlMemoryUtil>(memory_util), store, metrics);
    }
#endif
    throw std::invalid_argument("unsupported cache store transport: " + params.transport);
}

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