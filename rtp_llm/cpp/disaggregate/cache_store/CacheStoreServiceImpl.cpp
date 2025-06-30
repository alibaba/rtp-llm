#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include "autil/TimeUtility.h"
#include <unistd.h>

namespace rtp_llm {

CacheStoreServiceImpl::CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&              memory_util,
                                             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                             const kmonitor::MetricsReporterPtr&             metrics_reporter,
                                             const std::shared_ptr<arpc::TimerManager>&      timer_manager):
    memory_util_(memory_util),
    request_block_buffer_store_(request_block_buffer_store),
    metrics_reporter_(metrics_reporter),
    timer_manager_(timer_manager) {}

void CacheStoreServiceImpl::load(::google::protobuf::RpcController* controller,
                                 const ::CacheLoadRequest*          request,
                                 ::CacheLoadResponse*               response,
                                 ::google::protobuf::Closure*       done) {
    if (request_block_buffer_store_ == nullptr) {
        RTP_LLM_LOG_WARNING(
            "cache store service has no block cache store, request failed, request from [%s], request id [%s]",
            request->client_ip().c_str(),
            request->requestid().c_str());
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    loadImpl(controller, request, response, done);
}

}  // namespace rtp_llm
