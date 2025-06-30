#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"

namespace rtp_llm {

TcpCacheStoreServiceImpl::TcpCacheStoreServiceImpl(
    const std::shared_ptr<MemoryUtil>&              memory_util,
    const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
    const kmonitor::MetricsReporterPtr&             metrics_reporter,
    const std::shared_ptr<arpc::TimerManager>&      timer_manager):
    CacheStoreServiceImpl(memory_util, request_block_buffer_store, metrics_reporter, timer_manager) {}

void TcpCacheStoreServiceImpl::loadImpl(::google::protobuf::RpcController* controller,
                                        const ::CacheLoadRequest*          request,
                                        ::CacheLoadResponse*               response,
                                        ::google::protobuf::Closure*       done) {
    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(
        metrics_reporter_,
        request->blocks_size(),
        request->blocks_size() ? request->blocks(0).len() * request->blocks_size() : 0,
        currentTimeUs() - request->request_send_start_time_us());
    loadTcpBlocks(request, response, collector, done);
}

void TcpCacheStoreServiceImpl::loadTcpBlocks(const ::CacheLoadRequest*                                    request,
                                             ::CacheLoadResponse*                                         response,
                                             const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                             ::google::protobuf::Closure*                                 done) {
    auto context = std::make_shared<TcpCacheStoreServiceImplContext>(
        request, response, collector, done, request_block_buffer_store_);
    if (!context) {
        RTP_LLM_LOG_WARNING("cache store service new context failed, request id is %s, request from %s",
                            request->requestid().c_str(),
                            request->client_ip().c_str());
        collector->markEnd(false);
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }

    auto timer_callback = [context]() { context->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER); };

    auto timer = timer_manager_->addTimer(request->timeout_ms(), std::move(timer_callback));
    if (timer == nullptr) {
        RTP_LLM_LOG_WARNING("cache store service add timer failed, request id is %s, request from %s",
                            request->requestid().c_str(),
                            request->client_ip().c_str());
        collector->markEnd(false);
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    context->setTimer(timer);

    RequestBlockBuffer::WatchFunc watch_func = [context](bool                                             ok,
                                                         const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        context->loadBlockOnTcp(ok, blocks);
    };

    if (!request_block_buffer_store_->setRequestBlockBufferWatchFunc(request->requestid(), std::move(watch_func))) {
        RTP_LLM_LOG_WARNING(
            "cache store service set request block buffer watch func failed, request id %s, request from %s",
            request->requestid().c_str(),
            request->client_ip().c_str());
        context->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
    }
}

}  // namespace rtp_llm