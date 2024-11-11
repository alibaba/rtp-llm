#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"

#include "src/fastertransformer/utils/logger.h"

#include "autil/TimeUtility.h"
#include <unistd.h>

namespace rtp_llm {

CacheStoreServiceImpl::CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&              memory_util,
                                             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                                             const std::shared_ptr<arpc::TimerManager>&        timer_manager):
    memory_util_(memory_util),
    request_block_buffer_store_(request_block_buffer_store),
    metrics_reporter_(metrics_reporter),
    timer_manager_(timer_manager) {}

void CacheStoreServiceImpl::load(::google::protobuf::RpcController* controller,
                                 const ::CacheLoadRequest*          request,
                                 ::CacheLoadResponse*               response,
                                 ::google::protobuf::Closure*       done) {
    if (request_block_buffer_store_ == nullptr) {
        FT_LOG_WARNING(
            "cache store service has no block cache store, request failed, request from [%s], request id [%s]",
            request->client_ip().c_str(),
            request->requestid().c_str());
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    loadImpl(controller, request, response, done);
}

TcpCacheStoreServiceImpl::TcpCacheStoreServiceImpl(
    const std::shared_ptr<MemoryUtil>&                memory_util,
    const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
    const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
    const std::shared_ptr<arpc::TimerManager>&        timer_manager):
    CacheStoreServiceImpl(memory_util, request_block_buffer_store, metrics_reporter, timer_manager) {}

void TcpCacheStoreServiceImpl::loadImpl(::google::protobuf::RpcController* controller,
                                        const ::CacheLoadRequest*          request,
                                        ::CacheLoadResponse*               response,
                                        ::google::protobuf::Closure*       done) {
    int64_t start_time_us        = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t request_send_cost_us = start_time_us - request->request_send_start_time_us();
    auto    collector            = metrics_reporter_->makeServerLoadMetricsCollector(
        request->blocks_size(), request->blocks_size() ? request->blocks(0).len() : 0, request_send_cost_us);
    loadTcpBlocks(request, response, collector, done);
}

void TcpCacheStoreServiceImpl::loadTcpBlocks(const ::CacheLoadRequest*                                    request,
                                             ::CacheLoadResponse*                                         response,
                                             const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                             ::google::protobuf::Closure*                                 done) {
    auto context =
        std::make_shared<CacheStoreServiceImplContext>(request, response, collector, done, request_block_buffer_store_);
    if (!context) {
        FT_LOG_WARNING("cache store service new context failed");
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }

    auto timer_callback = [context]() { context->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER); };

    auto timer = timer_manager_->addTimer(request->timeout_ms(), std::move(timer_callback));
    if (timer == nullptr) {
        FT_LOG_WARNING("cache store service add timer failed");
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    context->setTimer(timer);

    RequestBlockBuffer::WatchFunc watch_func = [context](bool                                             ok,
                                                         const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        context->loadBlockOnTcp(ok, blocks);
    };
    request_block_buffer_store_->setRequestBlockBufferWatchFunc(request->requestid(), std::move(watch_func));
}
}  // namespace rtp_llm
