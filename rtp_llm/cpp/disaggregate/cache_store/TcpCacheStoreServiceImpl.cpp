#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceImplContext.h"

namespace rtp_llm {

TcpCacheStoreServiceImpl::TcpCacheStoreServiceImpl(
    const std::shared_ptr<MemoryUtil>&               memory_util,
    const std::shared_ptr<RequestBlockBufferStore>&  request_block_buffer_store,
    const kmonitor::MetricsReporterPtr&              metrics_reporter,
    const std::shared_ptr<TimerManager>&             timer_manager,
    const std::shared_ptr<LockedBlockBufferManager>& locked_block_buffer_manager,
    const std::shared_ptr<TcpClient>&                tcp_client):
    CacheStoreServiceImpl(
        memory_util, request_block_buffer_store, metrics_reporter, timer_manager, locked_block_buffer_manager),
    tcp_client_(tcp_client),
    device_(rtp_llm::DeviceFactory::getDefaultDevice()) {}

void TcpCacheStoreServiceImpl::loadImpl(::google::protobuf::RpcController* controller,
                                        const ::CacheLoadRequest*          request,
                                        ::CacheLoadResponse*               response,
                                        ::google::protobuf::Closure*       done) {
    // int64_t start_time_us        = currentTimeUs();
    // int64_t request_send_cost_us = start_time_us - request->request_send_start_time_us();
    auto    collector            = std::make_shared<CacheStoreServerLoadMetricsCollector>(
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

void TcpCacheStoreServiceImpl::transferImpl(::google::protobuf::RpcController*                   controller,
                                            const ::CacheTransferRequest*                        request,
                                            ::CacheTransferResponse*                             response,
                                            ::google::protobuf::Closure*                         done,
                                            const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                            const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks) {
    auto connection = tcp_client_->getTransferConnection(request->client_ip(), request->client_port());
    auto context    = std::make_shared<CacheTransferServiceImplContext>(
        request, response, done, local_blocks, remote_blocks, locked_block_buffer_manager_, memory_util_, connection);
    context->run();
}

void TcpCacheStoreServiceImpl::blockReadImpl(::google::protobuf::RpcController* controller,
                                             const ::BlockReadRequest*          request,
                                             BlockReadResponse*                 response,
                                             ::google::protobuf::Closure*       done) {
    for (int i = 0; i < request->blocks_size(); i++) {
        auto& block_info   = request->blocks(i);
        auto  block_buffer = request_block_buffer_store_->findUserBuffer(block_info.key());
        if (block_buffer == nullptr) {
            RTP_LLM_LOG_WARNING("cache store service block read find user buffer failed, block key %s",
                                request->blocks(i).key().c_str());
            response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
            done->Run();
            return;
        }
        int64_t block_buffer_start_addr = (int64_t)block_buffer->addr.get();
        int64_t block_buffer_end_addr   = block_buffer_start_addr + block_buffer->len;

        if (block_buffer_start_addr <= block_info.addr()
            && block_buffer_end_addr >= block_info.addr() + block_info.len()) {
            auto resp_block_info = response->add_blocks();
            resp_block_info->set_key(block_info.key());
            resp_block_info->set_addr(block_info.addr());
            resp_block_info->set_len(block_info.len());

            auto src_buffer = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_GPU,
                                              rtp_llm::DataType::TYPE_UINT8,
                                              {block_info.len()},
                                              (void*)block_info.addr());

            auto tmp_buffer = static_cast<char*>(malloc(block_info.len()));
            auto dst_buffer = rtp_llm::Buffer(
                rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_UINT8, {block_info.len()}, tmp_buffer);

            device_->noBlockCopy({dst_buffer, src_buffer});
            resp_block_info->set_content(tmp_buffer, block_info.len());
            free(tmp_buffer);
        }
    }

    response->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
    done->Run();
}

}  // namespace rtp_llm