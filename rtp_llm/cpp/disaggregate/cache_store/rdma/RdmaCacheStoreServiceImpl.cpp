#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceImplContext.h"
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaCacheStoreServiceImpl::RdmaCacheStoreServiceImpl(
    const std::shared_ptr<MemoryUtil>&               memory_util,
    const std::shared_ptr<RequestBlockBufferStore>&  request_block_buffer_store,
    const kmonitor::MetricsReporterPtr&              metrics_reporter,
    const std::shared_ptr<TimerManager>&             timer_manager,
    const std::shared_ptr<LockedBlockBufferManager>& locked_block_buffer_manager,
    const std::shared_ptr<RdmaClient>&               rdma_client,
    const std::shared_ptr<RdmaServer>&               rdma_server):
    CacheStoreServiceImpl(
        memory_util, request_block_buffer_store, metrics_reporter, timer_manager, locked_block_buffer_manager),
    rdma_client_(rdma_client),
    rdma_server_(rdma_server) {}

void RdmaCacheStoreServiceImpl::loadImpl(::google::protobuf::RpcController* controller,
                                         const ::CacheLoadRequest*          request,
                                         ::CacheLoadResponse*               response,
                                         ::google::protobuf::Closure*       done) {
    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(
        metrics_reporter_,
        request->blocks_size(),
        request->blocks_size() ? request->blocks(0).len() * request->blocks_size() : 0,
        currentTimeUs() - request->request_send_start_time_us());

    for (int i = 0; i < request->blocks_size(); i++) {
        if (!request->blocks(i).has_rdma_info()) {  // should have rdma info
            RTP_LLM_LOG_WARNING("cache store service run on rdma mode, request has no rdma info, request %s from [%s]",
                                request->requestid().c_str(),
                                request->client_ip().c_str());
            response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            done->Run();
            return;
        }
    }

    // connect to client on rdma, client ip may not same with rdma connection addr
    std::vector<AddrPortChannel> rdma_connection_addr_ports;
    for (int i = 0; i < request->rdma_connections_size(); i++) {
        auto& rdma_connection = request->rdma_connections(i);
        rdma_connection_addr_ports.push_back({rdma_connection.ip(), rdma_connection.port(), rdma_connection.peer_channel()});
    }
    auto connection = rdma_server_->getConnection(rdma_connection_addr_ports);
    if (!connection) {
        RTP_LLM_LOG_WARNING("cache store service connect failed, request %s from [%s]",
                            request->requestid().c_str(),
                            request->client_ip().c_str());
        collector->markEnd(false);
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION);
        done->Run();
        return;
    }
    loadWriteBlocks(request, response, connection, collector, done);
}

void RdmaCacheStoreServiceImpl::loadWriteBlocks(const ::CacheLoadRequest*                                    request,
                                                ::CacheLoadResponse*                                         response,
                                                const std::shared_ptr<RdmaConnection>&                       connection,
                                                const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                                ::google::protobuf::Closure*                                 done) {
    auto context = std::make_shared<RdmaCacheStoreServiceImplContext>(
        request, response, collector, done, request_block_buffer_store_, connection);

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
        context->loadBlockOnRdma(ok, blocks);
    };

    if (!request_block_buffer_store_->setRequestBlockBufferWatchFunc(request->requestid(), std::move(watch_func))) {
        RTP_LLM_LOG_WARNING(
            "cache store service set request block buffer watch func failed, request id %s, request from %s",
            request->requestid().c_str(),
            request->client_ip().c_str());
        context->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
    }
}

void RdmaCacheStoreServiceImpl::transferImpl(::google::protobuf::RpcController*                   controller,
                                             const ::CacheTransferRequest*                        request,
                                             ::CacheTransferResponse*                             response,
                                             ::google::protobuf::Closure*                         done,
                                             const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                             const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks) {
    auto connection = rdma_client_->getTransferConnection(request->client_ip(), request->rdma_client_port());
    if (connection == nullptr) {
        RTP_LLM_LOG_WARNING("cache store service connect failed, request %s from [%s]",
                            request->request_id().c_str(),
                            request->client_ip().c_str());
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION);
        done->Run();
        return;
    }

    auto context = std::make_shared<CacheTransferServiceImplContext>(
        request, response, done, local_blocks, remote_blocks, locked_block_buffer_manager_, memory_util_, connection);
        
    int block_count = local_blocks.size();
    int total_block_size = block_count > 0 ? local_blocks[0]->len * block_count : 0;
    auto collector = std::make_shared<CacheStoreTransferMetricsCollector>(metrics_reporter_, block_count, total_block_size);
    context->run();
}

void RdmaCacheStoreServiceImpl::blockReadImpl(::google::protobuf::RpcController* controller,
                                              const ::BlockReadRequest*          request,
                                              BlockReadResponse*                 response,
                                              ::google::protobuf::Closure*       done) {
    response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
    done->Run();
}

}  // namespace rtp_llm