#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"

#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

namespace rtp_llm {

bool TcpMessager::init(MessagerInitParams params) {
    init_params_ = params;

    tcp_client_ = std::make_shared<TcpClient>();
    if (!tcp_client_->init(init_params_.io_thread_count)) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp client init failed");
        return false;
    }

    tcp_server_ = std::make_shared<TcpServer>();
    if (!tcp_server_->init(init_params_.io_thread_count, init_params_.worker_thread_count, true)) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp server init failed");
        return false;
    }

    service_ = std::make_shared<TcpCacheStoreServiceImpl>(memory_util_,
                                                          request_block_buffer_store_,
                                                          metrics_reporter_,
                                                          timer_manager_,
                                                          locked_block_buffer_manager_,
                                                          tcp_client_);
    if (!tcp_server_->registerService(service_.get())) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp server register service failed");
        return false;
    }

    if (!tcp_server_->start(init_params_.server_port)) {
        RTP_LLM_LOG_WARNING("messager start failed, tcp server start failed");
        return false;
    }
    RTP_LLM_LOG_INFO("tcp messager init success, server port %u, io thread count %u, worker thread count %u",
                     init_params_.server_port,
                     init_params_.io_thread_count,
                     init_params_.worker_thread_count);
    return true;
}

void TcpMessager::load(const std::shared_ptr<LoadRequest>&                          request,
                       const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    auto channel = tcp_client_->getChannel(request->ip, request->port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("messager client get channel failed, ip %s", request->ip.c_str());
        request->callback(false, CacheStoreErrorCode::LoadConnectFailed);
        return;
    }

    auto load_request = makeLoadRequest(request);
    if (load_request == nullptr) {
        RTP_LLM_LOG_WARNING("messager client generate load request failed");
        request->callback(false, CacheStoreErrorCode::LoadSendRequestFailed);
        return;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(request->timeout_ms);

    CacheLoadResponse* load_response = new CacheLoadResponse;
    auto*              closure       = new TcpCacheStoreLoadServiceClosure(memory_util_,
                                                        request->request_block_buffer,
                                                        controller,
                                                        load_request,
                                                        load_response,
                                                        request->callback,
                                                        collector);

    collector->markRequestCallBegin();
    KvCacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.load(controller, load_request, load_response, closure);
}

bool TcpMessager::generateBlockInfo(BlockBufferInfo*                    block_info,
                                    const std::shared_ptr<BlockBuffer>& block,
                                    uint32_t                            partition_count,
                                    uint32_t                            partition_id) {
    block_info->set_key(block->key);
    block_info->set_len(block->len / partition_count);  // real len
    block_info->set_addr((int64_t)(block->addr.get()) + block_info->len() * partition_id);
    return true;
}

}  // namespace rtp_llm