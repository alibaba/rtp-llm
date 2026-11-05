#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreLoadServiceClosure.h"
#include "aios/network/accl-barex/include/accl/barex/xchannel.h"

#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"

namespace rtp_llm {

bool RdmaMessager::init(MessagerInitParams params) {
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

    rdma_client_ = std::make_shared<RdmaClient>(
        memory_util_, init_params_.rdma_connect_timeout_ms, init_params_.rdma_qp_count_per_connection);
    if (!rdma_client_->init(init_params_.rdma_io_thread_count, init_params_.rdma_worker_thread_count)) {
        RTP_LLM_LOG_WARNING("messager init failed, rdma client init failed");
        return false;
    }

    rdma_server_ = std::make_shared<RdmaServer>(memory_util_);
    if (!rdma_server_->init(
            init_params_.rdma_server_port, init_params_.rdma_io_thread_count, init_params_.rdma_worker_thread_count)) {
        RTP_LLM_LOG_WARNING("messager init failed, rdma server init failed, listen port %d",
                            init_params_.rdma_server_port);
        return false;
    }

    service_ = std::make_shared<RdmaCacheStoreServiceImpl>(memory_util_,
                                                           request_block_buffer_store_,
                                                           metrics_reporter_,
                                                           timer_manager_,
                                                           locked_block_buffer_manager_,
                                                           rdma_client_,
                                                           rdma_server_);

    if (!tcp_server_->registerService(service_.get())) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp server register service failed");
        return false;
    }

    if (!tcp_server_->start(init_params_.server_port)) {
        RTP_LLM_LOG_WARNING("messager start failed, tcp server start failed");
        return false;
    }
    RTP_LLM_LOG_INFO(
        "rdma messager init success, server port %u, io thread count %u, worker thread count %u, rdma server port %u, rdma io thread count %u, rdma worker thread count %u",
        init_params_.server_port,
        init_params_.io_thread_count,
        init_params_.worker_thread_count,
        init_params_.rdma_server_port,
        init_params_.rdma_io_thread_count,
        init_params_.rdma_worker_thread_count);
    return true;
}

void RdmaMessager::load(const std::shared_ptr<LoadRequest>&                          request,
                        const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    auto channel = tcp_client_->getChannel(request->ip, request->port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("messager client get channel failed, ip %s:%u", request->ip.c_str(), request->port);
        collector->markEnd(false);
        request->callback(false, CacheStoreErrorCode::LoadConnectFailed);
        return;
    }

    auto rdma_connection = rdma_client_->getConnection(request->ip, request->rdma_port);
    if (!rdma_connection) {
        RTP_LLM_LOG_WARNING(
            "messager client get rdma connection failed, ip %s:%u", request->ip.c_str(), request->rdma_port);
        collector->markEnd(false);
        request->callback(false, CacheStoreErrorCode::LoadConnectFailed);
        return;
    }

    auto load_request = makeLoadRequest(rdma_connection, request);
    if (load_request == nullptr) {
        RTP_LLM_LOG_WARNING("messager client generate load request failed");
        collector->markEnd(false);
        request->callback(false, CacheStoreErrorCode::LoadSendRequestFailed);
        return;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(request->timeout_ms);

    CacheLoadResponse*                load_response = new CacheLoadResponse;
    RdmaCacheStoreLoadServiceClosure* closure       = new RdmaCacheStoreLoadServiceClosure(rdma_client_,
                                                                                     rdma_connection,
                                                                                     memory_util_,
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

CacheLoadRequest* RdmaMessager::makeLoadRequest(const std::shared_ptr<RdmaConnection>& rdma_connection,
                                                const std::shared_ptr<LoadRequest>&    request) {
    auto load_request = Messager::makeLoadRequest(request);
    if (load_request == nullptr) {
        return nullptr;
    }

    auto blocks = request->request_block_buffer->getBlocks();
    for (int i = 0; i < load_request->blocks_size(); i++) {
        auto block_msg = load_request->mutable_blocks(i);
        auto block     = blocks[block_msg->key()];
        if (!block || !generateBlockInfo(block_msg, block, 1, 0)) {
            delete load_request;
            return nullptr;
        }
    }

    auto connections = rdma_connection->getAllConnections();
    for (auto& conn : connections) {
        auto local = conn->GetLocalAddrAndPort();
        auto connection_proto = load_request->add_rdma_connections();
        connection_proto->set_ip(local.first);
        connection_proto->set_port(local.second);
        connection_proto->set_peer_channel(conn->GetChannel()->GetPeerChannel());
    }
    return load_request;
}

bool RdmaMessager::generateBlockInfo(BlockBufferInfo*                    block_info,
                                     const std::shared_ptr<BlockBuffer>& block,
                                     uint32_t                            partition_count,
                                     uint32_t                            partition_id) {
    block_info->set_key(block->key);
    block_info->set_len(block->len / partition_count);  // real len
    block_info->set_addr((int64_t)(block->addr.get()) + block_info->len() * partition_id);

    auto rdma_info = block_info->mutable_rdma_info();

    ::accl::barex::memp_t mem;
    if (!memory_util_->findMemoryMr((void*)&mem, block->addr.get(), block->len, block->gpu_mem, block->adopted)) {
        RTP_LLM_LOG_WARNING("messager client find buffer mr failed, block is %s", block->key.c_str());
        return false;
    }

    rdma_info->set_addr(
        reinterpret_cast<uint64_t>((void*)((int64_t)(block->addr.get()) + block_info->len() * partition_id)));
    for (auto& [nicid, mr] : mem.mrs) {
        auto nic_rkey = rdma_info->add_nic_rkeys();
        nic_rkey->set_nicid(nicid);
        nic_rkey->set_rkey(mr->rkey);
    }
    return true;
}
}  // namespace rtp_llm
