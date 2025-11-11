#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClient.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace cache_store {

CacheStoreClientClosure::CacheStoreClientClosure(
    const std::shared_ptr<cache_store_proto::CacheLoadRequest>&  cache_load_request,
    const std::shared_ptr<cache_store_proto::CacheLoadResponse>& cache_load_response,
    arpc::ANetRPCController*                                     controller,
    const std::shared_ptr<CacheStoreClientLoadContext>&          load_context):
    cache_load_request_(cache_load_request),
    cache_load_response_(cache_load_response),
    controller_(controller),
    load_context_(load_context) {}

void CacheStoreClientClosure::Run() {
    if (controller_->Failed()) {
        load_context_->setFailed();
        RTP_LLM_LOG_ERROR("cache load request failed, controller err is %s", controller_->ErrorText().c_str());
        return;
    }

    if (!cache_load_response_->success()) {
        load_context_->setFailed();
        RTP_LLM_LOG_ERROR("cache load response failed, response err is %s", cache_load_response_->info().c_str());
        return;
    }

    delete this;
}

CacheStoreClient::CacheStoreClient(const std::shared_ptr<TcpClient>& tcp_client,
                                   const std::shared_ptr<TcpServer>& tcp_server,
                                   rtp_llm::DeviceBase*              device):
    tcp_client_(tcp_client), tcp_server_(tcp_server), device_(device), load_context_store_(new LoadContextStore()) {
    cache_store_client_service_ = std::make_unique<CacheStoreClientService>(load_context_store_, device_);
}

CacheStoreClient::~CacheStoreClient() {}

bool CacheStoreClient::init() {
    if (!tcp_server_->registerService(cache_store_client_service_.get())) {
        RTP_LLM_LOG_ERROR("cache store client init failed : register service failed");
        return false;
    }
    RTP_LLM_LOG_INFO("cache store client init success, server port %u", tcp_server_->getPort());
    return true;
}

std::vector<CacheStoreServerWorker> CacheStoreClient::getPeerWorkerInfo(const std::string& ip, uint32_t port) {
    auto channel = tcp_client_->getChannel(ip, port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("get channel failed, ip: %s, port: %u", ip.c_str(), port);
        return std::vector<CacheStoreServerWorker>();
    }
    cache_store_proto::CacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                   ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);

    cache_store_proto::WorkerInfoRequest  request;
    cache_store_proto::WorkerInfoResponse response;

    arpc::ANetRPCController controller;
    controller.SetExpireTime(100);

    // TODO(yujing.zc): may need to change to async
    stub.workerinfo(&controller, &request, &response, nullptr);

    if (controller.Failed()) {
        RTP_LLM_LOG_WARNING("get peer worker info failed, controller err is %s", controller.ErrorText().c_str());
        return std::vector<CacheStoreServerWorker>();
    }

    std::vector<CacheStoreServerWorker> worker_info;
    for (auto& worker : response.worker_infos()) {
        worker_info.push_back(CacheStoreServerWorker(worker.ip(), worker.port(), worker.rdma_port()));
    }
    return worker_info;
}

std::shared_ptr<CacheStoreClientLoadContext>
CacheStoreClient::asyncLoad(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                            int64_t                                               timeout_ms,
                            const std::string&                                    ip,
                            uint32_t                                              port,
                            int                                                   partition_count,
                            int                                                   partition_id) {
    auto channel = tcp_client_->getChannel(ip, port);
    if (channel == nullptr) {
        return nullptr;
    }
    cache_store_proto::CacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                   ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);

    int64_t deadline_ms = currentTimeMs() + timeout_ms;
    auto    context_id  = generateContextId();

    std::shared_ptr<cache_store_proto::CacheLoadRequest> cache_load_request(new cache_store_proto::CacheLoadRequest());
    if (!generateCacheLoadRequest(
            layer_cache_buffers, deadline_ms, context_id, partition_count, partition_id, cache_load_request)) {
        RTP_LLM_LOG_ERROR("generate cache load request failed");
        return nullptr;
    }

    auto load_context = std::make_shared<CacheStoreClientLoadContext>(layer_cache_buffers, context_id, deadline_ms);
    load_context_store_->addLoadContext(load_context);

    std::shared_ptr<cache_store_proto::CacheLoadResponse> cache_load_response(
        new cache_store_proto::CacheLoadResponse());

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(timeout_ms);

    auto closure = new CacheStoreClientClosure(cache_load_request, cache_load_response, controller, load_context);

    stub.load(controller, cache_load_request.get(), cache_load_response.get(), closure);
    return load_context;
}

bool CacheStoreClient::generateCacheLoadRequest(
    const std::vector<std::shared_ptr<LayerCacheBuffer>>&       layer_cache_buffers,
    int64_t                                                     deadline_ms,
    int64_t                                                     context_id,
    int                                                         partition_count,
    int                                                         partition_id,
    const std::shared_ptr<cache_store_proto::CacheLoadRequest>& cache_load_request) {

    cache_load_request->set_deadline_ms(deadline_ms);
    cache_load_request->set_partition_count(partition_count);
    cache_load_request->set_partition_id(partition_id);
    cache_load_request->set_ip(tcp_server_->getIP());
    cache_load_request->set_port(tcp_server_->getPort());
    cache_load_request->set_context_id(context_id);

    for (auto& layer_cache_buffer : layer_cache_buffers) {
        auto layer_cache_load_info = cache_load_request->add_layer_cache_load_infos();
        layer_cache_load_info->set_layer_id(layer_cache_buffer->layerId());
        for (auto& [key, block] : layer_cache_buffer->blockCacheBuffers()) {
            auto block_info = layer_cache_load_info->add_block_infos();
            block_info->set_key(key);
            if (block->buffer1 != nullptr) {
                block_info->add_buffer_size(block->buffer1->size());
            }
            if (block->buffer2 != nullptr) {
                block_info->add_buffer_size(block->buffer2->size());
            }
        }
    }
    return true;
}

int64_t CacheStoreClient::generateContextId() {
    static std::atomic<int64_t> context_id_generator(0);
    return context_id_generator.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace cache_store
}  // namespace rtp_llm