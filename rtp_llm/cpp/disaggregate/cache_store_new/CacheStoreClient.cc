#include "rtp_llm/cpp/disaggregate/cache_store_new/CacheStoreClient.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

CacheStoreClientClosure::CacheStoreClientClosure(const std::shared_ptr<CacheLoadRequest>&  cache_load_request,
                                                 const std::shared_ptr<CacheLoadResponse>& cache_load_response,
                                                 arpc::ANetRPCController*                  controller,
                                                 const std::shared_ptr<LoadContext>&       load_context):
    cache_load_request_(cache_load_request),
    cache_load_response_(cache_load_response),
    controller_(controller),
    load_context_(load_context) {}

void CacheStoreClientClosure::Run() {
    // TODO: if rpc failed

    // TODO: if response error code

    delete this;
}

// TODO: fill ip and port
CacheStoreClient::CacheStoreClient(const std::shared_ptr<TcpClient>& tcp_client):
    tcp_client_(tcp_client), ip_(""), port_(0) {}

CacheStoreClient::~CacheStoreClient() {}

std::shared_ptr<LoadContext>
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
    CacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);

    int64_t deadline_ms = currentTimeMs() + timeout_ms;
    auto    context_id  = generateContextId();

    std::shared_ptr<CacheLoadRequest> cache_load_request(new CacheLoadRequest());
    if (!generateCacheLoadRequest(
            layer_cache_buffers, deadline_ms, context_id, partition_count, partition_id, cache_load_request)) {
        RTP_LLM_LOG_ERROR("generate cache load request failed");
        return nullptr;
    }

    auto load_context = std::make_shared<LoadContext>(layer_cache_buffers, context_id);
    {
        std::lock_guard<std::mutex> lock(load_context_map_mutex_);
        load_context_map_[context_id] = load_context;
    }

    std::shared_ptr<CacheLoadResponse> cache_load_response(new CacheLoadResponse());

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(timeout_ms);

    auto closure = new CacheStoreClientClosure(cache_load_request, cache_load_response, controller, load_context);

    stub.load(controller, cache_load_request.get(), cache_load_response.get(), closure);
    return load_context;
}

bool CacheStoreClient::generateCacheLoadRequest(
    const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
    int64_t                                               deadline_ms,
    int64_t                                               context_id,
    int                                                   partition_count,
    int                                                   partition_id,
    const std::shared_ptr<CacheLoadRequest>&              cache_load_request) {

    cache_load_request->set_deadline_ms(deadline_ms);
    cache_load_request->set_partition_count(partition_count);
    cache_load_request->set_partition_id(partition_id);
    cache_load_request->set_ip(ip_);
    cache_load_request->set_port(port_);
    cache_load_request->set_context_id(context_id);

    for (auto& layer_cache_buffer : layer_cache_buffers) {
        auto layer_cache_load_info = cache_load_request->add_layer_cache_load_infos();
        layer_cache_load_info->set_layer_id(layer_cache_buffer->layerId());
        for (auto& [key, block] : layer_cache_buffer->blockCacheBuffers()) {
            auto cache_block_info = layer_cache_load_info->add_blocks();
            cache_block_info->set_cache_key(key);
            if (block->k_buffer) {
                cache_block_info->add_block_size(block->k_buffer->size());
            }
            if (block->v_buffer) {
                cache_block_info->add_block_size(block->v_buffer->size());
            }
        }
    }
    return true;
}

int64_t CacheStoreClient::generateContextId() {
    static std::atomic<int64_t> context_id_generator(0);
    return context_id_generator.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace rtp_llm