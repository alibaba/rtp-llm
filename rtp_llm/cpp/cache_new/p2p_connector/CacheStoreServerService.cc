#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServerService.h"

namespace rtp_llm {

CacheStoreServerServiceLayerWatcher::CacheStoreServerServiceLayerWatcher(
    const std::shared_ptr<TcpClient>&        tcp_client,
    const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator,
    int                                      layer_id,
    int                                      partition_count,
    int                                      partition_id,
    std::string                              ip,
    uint32_t                                 port,
    uint32_t                                 rdma_port,
    int                                      context_id,
    const std::vector<int64_t>&              cache_keys):
    SingleLayerCacheBufferStore::Watcher(layer_id),
    tcp_client_(tcp_client),
    kv_cache_allocator_(kv_cache_allocator),
    partition_count_(partition_count),
    partition_id_(partition_id),
    ip_(ip),
    port_(port),
    rdma_port_(rdma_port),
    context_id_(context_id),
    cache_keys_(cache_keys) {}

CacheStoreServerServiceLayerWatcher::~CacheStoreServerServiceLayerWatcher() {}

bool CacheStoreServerServiceLayerWatcher::notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    if (layer_cache_buffer->layerId() != layer_id_) {
        return false;
    }

    // TODO: optimize: 现在只会要么一层全部load, 要么不load, 不支持部分load, 不然淘汰策略不好实现.
    auto& block_cache_buffers = layer_cache_buffer->blockCacheBuffers();
    if (block_cache_buffers.size() != cache_keys_.size()) {
        return false;
    }

    auto request = makeTransferRequest(layer_cache_buffer);
    if (request == nullptr) {
        return false;
    }

    // load to remote
    loadToRemote(layer_cache_buffer, request);
    return true;
}

std::shared_ptr<LayerBlockTransferRequest>
CacheStoreServerServiceLayerWatcher::makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    // generate transfer request according to layer_cache_buffer and cache_keys_
    auto transfer_request = std::make_shared<LayerBlockTransferRequest>();
    transfer_request->set_context_id(context_id_);

    auto layer_block_info = transfer_request->add_layer_blocks();
    layer_block_info->set_layer_id(layer_cache_buffer->layerId());

    for (auto& key : cache_keys_) {
        auto block = layer_cache_buffer->getBlockCacheBuffer(key);
        if (block == nullptr) {
            return nullptr;
        }

        auto kv_buffer = kv_cache_allocator_->convertIndexToBuffer(
            layer_cache_buffer->layerId(), block->block_id, partition_count_, partition_id_);
        if (kv_buffer.k_addr != nullptr) {
            auto block_buffer_info = layer_block_info->add_blocks();
            block_buffer_info->set_key(key);
            block_buffer_info->set_len(kv_buffer.k_addr->size());
            // TODO: set content for tcp or rdma info for rdma
        }
    }
    return transfer_request;
}

void CacheStoreServerServiceLayerWatcher::loadToRemote(
    const std::shared_ptr<LayerCacheBuffer>&          layer_cache_buffer,
    const std::shared_ptr<LayerBlockTransferRequest>& transfer_request) {
    // send layer block transfer request to remote
    auto channel = tcp_client_->getChannel(ip_, port_);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("get channel failed, ip: %s, port: %d", ip_.c_str(), port_);
        return;
    }

    auto transfer_response = std::make_shared<LayerBlockTransferResponse>();
    auto controller        = new arpc::ANetRPCController();
    // TODO: timeout to config
    controller->SetExpireTime(100);

    auto closure = new LayerKVCacheTransferClosure(
        layer_cache_buffer, shared_from_this(), transfer_request, transfer_response, controller);

    TransferService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                              ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request.get(), transfer_response.get(), closure);
}

LayerKVCacheTransferClosure::LayerKVCacheTransferClosure(
    const std::shared_ptr<LayerCacheBuffer>&                    layer_cache_buffer,
    const std::shared_ptr<CacheStoreServerServiceLayerWatcher>& layer_watcher,
    const std::shared_ptr<LayerBlockTransferRequest>&           transfer_request,
    const std::shared_ptr<LayerBlockTransferResponse>&          transfer_response,
    arpc::ANetRPCController*                                    controller):
    layer_cache_buffer_(layer_cache_buffer),
    watcher_(layer_watcher),
    transfer_request_(transfer_request),
    transfer_response_(transfer_response),
    controller_(controller) {}

LayerKVCacheTransferClosure::~LayerKVCacheTransferClosure() {
    if (controller_) {
        delete controller_;
    }
}

void LayerKVCacheTransferClosure::Run() {
    // TODO: if rpc failed
    // TODO: if response error code

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("transfer request failed, controller err is %d", controller_->GetErrorCode());
        return;
    }

    if (!transfer_response_->success()) {
        RTP_LLM_LOG_WARNING("transfer request failed, response err is %s", transfer_response_->info().c_str());
        return;
    }

    RTP_LLM_LOG_DEBUG("transfer request success");
    delete this;
}

CacheStoreServerService::CacheStoreServerService(const std::shared_ptr<TcpClient>&             tcp_client,
                                                 const std::shared_ptr<KVCacheAllocator>&      kv_cache_allocator,
                                                 const std::shared_ptr<LayerCacheBufferStore>& layer_cache_buffer_store,
                                                 const std::vector<CacheStoreServerWorker>&    worker_addrs):
    tcp_client_(tcp_client),
    kv_cache_allocator_(kv_cache_allocator),
    layer_cache_buffer_store_(layer_cache_buffer_store),
    worker_addrs_(worker_addrs) {}

CacheStoreServerService::~CacheStoreServerService() {}

void CacheStoreServerService::load(::google::protobuf::RpcController* controller,
                                   const ::CacheLoadRequest*          request,
                                   ::CacheLoadResponse*               response,
                                   ::google::protobuf::Closure*       done) {
    auto context_id      = request->context_id();
    auto partition_count = request->partition_count();
    auto partition_id    = request->partition_id();
    auto ip              = request->ip();
    auto port            = request->port();
    auto rdma_port       = request->rdma_port();
    auto deadline_ms     = request->deadline_ms();

    for (auto& layer_cache_load_info : request->layer_cache_load_infos()) {
        auto                 layer_id = layer_cache_load_info.layer_id();
        std::vector<int64_t> cache_keys;
        for (auto& key : layer_cache_load_info.cache_keys()) {
            cache_keys.push_back(key);
        }
        auto watcher = std::make_shared<CacheStoreServerServiceLayerWatcher>(tcp_client_,
                                                                             kv_cache_allocator_,
                                                                             layer_id,
                                                                             partition_count,
                                                                             partition_id,
                                                                             ip,
                                                                             port,
                                                                             rdma_port,
                                                                             context_id,
                                                                             cache_keys);
        auto store   = layer_cache_buffer_store_->getSingleLayerCacheBufferStore(layer_id);
        if (store == nullptr) {
            RTP_LLM_LOG_WARNING("get single layer cache buffer store failed, layer id: %d", layer_id);
            response->set_success(false);
            response->set_info("get single layer cache buffer store failed");
            done->Run();
            return;
        }
        store->setLayerCacheBufferWatchFunc(watcher, deadline_ms);
    }

    response->set_success(true);
    done->Run();
}

void CacheStoreServerService::workerinfo(::google::protobuf::RpcController* controller,
                                         const ::WorkerInfoRequest*         request,
                                         ::WorkerInfoResponse*              response,
                                         ::google::protobuf::Closure*       done) {
    for (auto& worker_addr : worker_addrs_) {
        auto worker_info = response->add_worker_infos();
        worker_info->set_ip(worker_addr.ip);
        worker_info->set_port(worker_addr.port);
        worker_info->set_rdma_port(worker_addr.rdma_port);
    }
    done->Run();
}

}  // namespace rtp_llm