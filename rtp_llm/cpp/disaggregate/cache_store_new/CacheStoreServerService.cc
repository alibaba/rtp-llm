#include "rtp_llm/cpp/disaggregate/cache_store_new/CacheStoreServerService.h"

namespace rtp_llm {

CacheStoreServerServiceLayerWatcher::CacheStoreServerServiceLayerWatcher(
    const std::shared_ptr<TcpClient>&               tcp_client,
    int                                             layer_id,
    int                                             partition_count,
    int                                             partition_id,
    std::string                                     ip,
    uint32_t                                        port,
    uint32_t                                        rdma_port,
    int                                             context_id,
    const std::map<int64_t, std::vector<uint32_t>>& key_block_sizes):
    SingleLayerCacheBufferStore::Watcher(layer_id),
    tcp_client_(tcp_client),
    partition_count_(partition_count),
    partition_id_(partition_id),
    ip_(ip),
    port_(port),
    rdma_port_(rdma_port),
    context_id_(context_id),
    key_block_sizes_(key_block_sizes) {}

CacheStoreServerServiceLayerWatcher::~CacheStoreServerServiceLayerWatcher() {}

bool CacheStoreServerServiceLayerWatcher::notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    if (layer_cache_buffer->layerId() != layer_id_) {
        return false;
    }

    auto& block_cache_buffers = layer_cache_buffer->blockCacheBuffers();
    if (block_cache_buffers.size() != key_block_sizes_.size()) {
        // TODO: optimize: 现在只会要么一层全部load, 要么不load, 不支持部分load, 不然淘汰策略不好实现.
        return false;
    }

    // verify
    for (auto& [key, block] : block_cache_buffers) {
        auto iter = key_block_sizes_.find(key);
        if (iter == key_block_sizes_.end()) {
            return false;
        }
        auto& block_sizes = iter->second;
        if (block_sizes.size() > 0) {
            // verify k buffer
            if (block->k_buffer == nullptr) {
                return false;
            }
            if (block_sizes[0] != block->k_buffer->size() / partition_count_) {
                // block size not match, should not happen
                RTP_LLM_LOG_WARNING(
                    "k block size not match, key: %lld, block size: %d, partition count: %d, expect block size: %d",
                    key,
                    block->k_buffer->size(),
                    partition_count_,
                    block_sizes[0]);
                return false;
            }
        }
        if (block_sizes.size() == 2) {
            // verify v buffer
            if (block->v_buffer == nullptr) {
                return false;
            }
            if (block_sizes[1] != block->v_buffer->size() / partition_count_) {
                RTP_LLM_LOG_WARNING(
                    "v block size not match, key: %lld, block size: %d, partition count: %d, expect block size: %d",
                    key,
                    block->v_buffer->size(),
                    partition_count_,
                    block_sizes[1]);
                return false;
            }
        }
    }

    // load to remote
    loadToRemote(layer_cache_buffer);
    return true;
}

void CacheStoreServerServiceLayerWatcher::loadToRemote(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    // generate transfer request according to layer_cache_buffer and
    auto transfer_request = std::make_shared<TransferRequest>();
    transfer_request->set_context_id(context_id_);
    for (auto& [key, block] : layer_cache_buffer->blockCacheBuffers()) {
        auto layer_block_info = transfer_request->add_layer_blocks();
        layer_block_info->set_layer_id(layer_cache_buffer->layerId());
        if (block->k_buffer) {
            auto block_info = layer_block_info->add_blocks();
            block_info->set_key(key);
            block_info->set_len(block->k_buffer->size());
            // TODO: set content for tcp or rdma info for rdma
        }
        if (block->v_buffer) {
            auto block_info = layer_block_info->add_blocks();
            block_info->set_key(key);
            block_info->set_len(block->v_buffer->size());
            // TODO: set content for tcp or rdma info for rdma
        }
    }

    // send transfer request to remote
    auto channel = tcp_client_->getChannel(ip_, port_);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("get channel failed, ip: %s, port: %d", ip_.c_str(), port_);
        return;
    }

    auto transfer_response = std::make_shared<TransferResponse>();
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
    const std::shared_ptr<TransferRequest>&                     transfer_request,
    const std::shared_ptr<TransferResponse>&                    transfer_response,
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

CacheStoreServerService::CacheStoreServerService(
    const std::shared_ptr<TcpClient>&             tcp_client,
    const std::shared_ptr<LayerCacheBufferStore>& layer_cache_buffer_store):
    tcp_client_(tcp_client), layer_cache_buffer_store_(layer_cache_buffer_store) {}

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
        auto                                     layer_id = layer_cache_load_info.layer_id();
        std::map<int64_t, std::vector<uint32_t>> key_block_sizes;
        for (auto& block : layer_cache_load_info.blocks()) {
            auto key = block.cache_key();
            if (block.block_size_size() > 2 || block.block_size_size() == 0) {
                RTP_LLM_LOG_WARNING(
                    "now not support one cache key correspond to more than two buffers or no block size, key: %lld",
                    key);
                response->set_success(false);
                response->set_info(
                    "now not support one cache key correspond to more than two buffers or no block size");
                done->Run();
                return;
            }

            std::vector<uint32_t> block_sizes;
            for (auto& size : block.block_size()) {
                block_sizes.push_back(size);
            }
            key_block_sizes[key] = block_sizes;
        }
        auto watcher = std::make_shared<CacheStoreServerServiceLayerWatcher>(
            tcp_client_, layer_id, partition_count, partition_id, ip, port, rdma_port, context_id, key_block_sizes);
        auto store = layer_cache_buffer_store_->getSingleLayerCacheBufferStore(layer_id);
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

}  // namespace rtp_llm