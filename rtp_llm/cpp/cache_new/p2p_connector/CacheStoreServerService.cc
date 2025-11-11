#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServerService.h"

namespace rtp_llm {
namespace cache_store {

CacheStoreServerServiceLayerWatcher::CacheStoreServerServiceLayerWatcher(
    const std::shared_ptr<TcpClient>&          tcp_client,
    const std::shared_ptr<KVCacheAllocator>&   kv_cache_allocator,
    rtp_llm::DeviceBase*                       device,
    int                                        layer_id,
    int                                        partition_count,
    int                                        partition_id,
    std::string                                ip,
    uint32_t                                   port,
    uint32_t                                   rdma_port,
    int                                        context_id,
    const std::map<int64_t, std::vector<int>>& cache_key_blocks):
    SingleLayerCacheBufferStore::Watcher(layer_id),
    tcp_client_(tcp_client),
    kv_cache_allocator_(kv_cache_allocator),
    device_(device),
    partition_count_(partition_count),
    partition_id_(partition_id),
    ip_(ip),
    port_(port),
    rdma_port_(rdma_port),
    context_id_(context_id),
    cache_key_blocks_(cache_key_blocks) {}

CacheStoreServerServiceLayerWatcher::~CacheStoreServerServiceLayerWatcher() {}

bool CacheStoreServerServiceLayerWatcher::notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    RTP_LLM_LOG_WARNING("notify, layer id: %d", layer_cache_buffer->layerId());
    if (layer_cache_buffer->layerId() != layer_id_) {
        RTP_LLM_LOG_WARNING("layer id is not equal to watcher layer id, watcher layer id: %d, layer id: %d",
                            layer_id_,
                            layer_cache_buffer->layerId());
        return false;
    }

    // TODO: optimize: 现在只会要么一层全部load, 要么不load, 不支持部分load, 不然淘汰策略不好实现.
    auto& block_cache_buffers = layer_cache_buffer->blockCacheBuffers();
    if (block_cache_buffers.size() != cache_key_blocks_.size()) {
        RTP_LLM_LOG_WARNING(
            "block cache buffers size is not equal to cache key blocks size, block cache buffers size: %d, cache key blocks size: %d",
            block_cache_buffers.size(),
            cache_key_blocks_.size());
        return false;
    }

    auto request = makeTransferRequest(layer_cache_buffer);
    if (request == nullptr) {
        RTP_LLM_LOG_WARNING("make transfer request failed, layer id: %d", layer_cache_buffer->layerId());
        return false;
    }

    // load to remote
    loadToRemote(layer_cache_buffer, request);
    RTP_LLM_LOG_WARNING("load to remote success, layer id: %d", layer_cache_buffer->layerId());
    return true;
}

std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>
CacheStoreServerServiceLayerWatcher::makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    // generate transfer request according to layer_cache_buffer and cache_keys_
    auto transfer_request = std::make_shared<cache_store_proto::LayerBlockTransferRequest>();
    transfer_request->set_context_id(context_id_);

    auto layer_block_info = transfer_request->add_layer_blocks();
    layer_block_info->set_layer_id(layer_cache_buffer->layerId());

    for (auto& [key, block_sizes] : cache_key_blocks_) {
        auto block = layer_cache_buffer->getBlockCacheBuffer(key);
        if (block == nullptr) {
            return nullptr;
        }

        auto kv_buffer = kv_cache_allocator_->convertIndexToBuffer(
            layer_cache_buffer->layerId(), block->block_id, partition_count_, partition_id_);

        if (kv_buffer.k_addr == nullptr) {
            RTP_LLM_LOG_WARNING("convert index to buffer failed, layer id: %d, block id: %d",
                                layer_cache_buffer->layerId(),
                                block->block_id);
            return nullptr;
        }
        if (kv_buffer.k_addr->size() != block_sizes[0]) {
            RTP_LLM_LOG_WARNING("block sizes is not equal to buffer size, layer id: %d, block id: %d",
                                layer_cache_buffer->layerId(),
                                block->block_id);
            return nullptr;
        }
        auto block_info = layer_block_info->add_blocks();
        block_info->set_key(key);

        auto block_buffer_info = block_info->add_blocks();
        setBlockBufferInfo(block_buffer_info, key, kv_buffer.k_addr);

        if (kv_buffer.v_addr != nullptr) {
            if (block_sizes.size() != 2 || block_sizes[1] != kv_buffer.v_addr->size()) {
                RTP_LLM_LOG_WARNING("block sizes is not equal to buffer size, layer id: %d, block id: %d",
                                    layer_cache_buffer->layerId(),
                                    block->block_id);
                return nullptr;
            }
            auto block_buffer_info = block_info->add_blocks();
            setBlockBufferInfo(block_buffer_info, key, kv_buffer.v_addr);
        }
    }
    return transfer_request;
}

void CacheStoreServerServiceLayerWatcher::setBlockBufferInfo(cache_store_proto::BlockBufferInfo* block_buffer_info,
                                                             int64_t                             key,
                                                             BufferPtr                           buffer) {
    block_buffer_info->set_len(buffer->size());
    auto tmp_buffer = static_cast<char*>(malloc(buffer->size()));
    auto dst_buffer =
        rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_UINT8, {buffer->size()}, tmp_buffer);

    device_->noBlockCopy({dst_buffer, *buffer});
    block_buffer_info->set_content(tmp_buffer, buffer->size());
    free(tmp_buffer);
}

void CacheStoreServerServiceLayerWatcher::loadToRemote(
    const std::shared_ptr<LayerCacheBuffer>&                             layer_cache_buffer,
    const std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>& transfer_request) {
    // send layer block transfer request to remote
    auto channel = tcp_client_->getChannel(ip_, port_);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("get channel failed, ip: %s, port: %d", ip_.c_str(), port_);
        return;
    }

    auto transfer_response = std::make_shared<cache_store_proto::LayerBlockTransferResponse>();
    auto controller        = new arpc::ANetRPCController();
    // TODO: timeout to config
    controller->SetExpireTime(100);

    auto closure = new LayerKVCacheTransferClosure(
        layer_cache_buffer, shared_from_this(), transfer_request, transfer_response, controller);

    cache_store_proto::TransferService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                 ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request.get(), transfer_response.get(), closure);
}

LayerKVCacheTransferClosure::LayerKVCacheTransferClosure(
    const std::shared_ptr<LayerCacheBuffer>&                              layer_cache_buffer,
    const std::shared_ptr<CacheStoreServerServiceLayerWatcher>&           layer_watcher,
    const std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>&  transfer_request,
    const std::shared_ptr<cache_store_proto::LayerBlockTransferResponse>& transfer_response,
    arpc::ANetRPCController*                                              controller):
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
                                                 const std::vector<CacheStoreServerWorker>&    worker_addrs,
                                                 rtp_llm::DeviceBase*                          device):
    tcp_client_(tcp_client),
    kv_cache_allocator_(kv_cache_allocator),
    layer_cache_buffer_store_(layer_cache_buffer_store),
    worker_addrs_(worker_addrs),
    device_(device) {}

CacheStoreServerService::~CacheStoreServerService() {}

void CacheStoreServerService::load(::google::protobuf::RpcController*           controller,
                                   const ::cache_store_proto::CacheLoadRequest* request,
                                   ::cache_store_proto::CacheLoadResponse*      response,
                                   ::google::protobuf::Closure*                 done) {
    auto context_id      = request->context_id();
    auto partition_count = request->partition_count();
    auto partition_id    = request->partition_id();
    auto ip              = request->ip();
    auto port            = request->port();
    auto rdma_port       = request->rdma_port();
    auto deadline_ms     = request->deadline_ms();

    for (auto& layer_cache_load_info : request->layer_cache_load_infos()) {
        auto                                layer_id = layer_cache_load_info.layer_id();
        std::map<int64_t, std::vector<int>> cache_key_blocks;
        for (auto& block_info : layer_cache_load_info.block_infos()) {
            std::vector<int> block_sizes;
            for (auto& buffer_size : block_info.buffer_size()) {
                block_sizes.push_back(buffer_size);
            }
            if (block_sizes.size() == 0) {
                RTP_LLM_LOG_WARNING("block sizes is empty, layer id: %d, cache key: %lld", layer_id, block_info.key());
                continue;
            }
            cache_key_blocks[block_info.key()] = block_sizes;
        }
        if (cache_key_blocks.size() == 0) {
            RTP_LLM_LOG_WARNING("cache key blocks is empty, layer id: %d", layer_id);
            continue;
        }
        auto watcher = std::make_shared<CacheStoreServerServiceLayerWatcher>(tcp_client_,
                                                                             kv_cache_allocator_,
                                                                             device_,
                                                                             layer_id,
                                                                             partition_count,
                                                                             partition_id,
                                                                             ip,
                                                                             port,
                                                                             rdma_port,
                                                                             context_id,
                                                                             cache_key_blocks);
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

void CacheStoreServerService::workerinfo(::google::protobuf::RpcController*            controller,
                                         const ::cache_store_proto::WorkerInfoRequest* request,
                                         ::cache_store_proto::WorkerInfoResponse*      response,
                                         ::google::protobuf::Closure*                  done) {
    RTP_LLM_LOG_ERROR("workerinfo request, worker addrs size: %d", worker_addrs_.size());
    for (auto& worker_addr : worker_addrs_) {
        auto worker_info = response->add_worker_infos();
        worker_info->set_ip(worker_addr.ip);
        worker_info->set_port(worker_addr.port);
        worker_info->set_rdma_port(worker_addr.rdma_port);
    }
    done->Run();
}

}  // namespace cache_store
}  // namespace rtp_llm