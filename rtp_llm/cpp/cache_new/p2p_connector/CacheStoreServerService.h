#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/proto/service.pb.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreServerServiceLayerWatcher:
    public SingleLayerCacheBufferStore::Watcher,
    public std::enable_shared_from_this<CacheStoreServerServiceLayerWatcher> {
public:
    CacheStoreServerServiceLayerWatcher(const std::shared_ptr<TcpClient>&        tcp_client,
                                        const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator,
                                        int                                      layer_id,
                                        int                                      partition_count,
                                        int                                      partition_id,
                                        std::string                              ip,
                                        uint32_t                                 port,
                                        uint32_t                                 rdma_port,
                                        int                                      context_id,
                                        const std::vector<int64_t>&              cache_keys);
    ~CacheStoreServerServiceLayerWatcher();

public:
    bool notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) override;

private:
    std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>
         makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);
    void loadToRemote(const std::shared_ptr<LayerCacheBuffer>&                             layer_cache_buffer,
                      const std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>& transfer_request);

private:
    std::shared_ptr<TcpClient> tcp_client_;
    // 用kvcache allocator 来做 [block_id, partition_id] -> buffer 的映射
    std::shared_ptr<KVCacheAllocator> kv_cache_allocator_;

    int                  partition_count_;
    int                  partition_id_;
    std::string          ip_;
    uint32_t             port_;
    uint32_t             rdma_port_;
    int                  context_id_;
    std::vector<int64_t> cache_keys_;
};

class LayerKVCacheTransferClosure: public ::google::protobuf::Closure {
public:
    LayerKVCacheTransferClosure(const std::shared_ptr<LayerCacheBuffer>&                             layer_cache_buffer,
                                const std::shared_ptr<CacheStoreServerServiceLayerWatcher>&          layer_watcher,
                                const std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>& transfer_request,
                                const std::shared_ptr<cache_store_proto::LayerBlockTransferResponse>& transfer_response,
                                arpc::ANetRPCController*                                              controller);

    ~LayerKVCacheTransferClosure();

public:
    void Run() override;

private:
    std::shared_ptr<LayerCacheBuffer>                              layer_cache_buffer_;
    std::shared_ptr<CacheStoreServerServiceLayerWatcher>           watcher_;
    std::shared_ptr<cache_store_proto::LayerBlockTransferRequest>  transfer_request_;
    std::shared_ptr<cache_store_proto::LayerBlockTransferResponse> transfer_response_;
    arpc::ANetRPCController*                                       controller_;
};

class CacheStoreServerService: public cache_store_proto::CacheStoreService {
public:
    CacheStoreServerService(const std::shared_ptr<TcpClient>&             tcp_client,
                            const std::shared_ptr<KVCacheAllocator>&      kv_cache_allocator,
                            const std::shared_ptr<LayerCacheBufferStore>& layer_cache_buffer_store,
                            const std::vector<CacheStoreServerWorker>&    worker_addrs);
    ~CacheStoreServerService();

public:
    void load(::google::protobuf::RpcController*           controller,
              const ::cache_store_proto::CacheLoadRequest* request,
              ::cache_store_proto::CacheLoadResponse*      response,
              ::google::protobuf::Closure*                 done) override;

    void workerinfo(::google::protobuf::RpcController*            controller,
                    const ::cache_store_proto::WorkerInfoRequest* request,
                    ::cache_store_proto::WorkerInfoResponse*      response,
                    ::google::protobuf::Closure*                  done) override;

private:
    std::shared_ptr<TcpClient>             tcp_client_;
    std::shared_ptr<KVCacheAllocator>      kv_cache_allocator_;
    std::shared_ptr<LayerCacheBufferStore> layer_cache_buffer_store_;
    std::vector<CacheStoreServerWorker>    worker_addrs_;
};

}  // namespace cache_store
}  // namespace rtp_llm