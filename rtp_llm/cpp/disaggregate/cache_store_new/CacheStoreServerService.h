#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store_new/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/TcpClient.h"

namespace rtp_llm {

class CacheStoreServerServiceLayerWatcher:
    public SingleLayerCacheBufferStore::Watcher,
    public std::enable_shared_from_this<CacheStoreServerServiceLayerWatcher> {
public:
    CacheStoreServerServiceLayerWatcher(const std::shared_ptr<TcpClient>&               tcp_client,
                                        int                                             layer_id,
                                        int                                             partition_count,
                                        int                                             partition_id,
                                        std::string                                     ip,
                                        uint32_t                                        port,
                                        uint32_t                                        rdma_port,
                                        int                                             context_id,
                                        const std::map<int64_t, std::vector<uint32_t>>& key_block_sizes);
    ~CacheStoreServerServiceLayerWatcher();

public:
    bool notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) override;

private:
    void loadToRemote(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);

private:
    std::shared_ptr<TcpClient> tcp_client_;

    int                                      partition_count_;
    int                                      partition_id_;
    std::string                              ip_;
    uint32_t                                 port_;
    uint32_t                                 rdma_port_;
    int                                      context_id_;
    std::map<int64_t, std::vector<uint32_t>> key_block_sizes_;
};

class LayerKVCacheTransferClosure: public ::google::protobuf::Closure {
public:
    LayerKVCacheTransferClosure(const std::shared_ptr<LayerCacheBuffer>&                    layer_cache_buffer,
                                const std::shared_ptr<CacheStoreServerServiceLayerWatcher>& layer_watcher,
                                const std::shared_ptr<TransferRequest>&                     transfer_request,
                                const std::shared_ptr<TransferResponse>&                    transfer_response,
                                arpc::ANetRPCController*                                    controller);

    ~LayerKVCacheTransferClosure();

public:
    void Run() override;

private:
    std::shared_ptr<LayerCacheBuffer>                    layer_cache_buffer_;
    std::shared_ptr<CacheStoreServerServiceLayerWatcher> watcher_;
    std::shared_ptr<TransferRequest>                     transfer_request_;
    std::shared_ptr<TransferResponse>                    transfer_response_;
    arpc::ANetRPCController*                             controller_;
};

class CacheStoreServerService: public CacheStoreService {
public:
    CacheStoreServerService(const std::shared_ptr<TcpClient>&             tcp_client,
                            const std::shared_ptr<LayerCacheBufferStore>& layer_cache_buffer_store);
    ~CacheStoreServerService();

public:
    void load(::google::protobuf::RpcController* controller,
              const ::CacheLoadRequest*          request,
              ::CacheLoadResponse*               response,
              ::google::protobuf::Closure*       done) override;

private:
    std::shared_ptr<TcpClient>             tcp_client_;
    std::shared_ptr<LayerCacheBufferStore> layer_cache_buffer_store_;
};

}  // namespace rtp_llm