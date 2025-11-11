#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/proto/service.pb.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClientLoadContext.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreClientService: public cache_store_proto::TransferService {
public:
    CacheStoreClientService(const std::shared_ptr<LoadContextStore>& load_context_store, rtp_llm::DeviceBase* device);
    ~CacheStoreClientService();

public:
    void transfer(::google::protobuf::RpcController*                  controller,
                  const cache_store_proto::LayerBlockTransferRequest* request,
                  cache_store_proto::LayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                        done) override;

private:
    bool loadLayerBlocks(const std::shared_ptr<LayerCacheBuffer>&       layer_cache_buffer,
                         const cache_store_proto::LayerBlockBufferInfo& layer_block_info);
    void copyBlockBuffer(const BufferPtr& block_buffer, const cache_store_proto::BlockBufferInfo& block_buffer_info);

private:
    std::shared_ptr<LoadContextStore> load_context_store_;
    rtp_llm::DeviceBase*              device_;
};

}  // namespace cache_store
}  // namespace rtp_llm