#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/proto/service.pb.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClientLoadContext.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreClientService: public cache_store_proto::TransferService {
public:
    CacheStoreClientService(const std::shared_ptr<LoadContextStore>& load_context_store);
    ~CacheStoreClientService();

public:
    void transfer(::google::protobuf::RpcController*                  controller,
                  const cache_store_proto::LayerBlockTransferRequest* request,
                  cache_store_proto::LayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                        done) override;

private:
    void loadLayerBlocks(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                         const cache_store_proto::LayerBlockTransferRequest*   request,
                         cache_store_proto::LayerBlockTransferResponse*        response,
                         ::google::protobuf::Closure*                          done);

private:
    std::shared_ptr<LoadContextStore> load_context_store_;
};

}  // namespace cache_store
}  // namespace rtp_llm