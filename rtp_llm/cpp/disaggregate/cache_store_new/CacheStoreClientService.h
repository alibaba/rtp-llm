#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store_new/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/LoadContext.h"

namespace rtp_llm {

class CacheStoreClientService: public TransferService {
public:
    CacheStoreClientService(const std::shared_ptr<LoadContextStore>& load_context_store);
    ~CacheStoreClientService();

public:
    void transfer(::google::protobuf::RpcController* controller,
                  const ::TransferRequest*           request,
                  ::TransferResponse*                response,
                  ::google::protobuf::Closure*       done) override;

private:
    void loadLayerBlocks(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                         const ::TransferRequest*                              request,
                         ::TransferResponse*                                   response,
                         ::google::protobuf::Closure*                          done);

private:
    std::shared_ptr<LoadContextStore> load_context_store_;
};

}  // namespace rtp_llm