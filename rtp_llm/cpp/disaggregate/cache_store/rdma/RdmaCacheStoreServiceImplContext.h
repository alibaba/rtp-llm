#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"

namespace rtp_llm {

class RdmaCacheStoreServiceImplContext: public CacheStoreServiceImplContext {
public:
    RdmaCacheStoreServiceImplContext(const CacheLoadRequest*                                      request,
                                     CacheLoadResponse*                                           response,
                                     const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                     ::google::protobuf::Closure*                                 done,
                                     const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                     const std::shared_ptr<RdmaConnection>&          connection);
    virtual ~RdmaCacheStoreServiceImplContext() = default;

public:
    void loadBlockOnRdma(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks);

private:
    void onWriteDone(bool success, const std::string& block_key);

private:
    std::shared_ptr<RdmaConnection> rdma_connection_;
};

}  // namespace rtp_llm