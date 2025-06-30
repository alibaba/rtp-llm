#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"

namespace rtp_llm {

class TcpCacheStoreServiceImplContext: public CacheStoreServiceImplContext {
public:
    TcpCacheStoreServiceImplContext(const CacheLoadRequest*                                      request,
                                    CacheLoadResponse*                                           response,
                                    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                    ::google::protobuf::Closure*                                 done,
                                    const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store):
        CacheStoreServiceImplContext(request, response, collector, done, request_block_buffer_store) {}
    virtual ~TcpCacheStoreServiceImplContext() = default;

public:
    void loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& block);

private:
    bool writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                            const std::shared_ptr<BlockBufferInfo>& peer_block);
};

}  // namespace rtp_llm