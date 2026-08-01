#pragma once

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"

namespace rtp_llm {

class Meta;
class KVCacheResource;
class RequestPrefixMatchView;

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    virtual bool bindRequestResource(const std::shared_ptr<KVCacheResource>& resource,
                                     const std::shared_ptr<Meta>&            meta) {
        return true;
    }
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const RequestPrefixMatchView& view,
                                                          const std::shared_ptr<Meta>&  meta) = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context,
                                                         size_t                                    start_token,
                                                         size_t                                    token_count)                 = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta) = 0;
    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) = 0;
};

}  // namespace rtp_llm
