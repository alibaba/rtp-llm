#pragma once

#include "rtp_llm/cpp/cache_new/AsyncContext.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    class Meta {
    public:
        virtual ~Meta() = default;
    };

public:
    virtual bool                          init()                                               = 0;
    virtual std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta)         = 0;
    virtual std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                     const std::shared_ptr<Meta>&              meta)        = 0;
    virtual std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                            const std::shared_ptr<KVCacheResourceV1>& resource,
                                                            const std::shared_ptr<Meta>&              meta) = 0;
};

}  // namespace rtp_llm