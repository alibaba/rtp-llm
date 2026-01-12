#pragma once

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    class AsyncContext {
    public:
        AsyncContext()          = default;
        virtual ~AsyncContext() = default;

    public:
        virtual bool success() const = 0;
        virtual void cancel()        = 0;
        virtual void waitDone()      = 0;
    };

    class Meta {
    public:
        virtual ~Meta() = default;
    };

public:
    virtual bool                          init()                                               = 0;
    virtual std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResource>& resource,
                                                    const std::shared_ptr<Meta>&            meta)         = 0;
    virtual std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                     const std::shared_ptr<Meta>&            meta)        = 0;
    virtual std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                     layer_id,
                                                            const std::shared_ptr<KVCacheResource>& resource,
                                                            const std::shared_ptr<Meta>&            meta) = 0;
};

}  // namespace rtp_llm