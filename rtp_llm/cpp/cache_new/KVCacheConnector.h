#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

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

public:
    virtual bool init() = 0;
    virtual std::shared_ptr<AsyncContext>
    asyncRead(const BatchKVCacheResourcePtr& resource, const std::string& ip, uint32_t port)                        = 0;
    virtual std::shared_ptr<AsyncContext> asyncWrite(const BatchKVCacheResourcePtr& resource, DeviceEventPtr event) = 0;
    virtual std::shared_ptr<AsyncContext>
                   asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, DeviceEventPtr event) = 0;
    virtual size_t match(const std::vector<int64_t>& keys) const                                                  = 0;
};

}  // namespace rtp_llm