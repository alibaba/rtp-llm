#pragma once

#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/core/Event.h"

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
        Meta() = default;
        Meta(int64_t request_id, DeviceEventPtr event): request_id_(request_id), event_(event) {}
        virtual ~Meta() = default;

    public:
        int64_t requestId() const {
            return request_id_;
        }
        DeviceEventPtr event() const {
            return event_;
        }

    private:
        int64_t        request_id_;
        DeviceEventPtr event_;
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