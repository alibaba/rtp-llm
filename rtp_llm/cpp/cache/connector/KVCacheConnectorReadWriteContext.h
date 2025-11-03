#pragma once

namespace rtp_llm {

class KVCacheConnectorReadWriteContext {
public:
    explicit KVCacheConnectorReadWriteContext(long request_id = 0): request_id_(request_id) {}
    virtual ~KVCacheConnectorReadWriteContext() = default;

public:
    virtual const KVCacheResource& kvCacheResource() const   = 0;
    virtual bool                   enableMemoryCache() const = 0;
    virtual bool                   enableRemoteCache() const = 0;

    inline long request_id() const {
        return request_id_;
    }

private:
    long request_id_ = 0;
};

}  // namespace rtp_llm