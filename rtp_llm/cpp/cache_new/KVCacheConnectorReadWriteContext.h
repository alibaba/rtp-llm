#pragma once

namespace rtp_llm {

class KVCacheConnectorReadWriteContext {
public:
    virtual ~KVCacheConnectorReadWriteContext() = default;

public:
    virtual const KVCacheResourceV1& kvCacheResource() const   = 0;
    virtual bool                     enableMemoryCache() const = 0;
};

}  // namespace rtp_llm