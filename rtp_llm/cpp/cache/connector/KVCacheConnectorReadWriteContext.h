#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"

namespace rtp_llm {

class KVCacheConnectorReadWriteContext {
public:
    virtual ~KVCacheConnectorReadWriteContext() = default;

public:
    using Meta                                                     = KVCacheConnector::Meta;
    virtual const std::shared_ptr<Meta>& meta() const              = 0;
    virtual const KVCacheResource&       kvCacheResource() const   = 0;
    virtual bool                         enableMemoryCache() const = 0;
};

}  // namespace rtp_llm