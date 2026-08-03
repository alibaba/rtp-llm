#pragma once

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class KVCacheConnectorReadWriteContext {
public:
    virtual ~KVCacheConnectorReadWriteContext() = default;

public:
    virtual const std::shared_ptr<Meta>& meta() const            = 0;
    virtual const KVCacheResource&       kvCacheResource() const = 0;
};

}  // namespace rtp_llm