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
    // Tree-owned prefix that P2P must skip. It may include blocks whose tiered
    // load is still pending, so it is not equivalent to device-ready reuse.
    virtual size_t treeCoveredBlockNum() const {
        return kvCacheResource().deviceReuseBlockNum();
    }
};

}  // namespace rtp_llm
