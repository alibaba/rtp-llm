#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"

namespace rtp_llm {

struct KVCacheConnectorReadWriteContext {
    KVCacheConnectorReadWriteContext() = default;
    KVCacheConnectorReadWriteContext(const KVCacheResource&                  resource,
                                     std::shared_ptr<KVCacheConnector::Meta> meta,
                                     bool                                    enable_memory_cache = false):
        resource_(&resource), meta_(std::move(meta)), enable_memory_cache_(enable_memory_cache) {}

    virtual ~KVCacheConnectorReadWriteContext() = default;

    virtual const KVCacheResource& kvCacheResource() const {
        return *resource_;
    }

    virtual bool enableMemoryCache() const {
        return enable_memory_cache_;
    }

    virtual std::shared_ptr<KVCacheConnector::Meta> meta() const {
        return meta_;
    }

protected:
    const KVCacheResource*                  resource_ = nullptr;
    std::shared_ptr<KVCacheConnector::Meta> meta_;
    bool                                    enable_memory_cache_ = false;
};

}  // namespace rtp_llm