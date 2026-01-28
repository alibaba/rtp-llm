#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"

namespace rtp_llm {

class IKVCacheConnectorCoordinator {
public:
    virtual ~IKVCacheConnectorCoordinator() = default;

public:
    virtual std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& context,
                                                    const std::shared_ptr<KVCacheConnector::Meta>&           meta) = 0;

    virtual std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& context,
                                                     const std::shared_ptr<KVCacheConnector::Meta>&           meta) = 0;

    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int                                                      layer_id,
                      const std::shared_ptr<KVCacheConnectorReadWriteContext>& context,
                      const std::shared_ptr<KVCacheConnector::Meta>&           meta) = 0;

    virtual uint32_t convertToGlobalLayerId(size_t model_id, int local_layer_id) const = 0;
};

}  // namespace rtp_llm