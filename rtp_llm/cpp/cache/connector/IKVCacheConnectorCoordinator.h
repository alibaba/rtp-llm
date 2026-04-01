#pragma once

#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"

namespace rtp_llm {

class KVCacheConnectorReadWriteContext;

class IKVCacheConnectorCoordinator {
public:
    virtual ~IKVCacheConnectorCoordinator() = default;

    virtual bool hasActiveConnectors() const = 0;

    /// Returns global layer id; std::numeric_limits<uint32_t>::max() indicates invalid (caller must check before use).
    virtual uint32_t convertToGlobalLayerId(int model_id, int layer_id) const = 0;

    virtual std::shared_ptr<AsyncContext>
    asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) = 0;

    virtual std::shared_ptr<AsyncContext>
    asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) = 0;

    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) = 0;
};

}  // namespace rtp_llm
