#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"

namespace rtp_llm {

struct KVCacheConnectorControlParams {
    bool enable_memory_cache = false;
};

class IKVCacheConnectorCoordinator {
public:
    virtual ~IKVCacheConnectorCoordinator() = default;

public:
    virtual std::shared_ptr<AsyncContext> asyncRead(const KVCacheResource&                       resource,
                                                    const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                                    const KVCacheConnectorControlParams&         control_params) = 0;

    virtual std::shared_ptr<AsyncContext> asyncWrite(const KVCacheResource&                       resource,
                                                     const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                                     const KVCacheConnectorControlParams&         control_params) = 0;

    virtual std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                          layer_id,
                                                            const KVCacheResource&                       resource,
                                                            const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                                            const KVCacheConnectorControlParams& control_params) = 0;
};

}  // namespace rtp_llm