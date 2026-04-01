#pragma once

#include <memory>
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/core/Event.h"

namespace rtp_llm {

/// @brief Context for asyncWriteByLayer (P2P write path, per-layer granularity).
/// Carries the physical transfer metadata produced after each GPU kernel layer completes.
/// Intentionally separate from KVCacheConnectorReadWriteContext to avoid mixing
/// request-level semantics (Meta) with layer-level GPU transfer state.
class KVCacheConnectorLayerContext {
public:
    virtual ~KVCacheConnectorLayerContext() = default;

    virtual const KVCacheResource& kvCacheResource() const = 0;
    virtual int64_t                requestId() const       = 0;
    virtual AsyncEventPtr          attentionEvent() const  = 0;
};

}  // namespace rtp_llm
