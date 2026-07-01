#pragma once

#include <memory>
#include <torch/extension.h>
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

/// @brief Context for asyncWriteByLayer (P2P write path, per-layer granularity).
/// Carries the physical transfer metadata produced after each GPU kernel layer completes.
/// Intentionally separate from KVCacheConnectorReadWriteContext to avoid mixing
/// request-level semantics (Meta) with layer-level GPU transfer state.
class KVCacheConnectorLayerContext {
public:
    virtual ~KVCacheConnectorLayerContext() = default;

    virtual const KVCacheResource&        kvCacheResource() const     = 0;
    virtual KVCacheResourcePtr            heldKVCacheResource() const = 0;
    virtual int64_t                       requestId() const           = 0;
    virtual std::shared_ptr<torch::Event> attentionEvent() const      = 0;

    // Absolute deadline (ms since epoch) for this request. Used by P2P prefill
    // to align the per-layer ComputedLayerCacheBuffer lifetime with the
    // request's business deadline, instead of a hard-coded store-wait timeout.
    // Implementations may return INT64_MAX when the request has no business
    // deadline; the worker treats that as "no deadline" and falls back to its
    // configured store-wait timeout.
    virtual int64_t deadlineMs() const = 0;
};

}  // namespace rtp_llm
