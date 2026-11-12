#pragma once

#include <memory>
#include <optional>

#include <c10/core/Event.h>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class KVCacheConnectorLayerContext {
public:
    virtual ~KVCacheConnectorLayerContext() = default;

    virtual const ::rtp_llm::KVCacheResource& kvCacheResource() const = 0;
    virtual int64_t                           requestId() const       = 0;
    virtual std::optional<c10::Event>         attentionEvent() const  = 0;
    // Module that owns the local layer being published (0 = main, 1..N = MTP).
    // Defaults to 0 for backward compatibility with single-module callers.
    virtual size_t moduleId() const {
        return 0;
    }
};

}  // namespace rtp_llm
