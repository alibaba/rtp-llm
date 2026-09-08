#pragma once

#include <memory>
#include <optional>

#include <c10/core/Event.h>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm::legacy::p2p {

class KVCacheConnectorLayerContext {
public:
    virtual ~KVCacheConnectorLayerContext() = default;

    virtual const ::rtp_llm::KVCacheResource& kvCacheResource() const = 0;
    virtual int64_t                           requestId() const       = 0;
    virtual std::optional<c10::Event>         attentionEvent() const  = 0;
};

}  // namespace rtp_llm::legacy::p2p
