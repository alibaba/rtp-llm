#pragma once

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"

namespace rtp_llm {

class NullPublisher final: public KVCacheEventPublisher {
public:
    bool            start() noexcept override;
    PublishResult   tryPublish(KVCacheEvent event) noexcept override;
    void            stop() noexcept override;
    PublisherStatus status() const noexcept override;
    bool            enabled() const noexcept override;
};

}  // namespace rtp_llm
