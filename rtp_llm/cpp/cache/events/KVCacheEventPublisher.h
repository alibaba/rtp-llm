#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/cache/events/KVCacheEvent.h"

namespace rtp_llm {

enum class PublishResult {
    ACCEPTED,
    DISABLED,
    NOT_RUNNING,
    QUEUE_FULL,
};

enum class PublisherState {
    DISABLED,
    STARTING,
    LOGGING,
    REGISTERING,
    RESYNCING,
    READY,
    DEGRADED,
    STOPPED,
};

struct PublisherStatus {
    PublisherState state          = PublisherState::DISABLED;
    size_t         queue_size     = 0;
    uint64_t       accepted_count = 0;
    uint64_t       dropped_count  = 0;
};

// Cache mutation points depend only on this interface. Concrete publishers,
// transport code, batching, retries, and snapshots remain outside the cache.
class KVCacheEventPublisher {
public:
    virtual ~KVCacheEventPublisher() = default;

    virtual bool            start() noexcept                        = 0;
    virtual PublishResult   tryPublish(KVCacheEvent event) noexcept = 0;
    virtual void            stop() noexcept                         = 0;
    virtual PublisherStatus status() const noexcept                 = 0;
    virtual bool            enabled() const noexcept                = 0;
};

using KVCacheEventPublisherPtr = std::shared_ptr<KVCacheEventPublisher>;

}  // namespace rtp_llm
