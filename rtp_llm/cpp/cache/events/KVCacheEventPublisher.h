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

// Values are exported through metrics and documented for alerting. New states
// must be appended without renumbering existing entries.
enum class PublisherState {
    DISABLED    = 0,
    STARTING    = 1,
    REGISTERING = 3,
    RESYNCING   = 4,
    READY       = 5,
    DEGRADED    = 6,
    STOPPED     = 7,
};
static_assert(static_cast<int>(PublisherState::STOPPED) == 7);

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
};

using KVCacheEventPublisherPtr = std::shared_ptr<KVCacheEventPublisher>;

}  // namespace rtp_llm
