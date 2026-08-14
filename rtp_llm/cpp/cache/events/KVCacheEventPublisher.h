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
    // The event was not enqueued, but the publisher has atomically paused
    // incremental admission and scheduled an authoritative snapshot. Cache
    // mutation must continue and later transitions should keep probing the
    // publisher so it can resume after the snapshot handoff.
    DROPPED_RECOVERABLE,
};

// Values are exported through metrics and documented for alerting. New states
// must be appended without renumbering existing entries.
enum class PublisherState {
    DISABLED     = 0,
    STARTING     = 1,
    LOGGING      = 2,
    REGISTERING  = 3,
    RESYNCING    = 4,
    READY        = 5,
    DEGRADED     = 6,
    STOPPED      = 7,
    CIRCUIT_OPEN = 8,
    // The feature was requested but rejected before a publisher lifecycle
    // could start (for example by a topology or resource gate).
    GATED = 9,
};
static_assert(static_cast<int>(PublisherState::DISABLED) == 0);
static_assert(static_cast<int>(PublisherState::STARTING) == 1);
static_assert(static_cast<int>(PublisherState::LOGGING) == 2);
static_assert(static_cast<int>(PublisherState::REGISTERING) == 3);
static_assert(static_cast<int>(PublisherState::RESYNCING) == 4);
static_assert(static_cast<int>(PublisherState::READY) == 5);
static_assert(static_cast<int>(PublisherState::DEGRADED) == 6);
static_assert(static_cast<int>(PublisherState::STOPPED) == 7);
static_assert(static_cast<int>(PublisherState::CIRCUIT_OPEN) == 8);
static_assert(static_cast<int>(PublisherState::GATED) == 9);

struct PublisherStatus {
    PublisherState state                = PublisherState::DISABLED;
    size_t         queue_size           = 0;
    uint64_t       accepted_count       = 0;
    uint64_t       dropped_count        = 0;
    size_t         queue_high_watermark = 0;
    // Every unsuccessful KVCM HTTP/protocol response is retained even if the
    // publisher recovers between one-second metrics samples.
    uint64_t request_failure_count = 0;
    // Number of authoritative cache snapshots captured after bounded ingress
    // overflow. This is cumulative so a fast recovery remains observable.
    uint64_t overflow_recovery_count = 0;
    // Full-snapshot upload attempts and commits are separate cumulative
    // counters: retries can amplify control-plane traffic without losing
    // stream correctness.
    uint64_t snapshot_attempt_count = 0;
    uint64_t snapshot_commit_count  = 0;
};

// Cache mutation points depend only on this interface. Concrete publishers,
// transport code, batching, retries, and snapshots remain outside the cache.
class KVCacheEventPublisher {
public:
    virtual ~KVCacheEventPublisher() = default;

    virtual bool start() noexcept = 0;
    // SharedBlockCache calls tryPublish() while holding its mutation mutex.
    // Implementations must use a bounded, non-waiting path, be noexcept, and
    // avoid transport or logging I/O; slow work belongs on a publisher-owned
    // worker. A lock-free implementation may still retry briefly under
    // producer contention, but must never wait for queue space or a consumer.
    // The publisher object must remain alive until callers that observed a
    // stopped publisher have returned; stop() fences calls admitted before it
    // closes the implementation's one-shot publication gate.
    virtual PublishResult   tryPublish(KVCacheEvent event) noexcept = 0;
    virtual void            stop() noexcept                         = 0;
    virtual PublisherStatus status() const noexcept                 = 0;
    virtual bool            enabled() const noexcept                = 0;
};

using KVCacheEventPublisherPtr = std::shared_ptr<KVCacheEventPublisher>;

}  // namespace rtp_llm
