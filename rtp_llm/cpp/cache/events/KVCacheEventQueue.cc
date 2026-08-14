#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm::detail {
namespace {

size_t validatedQueueCapacity(size_t requested_capacity) {
    const size_t capacity = std::max<size_t>(requested_capacity, 1);
    if (capacity > kKVCacheEventMaxQueueCapacity) {
        throw std::length_error("KV cache event queue capacity exceeds resource safety limit");
    }
    return capacity;
}

size_t ringCapacityFor(size_t usable_capacity) {
    size_t           value              = std::max<size_t>(usable_capacity, 2);
    constexpr size_t kHighestPowerOfTwo = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    if (value > kHighestPowerOfTwo) {
        throw std::length_error("KV cache event queue capacity is too large");
    }
    --value;
    for (size_t shift = 1; shift < std::numeric_limits<size_t>::digits; shift *= 2) {
        value |= value >> shift;
    }
    return value + 1;
}

}  // namespace

KVCacheEventQueue::KVCacheEventQueue(size_t capacity):
    capacity_(validatedQueueCapacity(capacity)),
    // A power-of-two backing ring keeps sequence/index wrap mathematically
    // consistent; size_ still enforces the exact configured usable capacity.
    ring_capacity_(ringCapacityFor(capacity_)),
    ring_mask_(ring_capacity_ - 1),
    cells_(std::make_unique<Cell[]>(ring_capacity_)) {
    if (sem_init(&consumer_signal_, /*pshared=*/0, /*value=*/0) != 0) {
        throw std::system_error(errno, std::generic_category(), "initialize KV cache event queue signal");
    }
    for (size_t i = 0; i < ring_capacity_; ++i) {
        cells_[i].sequence.store(i, std::memory_order_relaxed);
    }
}

KVCacheEventQueue::~KVCacheEventQueue() {
    sem_destroy(&consumer_signal_);
}

QueuePushResult KVCacheEventQueue::tryPush(KVCacheEvent event) noexcept {
    auto producer = producer_gate_.tryEnter();
    if (!producer) {
        return QueuePushResult::STOPPED;
    }

    size_t current_size = size_.load(std::memory_order_relaxed);
    do {
        if (current_size >= capacity_) {
            // Shutdown wins over overflow once it is observable. This keeps
            // an explicit stop from being misreported as backpressure by a
            // producer that was admitted immediately before stop().
            return stopped() ? QueuePushResult::STOPPED : QueuePushResult::FULL;
        }
    } while (!size_.compare_exchange_weak(
        current_size, current_size + 1, std::memory_order_acq_rel, std::memory_order_relaxed));

    if (stopped()) {
        size_.fetch_sub(1, std::memory_order_release);
        return QueuePushResult::STOPPED;
    }

    updateHighWatermark(current_size + 1);
    enqueue(std::move(event));
    signalConsumer();
    return QueuePushResult::ACCEPTED;
}

std::vector<KVCacheEvent> KVCacheEventQueue::waitPop(size_t max_batch_size, std::chrono::milliseconds timeout) {
    const size_t              max_count = std::max<size_t>(max_batch_size, 1);
    std::vector<KVCacheEvent> batch;
    batch.reserve(std::min(max_count, published_size_.load(std::memory_order_acquire)));
    KVCacheEvent event;
    while (batch.size() < max_count && tryDequeue(event)) {
        batch.push_back(std::move(event));
    }
    if (!batch.empty()) {
        return batch;
    }

    // A producer reserves capacity before publishing its ring cell. Sleep on
    // the coalesced semaphore rather than spinning on reserved occupancy. A
    // signal is posted only after the cell is published, and semaphore state
    // persists when publication races this wait, so notifications cannot be
    // lost between the empty drain above and blocking here.
    // stop() is terminal. Keep every post-stop call non-blocking as well as
    // waking a consumer that was already asleep when shutdown began.
    if (stopped() || timeout <= std::chrono::milliseconds::zero() || !waitForSignal(timeout)) {
        return batch;
    }
    while (batch.size() < max_count && tryDequeue(event)) {
        batch.push_back(std::move(event));
    }
    return batch;
}

void KVCacheEventQueue::stop() noexcept {
    producer_gate_.close();
    signalConsumer();
}

void KVCacheEventQueue::quiescePushes() noexcept {
    // Producers are non-blocking and only stay active long enough to reserve
    // and publish one ring cell. Waiting here closes the reservation window so
    // the queue cannot be destroyed while a producer still owns that cell.
    producer_gate_.quiesce();
}

void KVCacheEventQueue::discardAvailable() noexcept {
    KVCacheEvent ignored;
    while (tryDequeue(ignored)) {}
}

void KVCacheEventQueue::discardPending() noexcept {
    stop();
    quiescePushes();
    discardAvailable();
}

size_t KVCacheEventQueue::size() const noexcept {
    return size_.load(std::memory_order_relaxed);
}

size_t KVCacheEventQueue::highWatermark() const noexcept {
    return high_watermark_.load(std::memory_order_relaxed);
}

void KVCacheEventQueue::signalConsumer() noexcept {
    if (!signal_pending_.exchange(true, std::memory_order_acq_rel)) {
        if (sem_post(&consumer_signal_) != 0) {
            // EOVERFLOW is impossible for the binary protocol unless the
            // semaphore itself is corrupted. Let a later publication retry
            // instead of leaving the queue permanently marked as signalled.
            signal_pending_.store(false, std::memory_order_release);
        }
    }
}

bool KVCacheEventQueue::waitForSignal(std::chrono::milliseconds timeout) noexcept {
    timespec deadline{};
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        return false;
    }
    const auto timeout_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout);
    deadline.tv_sec += static_cast<time_t>(timeout_ns.count() / 1000000000LL);
    deadline.tv_nsec += static_cast<long>(timeout_ns.count() % 1000000000LL);
    if (deadline.tv_nsec >= 1000000000L) {
        ++deadline.tv_sec;
        deadline.tv_nsec -= 1000000000L;
    }

    int result;
    do {
        result = sem_timedwait(&consumer_signal_, &deadline);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
        return false;
    }
    // This RMW reads the last producer exchange in the atomic modification
    // order. Its acquire half therefore observes any event whose producer saw
    // an already-pending signal and skipped sem_post(); a later producer sees
    // false and posts the next token. The handoff closes the only coalescing
    // race without a mutex or a producer-side wait.
    signal_pending_.exchange(false, std::memory_order_acq_rel);
    return true;
}

bool KVCacheEventQueue::stopped() const noexcept {
    return producer_gate_.closed();
}

void KVCacheEventQueue::updateHighWatermark(size_t size) noexcept {
    auto observed = high_watermark_.load(std::memory_order_relaxed);
    while (observed < size
           && !high_watermark_.compare_exchange_weak(
               observed, size, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

void KVCacheEventQueue::enqueue(KVCacheEvent event) noexcept {
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    Cell*  cell;
    for (;;) {
        cell             = &cells_[pos & ring_mask_];
        const size_t seq = cell->sequence.load(std::memory_order_acquire);
        if (seq == pos) {
            if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                break;
            }
        } else {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }
    // The queue position is the publication order. Assigning sequence here,
    // after the position is reserved, keeps sequences monotonic even when
    // multiple producers enter tryPush concurrently.
    event.sequence = static_cast<uint64_t>(pos + 1);
    cell->event    = std::move(event);
    published_size_.fetch_add(1, std::memory_order_relaxed);
    cell->sequence.store(pos + 1, std::memory_order_release);
}

bool KVCacheEventQueue::tryDequeue(KVCacheEvent& event) noexcept {
    // This is an MPSC queue: only the publisher worker calls tryDequeue(). A
    // cell is ready exactly when its producer has advanced the sequence to
    // pos + 1. Equality is sufficient and remains well-defined when size_t
    // wraps; signed sequence subtraction would make that boundary dependent
    // on implementation-defined integer conversion.
    const size_t pos  = dequeue_pos_.load(std::memory_order_relaxed);
    Cell*        cell = &cells_[pos & ring_mask_];
    if (cell->sequence.load(std::memory_order_acquire) != pos + 1) {
        return false;
    }
    dequeue_pos_.store(pos + 1, std::memory_order_relaxed);
    event = std::move(cell->event);
    cell->sequence.store(pos + ring_capacity_, std::memory_order_release);
    published_size_.fetch_sub(1, std::memory_order_release);
    size_.fetch_sub(1, std::memory_order_release);
    return true;
}

PublishResult toPublishResult(QueuePushResult result) noexcept {
    switch (result) {
        case QueuePushResult::ACCEPTED:
            return PublishResult::ACCEPTED;
        case QueuePushResult::STOPPED:
            return PublishResult::NOT_RUNNING;
        case QueuePushResult::FULL:
            return PublishResult::QUEUE_FULL;
    }
    return PublishResult::NOT_RUNNING;
}

}  // namespace rtp_llm::detail
