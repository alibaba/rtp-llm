#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <semaphore.h>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEventAdmissionGate.h"
#include "rtp_llm/cpp/cache/events/KVCacheEvent.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"

namespace rtp_llm::detail {

enum class QueuePushResult {
    ACCEPTED,
    STOPPED,
    FULL,
};

class KVCacheEventQueue {
public:
    explicit KVCacheEventQueue(size_t capacity);
    ~KVCacheEventQueue();

    // ACCEPTED pushes assign and consume the next monotonically increasing
    // event.sequence. FULL/STOPPED pushes and stop() do not consume or reset
    // the sequence within this queue instance. The queue supports multiple
    // producers and one consumer. Destruction still requires external lifetime
    // ownership, while stop() atomically closes admission and quiescePushes()
    // fences every producer admitted before that close.
    QueuePushResult           tryPush(KVCacheEvent event) noexcept;
    std::vector<KVCacheEvent> waitPop(size_t max_batch_size, std::chrono::milliseconds timeout);
    void                      stop() noexcept;
    // Lifecycle teardown only: wait until calls admitted before stop() have
    // left the queue. New calls are rejected by the stopped bit in the same
    // atomic state, so they cannot race past this fence.
    void quiescePushes() noexcept;
    // Sole-consumer recovery helper. External admission must already be
    // paused and quiesced so no producer can reserve a cell concurrently.
    // Unlike discardPending(), the queue remains reusable afterwards.
    void discardAvailable() noexcept;
    // Sole-consumer teardown helper. Closes admission, waits for every
    // reserved cell to be published, and releases all pending events while
    // preserving lifetime counters such as the high-water mark.
    void   discardPending() noexcept;
    size_t size() const noexcept;
    // Lifetime maximum reserved occupancy. It is diagnostic-only and never
    // resets on reads, so status collection cannot perturb queue behavior.
    size_t highWatermark() const noexcept;

private:
    struct Cell {
        std::atomic<size_t> sequence{0};
        KVCacheEvent        event;
    };

    void enqueue(KVCacheEvent event) noexcept;
    bool tryDequeue(KVCacheEvent& event) noexcept;
    void updateHighWatermark(size_t size) noexcept;
    void signalConsumer() noexcept;
    bool waitForSignal(std::chrono::milliseconds timeout) noexcept;
    bool stopped() const noexcept;

private:
    const size_t              capacity_;
    const size_t              ring_capacity_;
    const size_t              ring_mask_;
    std::unique_ptr<Cell[]>   cells_;
    std::atomic<size_t>       enqueue_pos_{0};
    std::atomic<size_t>       dequeue_pos_{0};
    std::atomic<size_t>       size_{0};
    std::atomic<size_t>       high_watermark_{0};
    std::atomic<size_t>       published_size_{0};
    KVCacheEventAdmissionGate producer_gate_;
    // A binary, coalesced semaphore keeps notifications lossless without
    // taking a mutex or performing an unbounded wait on a producer thread.
    // Keep it last among resources that need cleanup: if sem_init() throws,
    // the already-constructed C++ members unwind normally and no semaphore
    // destruction is required.
    sem_t             consumer_signal_{};
    std::atomic<bool> signal_pending_{false};
};

PublishResult toPublishResult(QueuePushResult result) noexcept;

}  // namespace rtp_llm::detail
