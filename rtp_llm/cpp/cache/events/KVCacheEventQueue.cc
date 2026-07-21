#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace rtp_llm::detail {

KVCacheEventQueue::KVCacheEventQueue(size_t capacity):
    capacity_(std::max<size_t>(capacity, 1)),
    ring_capacity_(std::max<size_t>(capacity_, 2)),
    cells_(std::make_unique<Cell[]>(ring_capacity_)) {
    for (size_t i = 0; i < ring_capacity_; ++i) {
        cells_[i].sequence.store(i, std::memory_order_relaxed);
    }
}

QueuePushResult KVCacheEventQueue::tryPush(KVCacheEvent event) noexcept {
    if (stopped_.load(std::memory_order_acquire)) {
        return QueuePushResult::STOPPED;
    }

    size_t current_size = size_.load(std::memory_order_relaxed);
    do {
        if (current_size >= capacity_) {
            return QueuePushResult::FULL;
        }
    } while (!size_.compare_exchange_weak(
        current_size, current_size + 1, std::memory_order_acq_rel, std::memory_order_relaxed));

    if (stopped_.load(std::memory_order_acquire)) {
        size_.fetch_sub(1, std::memory_order_release);
        return QueuePushResult::STOPPED;
    }

    enqueue(std::move(event));
    cv_.notify_one();
    return QueuePushResult::ACCEPTED;
}

std::vector<KVCacheEvent>
KVCacheEventQueue::waitPop(size_t max_batch_size, std::chrono::milliseconds timeout) {
    const size_t max_count = std::max<size_t>(max_batch_size, 1);
    if (published_size_.load(std::memory_order_acquire) == 0) {
        std::unique_lock<std::mutex> lock(wait_mu_);
        cv_.wait_for(lock, timeout, [this] {
            return stopped_.load(std::memory_order_acquire)
                   || published_size_.load(std::memory_order_acquire) > 0;
        });
    }

    std::vector<KVCacheEvent> batch;
    batch.reserve(std::min(max_count, published_size_.load(std::memory_order_acquire)));
    KVCacheEvent event;
    while (batch.size() < max_count && tryDequeue(event)) {
        batch.push_back(std::move(event));
    }
    return batch;
}

void KVCacheEventQueue::waitForStop(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(wait_mu_);
    cv_.wait_for(lock, timeout, [this] { return stopped_.load(std::memory_order_acquire); });
}

void KVCacheEventQueue::discardPending() {
    // Drain only items fully published at this boundary. Concurrently
    // published events remain queued and are applied after the snapshot ACK.
    const size_t target = published_size_.load(std::memory_order_acquire);
    KVCacheEvent event;
    for (size_t i = 0; i < target && tryDequeue(event); ++i) {
    }
}

void KVCacheEventQueue::wake() {
    cv_.notify_one();
}

void KVCacheEventQueue::stop() {
    stopped_.store(true, std::memory_order_release);
    cv_.notify_all();
}

size_t KVCacheEventQueue::size() const noexcept {
    return size_.load(std::memory_order_relaxed);
}

void KVCacheEventQueue::enqueue(KVCacheEvent event) noexcept {
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    Cell*  cell;
    for (;;) {
        cell             = &cells_[pos % ring_capacity_];
        const size_t seq = cell->sequence.load(std::memory_order_acquire);
        const auto   dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
        if (dif == 0) {
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
    cell->event = std::move(event);
    published_size_.fetch_add(1, std::memory_order_relaxed);
    cell->sequence.store(pos + 1, std::memory_order_release);
}

bool KVCacheEventQueue::tryDequeue(KVCacheEvent& event) noexcept {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    Cell*  cell;
    for (;;) {
        cell             = &cells_[pos % ring_capacity_];
        const size_t seq = cell->sequence.load(std::memory_order_acquire);
        const auto   dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (dif == 0) {
            if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                break;
            }
        } else if (dif < 0) {
            return false;
        } else {
            pos = dequeue_pos_.load(std::memory_order_relaxed);
        }
    }
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
