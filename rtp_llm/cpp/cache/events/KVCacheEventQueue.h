#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

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

    QueuePushResult          tryPush(KVCacheEvent event) noexcept;
    std::vector<KVCacheEvent> waitPop(size_t max_batch_size, std::chrono::milliseconds timeout);
    void                     waitForStop(std::chrono::milliseconds timeout);
    void                     discardPending();
    void                     wake();
    void                     stop();
    size_t                   size() const noexcept;

private:
    struct Cell {
        std::atomic<size_t> sequence{0};
        KVCacheEvent       event;
    };

    void enqueue(KVCacheEvent event) noexcept;
    bool tryDequeue(KVCacheEvent& event) noexcept;

private:
    const size_t            capacity_;
    const size_t            ring_capacity_;
    std::unique_ptr<Cell[]> cells_;
    std::atomic<size_t>     enqueue_pos_{0};
    std::atomic<size_t>     dequeue_pos_{0};
    std::atomic<size_t>     size_{0};
    std::atomic<size_t>     published_size_{0};
    std::atomic<bool>       stopped_{false};
    mutable std::mutex      wait_mu_;
    std::condition_variable cv_;
};

PublishResult toPublishResult(QueuePushResult result) noexcept;

}  // namespace rtp_llm::detail
