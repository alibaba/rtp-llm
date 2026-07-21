#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace rtp_llm::detail {
namespace {

TEST(KVCacheEventQueueTest, ConcurrentProducersCommitMonotonicSequence) {
    constexpr size_t kProducerCount     = 8;
    constexpr size_t kEventsPerProducer = 2000;
    constexpr size_t kEventCount        = kProducerCount * kEventsPerProducer;

    KVCacheEventQueue   queue(kEventCount);
    std::atomic<bool>   start{false};
    std::atomic<size_t> accepted{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_id = 0; producer_id < kProducerCount; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < kEventsPerProducer; ++i) {
                const auto key = static_cast<int64_t>(producer_id * kEventsPerProducer + i);
                if (queue.tryPush({KVCacheEventType::BLOCK_ADD, key, 0}) == QueuePushResult::ACCEPTED) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& producer : producers) {
        producer.join();
    }
    ASSERT_EQ(kEventCount, accepted.load(std::memory_order_relaxed));
    ASSERT_EQ(kEventCount, queue.size());

    std::vector<KVCacheEvent> received;
    received.reserve(kEventCount);
    while (received.size() < kEventCount) {
        auto batch = queue.waitPop(256, std::chrono::milliseconds(100));
        received.insert(received.end(), batch.begin(), batch.end());
    }

    ASSERT_EQ(kEventCount, received.size());
    for (size_t i = 0; i < received.size(); ++i) {
        EXPECT_EQ(i + 1, received[i].sequence);
    }
    EXPECT_EQ(0, queue.size());
}

TEST(KVCacheEventQueueTest, CapacityDiscardAndStopHaveExplicitResults) {
    KVCacheEventQueue queue(2);
    EXPECT_EQ(QueuePushResult::ACCEPTED,
              queue.tryPush({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_EQ(QueuePushResult::ACCEPTED,
              queue.tryPush({KVCacheEventType::BLOCK_ADD, 20, 0}));
    EXPECT_EQ(QueuePushResult::FULL,
              queue.tryPush({KVCacheEventType::BLOCK_ADD, 30, 0}));

    queue.discardPending();
    EXPECT_EQ(0, queue.size());
    EXPECT_EQ(QueuePushResult::ACCEPTED,
              queue.tryPush({KVCacheEventType::BLOCK_ADD, 40, 0}));
    auto batch = queue.waitPop(2, std::chrono::milliseconds(10));
    ASSERT_EQ(1, batch.size());
    EXPECT_EQ(3, batch[0].sequence);
    EXPECT_EQ(40, batch[0].block_key);

    queue.stop();
    EXPECT_EQ(QueuePushResult::STOPPED,
              queue.tryPush({KVCacheEventType::BLOCK_ADD, 50, 0}));
}

TEST(KVCacheEventQueueTest, SmallRingWrapsUnderConcurrentProducerConsumerLoad) {
    constexpr size_t kProducerCount     = 4;
    constexpr size_t kEventsPerProducer = 5000;
    constexpr size_t kEventCount        = kProducerCount * kEventsPerProducer;

    KVCacheEventQueue queue(64);
    std::atomic<bool> start{false};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_id = 0; producer_id < kProducerCount; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < kEventsPerProducer; ++i) {
                const auto key = static_cast<int64_t>(producer_id * kEventsPerProducer + i);
                while (queue.tryPush({KVCacheEventType::BLOCK_ADD, key, 0}) == QueuePushResult::FULL) {
                    std::this_thread::yield();
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    std::vector<KVCacheEvent> received;
    received.reserve(kEventCount);
    while (received.size() < kEventCount) {
        auto batch = queue.waitPop(32, std::chrono::milliseconds(100));
        received.insert(received.end(), batch.begin(), batch.end());
    }
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_EQ(kEventCount, received.size());
    std::vector<bool> seen(kEventCount, false);
    for (size_t i = 0; i < received.size(); ++i) {
        EXPECT_EQ(i + 1, received[i].sequence);
        ASSERT_GE(received[i].block_key, 0);
        ASSERT_LT(static_cast<size_t>(received[i].block_key), seen.size());
        EXPECT_FALSE(seen[static_cast<size_t>(received[i].block_key)]);
        seen[static_cast<size_t>(received[i].block_key)] = true;
    }
    EXPECT_EQ(0, queue.size());
}

}  // namespace
}  // namespace rtp_llm::detail
