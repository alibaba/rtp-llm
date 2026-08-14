#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"

#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm::detail {
namespace {

TEST(KVCacheEventQueueTest, ConcurrentProducersCommitMonotonicSequence) {
    constexpr size_t kProducerCount     = 8;
    constexpr size_t kEventsPerProducer = 2000;
    constexpr size_t kEventCount        = kProducerCount * kEventsPerProducer;

    KVCacheEventQueue        queue(kEventCount);
    std::atomic<bool>        start{false};
    std::atomic<size_t>      accepted{0};
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
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (received.size() < kEventCount && std::chrono::steady_clock::now() < deadline) {
        auto batch = queue.waitPop(256, std::chrono::milliseconds(100));
        received.insert(received.end(), batch.begin(), batch.end());
    }

    ASSERT_EQ(kEventCount, received.size())
        << "queue drain timed out after receiving " << received.size() << " of " << kEventCount << " events";
    for (size_t i = 0; i < received.size(); ++i) {
        EXPECT_EQ(i + 1, received[i].sequence);
    }
    EXPECT_EQ(0, queue.size());
}

TEST(KVCacheEventQueueTest, CapacityDrainAndStopHaveExplicitResults) {
    KVCacheEventQueue queue(2);
    EXPECT_EQ(0u, queue.highWatermark());
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 20, 0}));
    EXPECT_EQ(QueuePushResult::FULL, queue.tryPush({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_EQ(2u, queue.highWatermark());

    auto initial = queue.waitPop(2, std::chrono::milliseconds(10));
    ASSERT_EQ(2u, initial.size());
    EXPECT_EQ(0u, queue.size());
    EXPECT_EQ(2u, queue.highWatermark());
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 40, 0}));
    auto batch = queue.waitPop(2, std::chrono::milliseconds(10));
    ASSERT_EQ(1, batch.size());
    EXPECT_EQ(3, batch[0].sequence);
    EXPECT_EQ(40, batch[0].block_key);
    EXPECT_EQ(2u, queue.highWatermark());

    queue.stop();
    queue.quiescePushes();
    EXPECT_EQ(QueuePushResult::STOPPED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 50, 0}));
}

TEST(KVCacheEventQueueTest, DiscardPendingClearsTerminalBacklogAndPreservesHighWatermark) {
    KVCacheEventQueue queue(3);
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_EQ(2u, queue.size());

    queue.discardPending();

    EXPECT_EQ(0u, queue.size());
    EXPECT_EQ(2u, queue.highWatermark());
    EXPECT_EQ(QueuePushResult::STOPPED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_TRUE(queue.waitPop(3, std::chrono::seconds(1)).empty());
}

TEST(KVCacheEventQueueTest, DiscardAvailableKeepsQueueReusableForSnapshotRecovery) {
    KVCacheEventQueue queue(2);
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 20, 0}));

    queue.discardAvailable();

    EXPECT_EQ(0u, queue.size());
    EXPECT_EQ(2u, queue.highWatermark());
    EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 30, 0}));
    const auto recovered = queue.waitPop(1, std::chrono::milliseconds(10));
    ASSERT_EQ(1u, recovered.size());
    EXPECT_EQ(3u, recovered.front().sequence);
    EXPECT_EQ(30, recovered.front().block_key);
}

TEST(KVCacheEventQueueTest, ZeroAndOneCapacityClampToOneUsableSlot) {
    for (const size_t configured_capacity : {size_t{0}, size_t{1}}) {
        SCOPED_TRACE(configured_capacity);
        KVCacheEventQueue queue(configured_capacity);
        EXPECT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 10, 0}));
        EXPECT_EQ(QueuePushResult::FULL, queue.tryPush({KVCacheEventType::BLOCK_ADD, 20, 0}));

        auto batch = queue.waitPop(1, std::chrono::milliseconds(10));
        ASSERT_EQ(1u, batch.size());
        EXPECT_EQ(1u, batch.front().sequence);
        EXPECT_EQ(10, batch.front().block_key);
        EXPECT_EQ(0u, queue.size());
    }
}

TEST(KVCacheEventQueueTest, OversizedCapacityFailsBeforeRingAllocation) {
    EXPECT_THROW(KVCacheEventQueue(kKVCacheEventMaxQueueCapacity + 1), std::length_error);
    EXPECT_THROW(KVCacheEventQueue(std::numeric_limits<size_t>::max()), std::length_error);
}

TEST(KVCacheEventQueueTest, NonPowerOfTwoCapacityKeepsExactLimitAcrossWraps) {
    KVCacheEventQueue queue(3);
    uint64_t          expected_sequence = 1;
    for (size_t cycle = 0; cycle < 100; ++cycle) {
        for (int64_t key = 0; key < 3; ++key) {
            EXPECT_EQ(QueuePushResult::ACCEPTED,
                      queue.tryPush({KVCacheEventType::BLOCK_ADD, static_cast<int64_t>(cycle * 3) + key, 0}));
        }
        EXPECT_EQ(QueuePushResult::FULL, queue.tryPush({KVCacheEventType::BLOCK_ADD, -1, 0}));
        const auto batch = queue.waitPop(3, std::chrono::milliseconds::zero());
        ASSERT_EQ(3u, batch.size());
        for (const auto& event : batch) {
            EXPECT_EQ(expected_sequence++, event.sequence);
        }
    }
    EXPECT_EQ(0u, queue.size());
    EXPECT_EQ(3u, queue.highWatermark());
}

TEST(KVCacheEventQueueTest, PublishedEventWakesWaitingConsumer) {
    KVCacheEventQueue queue(2);
    auto              waiter =
        std::async(std::launch::async, [&] { return queue.waitPop(/*max_batch_size=*/1, std::chrono::seconds(2)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto published_at = std::chrono::steady_clock::now();
    ASSERT_EQ(QueuePushResult::ACCEPTED, queue.tryPush({KVCacheEventType::BLOCK_ADD, 42, 0}));

    ASSERT_EQ(std::future_status::ready, waiter.wait_for(std::chrono::seconds(1)));
    const auto batch = waiter.get();
    ASSERT_EQ(1u, batch.size());
    EXPECT_EQ(42, batch.front().block_key);
    EXPECT_LT(std::chrono::steady_clock::now() - published_at, std::chrono::seconds(1));
}

TEST(KVCacheEventQueueTest, RepeatedEmptyToNonemptyHandoffsNeverLoseWakeups) {
    KVCacheEventQueue queue(3);
    constexpr size_t  kEventCount = 10000;

    auto consumer = std::async(std::launch::async, [&] {
        std::vector<KVCacheEvent> received;
        received.reserve(kEventCount);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (received.size() < kEventCount && std::chrono::steady_clock::now() < deadline) {
            auto batch = queue.waitPop(/*max_batch_size=*/1, std::chrono::milliseconds(100));
            received.insert(received.end(), batch.begin(), batch.end());
        }
        return received;
    });

    for (size_t key = 0; key < kEventCount; ++key) {
        while (queue.tryPush({KVCacheEventType::BLOCK_ADD, static_cast<int64_t>(key), 0}) == QueuePushResult::FULL) {
            std::this_thread::yield();
        }
        // Encourage every semaphore-token handoff ordering, including a
        // producer arriving while the consumer clears the coalescing bit.
        if ((key & 7) == 0) {
            std::this_thread::yield();
        }
    }

    ASSERT_EQ(std::future_status::ready, consumer.wait_for(std::chrono::seconds(10)));
    const auto received = consumer.get();
    ASSERT_EQ(kEventCount, received.size());
    for (size_t i = 0; i < received.size(); ++i) {
        EXPECT_EQ(i + 1, received[i].sequence);
        EXPECT_EQ(static_cast<int64_t>(i), received[i].block_key);
    }
}

TEST(KVCacheEventQueueTest, AdmissionGateCloseAndQuiesceWaitForAdmittedCall) {
    KVCacheEventAdmissionGate gate;
    std::atomic<bool>         admitted{false};
    std::promise<void>        release;
    auto                      release_future = release.get_future().share();
    std::thread               producer([&] {
        auto guard = gate.tryEnter();
        admitted.store(static_cast<bool>(guard), std::memory_order_release);
        release_future.wait();
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!admitted.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    // Keep the cleanup path alive even if scheduling unexpectedly misses the
    // deadline; an ASSERT here would destroy a joinable thread.
    EXPECT_TRUE(admitted.load(std::memory_order_acquire));

    gate.close();
    auto quiescer = std::async(std::launch::async, [&] { gate.quiesce(); });
    EXPECT_EQ(std::future_status::timeout, quiescer.wait_for(std::chrono::milliseconds(10)));
    EXPECT_FALSE(gate.tryEnter());

    release.set_value();
    producer.join();
    EXPECT_EQ(std::future_status::ready, quiescer.wait_for(std::chrono::seconds(1)));

    ASSERT_TRUE(gate.reopenAfterQuiesce());
    EXPECT_FALSE(gate.reopenAfterQuiesce());
    {
        auto next_epoch = gate.tryEnter();
        EXPECT_TRUE(next_epoch);
    }
    gate.close();
    gate.quiesce();
}

TEST(KVCacheEventQueueTest, StopInterruptsEmptyConsumerWait) {
    KVCacheEventQueue queue(2);
    auto              waiter =
        std::async(std::launch::async, [&] { return queue.waitPop(/*max_batch_size=*/1, std::chrono::seconds(2)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    queue.stop();
    queue.quiescePushes();
    ASSERT_EQ(std::future_status::ready, waiter.wait_for(std::chrono::seconds(1)));
    EXPECT_TRUE(waiter.get().empty());

    // The wake token is coalesced, but stop is a terminal state: repeated
    // callers must not consume one token and then sleep until their timeout.
    for (size_t i = 0; i < 2; ++i) {
        auto stopped_waiter = std::async(std::launch::async,
                                         [&] { return queue.waitPop(/*max_batch_size=*/1, std::chrono::seconds(2)); });
        ASSERT_EQ(std::future_status::ready, stopped_waiter.wait_for(std::chrono::seconds(1)));
        EXPECT_TRUE(stopped_waiter.get().empty());
    }
}

TEST(KVCacheEventQueueTest, ConcurrentStopRejectsNewPushesAndQuiescesAdmittedProducers) {
    constexpr size_t kProducerCount = 8;
    constexpr size_t kCapacity      = 65536;

    KVCacheEventQueue        queue(kCapacity);
    std::atomic<bool>        start{false};
    std::atomic<size_t>      ready{0};
    std::atomic<size_t>      accepted{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_id = 0; producer_id < kProducerCount; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            size_t ordinal = 0;
            for (;;) {
                const auto key    = static_cast<int64_t>(producer_id * kCapacity + ordinal++);
                const auto result = queue.tryPush({KVCacheEventType::BLOCK_ADD, key, 0});
                if (result == QueuePushResult::ACCEPTED) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                } else if (result == QueuePushResult::STOPPED) {
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducerCount) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (accepted.load(std::memory_order_relaxed) < 1000 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool produced_enough = accepted.load(std::memory_order_relaxed) >= 1000;

    queue.stop();
    queue.quiescePushes();
    for (auto& producer : producers) {
        producer.join();
    }
    ASSERT_TRUE(produced_enough);
    EXPECT_EQ(QueuePushResult::STOPPED, queue.tryPush({KVCacheEventType::BLOCK_ADD, -1, 0}));

    std::vector<KVCacheEvent> received;
    received.reserve(accepted.load(std::memory_order_relaxed));
    for (;;) {
        auto batch = queue.waitPop(256, std::chrono::milliseconds::zero());
        if (batch.empty()) {
            break;
        }
        received.insert(received.end(), batch.begin(), batch.end());
    }
    ASSERT_EQ(accepted.load(std::memory_order_relaxed), received.size());
    EXPECT_EQ(0u, queue.size());
    for (size_t i = 0; i < received.size(); ++i) {
        EXPECT_EQ(i + 1, received[i].sequence);
    }
}

TEST(KVCacheEventQueueTest, SmallRingWrapsUnderConcurrentProducerConsumerLoad) {
    constexpr size_t kProducerCount     = 4;
    constexpr size_t kEventsPerProducer = 5000;
    constexpr size_t kEventCount        = kProducerCount * kEventsPerProducer;

    KVCacheEventQueue        queue(64);
    std::atomic<bool>        start{false};
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
    bool       timed_out = false;
    const auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (received.size() < kEventCount) {
        if (std::chrono::steady_clock::now() >= deadline) {
            timed_out = true;
            queue.stop();
            break;
        }
        auto batch = queue.waitPop(32, std::chrono::milliseconds(100));
        received.insert(received.end(), batch.begin(), batch.end());
    }
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_FALSE(timed_out) << "queue drain timed out after receiving " << received.size() << " of " << kEventCount
                            << " events";
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
