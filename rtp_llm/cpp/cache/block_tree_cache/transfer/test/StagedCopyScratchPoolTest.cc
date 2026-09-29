#include "rtp_llm/cpp/cache/block_tree_cache/transfer/StagedCopyScratchPool.h"

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

using Pool = StagedCopyScratchPool;
constexpr Pool::Limits kLimits{1024, 8};

static_assert(!std::is_copy_constructible_v<Pool>);
static_assert(!std::is_copy_constructible_v<Pool::Lease>);
static_assert(std::is_nothrow_move_constructible_v<Pool::Lease>);

TEST(StagedCopyScratchPoolTest, FixedCapacityAcrossConcurrentCallers) {
    for (size_t capacity : {1u, 4u}) {
        Pool pool(capacity, {0, 0, 1}, kLimits);
        std::mutex mutex;
        std::condition_variable cv;
        size_t holding = 0;
        bool release = false;
        std::vector<std::thread> workers;
        for (size_t i = 0; i < capacity; ++i) {
            workers.emplace_back([&] {
                auto result = pool.tryAcquire();
                if (result.status != Pool::AcquireStatus::ACQUIRED) {
                    return;
                }
                EXPECT_EQ(result.lease->scratchFor(0).device_index, -1);
                std::unique_lock<std::mutex> lock(mutex);
                ++holding;
                cv.notify_all();
                cv.wait(lock, [&] { return release; });
            });
        }
        bool all_holding = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            all_holding = cv.wait_for(lock, std::chrono::seconds(5), [&] { return holding == capacity; });
        }
        if (all_holding) {
            EXPECT_EQ(pool.tryAcquire().status, Pool::AcquireStatus::EXHAUSTED);
            const auto stats = pool.stats();
            EXPECT_EQ(stats.capacity, capacity);
            EXPECT_EQ(stats.in_use, capacity);
            EXPECT_EQ(stats.peak_in_use, capacity);
            EXPECT_EQ(stats.acquire_misses, 1u);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
        }
        cv.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
        ASSERT_TRUE(all_holding);
        EXPECT_EQ(pool.stats().in_use, 0u);
        EXPECT_EQ(pool.tryAcquire().status, Pool::AcquireStatus::ACQUIRED);
    }
}

TEST(StagedCopyScratchPoolTest, MoveExceptionAndNestedAcquireReturnExactlyOnce) {
    Pool pool(2, {0}, kLimits);
    {
        auto first = pool.tryAcquire();
        auto second = pool.tryAcquire();
        ASSERT_TRUE(first.lease);
        ASSERT_TRUE(second.lease);
        EXPECT_EQ(pool.tryAcquire().status, Pool::AcquireStatus::EXHAUSTED);
        *first.lease = std::move(*second.lease);
        EXPECT_EQ(pool.stats().in_use, 1u);
        EXPECT_EQ(pool.tryAcquire().status, Pool::AcquireStatus::ACQUIRED);
    }
    EXPECT_EQ(pool.stats().in_use, 0u);
    try {
        auto lease = pool.tryAcquire();
        ASSERT_TRUE(lease.lease);
        throw std::runtime_error("test unwind");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(pool.stats().in_use, 0u);
}

TEST(StagedCopyScratchPoolTest, QuarantineDisablesFurtherAcquire) {
    Pool pool(2, {0}, kLimits);
    {
        auto first = pool.tryAcquire();
        auto second = pool.tryAcquire();
        ASSERT_TRUE(first.lease);
        ASSERT_TRUE(second.lease);
        first.lease->quarantine();
    }
    const auto stats = pool.stats();
    EXPECT_TRUE(stats.disabled);
    EXPECT_EQ(stats.quarantined, 1u);
    EXPECT_EQ(stats.in_use, 0u);
    EXPECT_EQ(pool.tryAcquire().status, Pool::AcquireStatus::DISABLED);
}

TEST(StagedCopyScratchPoolTest, RejectsInvalidConstructionAndUnknownDevice) {
    EXPECT_THROW((Pool{0, {0}, kLimits}), std::invalid_argument);
    EXPECT_THROW((Pool{1, {-1}, kLimits}), std::invalid_argument);
    EXPECT_THROW((Pool{1, {0}, {0, 1}}), std::invalid_argument);
    EXPECT_THROW((Pool{1, {0}, {1, 0}}), std::invalid_argument);
    Pool pool(1, {1, 1}, kLimits);
    EXPECT_TRUE(pool.allowsDevice(1));
    EXPECT_FALSE(pool.allowsDevice(0));
    auto acquired = pool.tryAcquire();
    ASSERT_TRUE(acquired.lease);
    EXPECT_THROW(acquired.lease->scratchFor(0), std::out_of_range);
}

}  // namespace
}  // namespace rtp_llm
