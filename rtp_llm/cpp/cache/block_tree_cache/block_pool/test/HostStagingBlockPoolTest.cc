#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(HostStagingBlockPoolTest, UsesCallerProvidedStride) {
    HostStagingBlockPool pool(1, 65);

    auto leases = pool.tryMallocBatch(1);
    ASSERT_TRUE(leases.has_value());
    const auto view = (*leases)[0].blockBuffer(65);
    EXPECT_EQ(view.capacity_bytes, 65u);
}

TEST(HostStagingBlockPoolTest, PinnedBackingServesLeases) {
    HostStagingBlockPool pool(1, 4096);

    auto lease = pool.tryMallocBatch(1);
    ASSERT_TRUE(lease.has_value());
    const auto view = (*lease)[0].blockBuffer(64);
    ASSERT_NE(view.base, nullptr);
    const auto tensor = torch::from_blob(view.base, {64}, torch::TensorOptions().dtype(torch::kUInt8));
    EXPECT_TRUE(tensor.is_pinned());
    EXPECT_EQ(view.payload_bytes, 64u);
    EXPECT_EQ(view.capacity_bytes, 4096u);
    std::memset(view.base, 0xAB, view.payload_bytes);

    EXPECT_FALSE(pool.tryMallocBatch(1).has_value());
    lease.reset();
    EXPECT_TRUE(pool.tryMallocBatch(1).has_value());
}

TEST(HostStagingBlockPoolTest, BatchAllocationIsAtomic) {
    HostStagingBlockPool pool(2, 4096);

    EXPECT_FALSE(pool.tryMallocBatch(3).has_value());
    auto leases = pool.tryMallocBatch(2);
    ASSERT_TRUE(leases.has_value());
    EXPECT_EQ(leases->size(), 2u);
    EXPECT_FALSE(pool.tryMallocBatch(1).has_value());
}

TEST(HostStagingBlockPoolTest, BatchWaitersAreStrictFifo) {
    HostStagingBlockPool pool(2, 4096);
    auto                 held = pool.tryMallocBatch(1);
    ASSERT_TRUE(held.has_value());

    std::vector<int>                                           callback_order;
    std::optional<HostStagingBlockPool::HostStagingBlockBatch> first_leases;
    std::optional<HostStagingBlockPool::HostStagingBlockBatch> second_leases;
    const auto deadline = HostStagingBlockPool::Clock::now() + std::chrono::seconds(1);
    pool.requestBatch(2, deadline, [&](auto result) {
        ASSERT_TRUE(result.has_value());
        callback_order.push_back(2);
        first_leases.emplace(std::move(*result));
    });
    pool.requestBatch(1, deadline, [&](auto result) {
        ASSERT_TRUE(result.has_value());
        callback_order.push_back(1);
        second_leases.emplace(std::move(*result));
    });

    EXPECT_TRUE(callback_order.empty());
    held.reset();
    ASSERT_EQ(callback_order, std::vector<int>({2}));
    ASSERT_TRUE(first_leases.has_value());
    EXPECT_FALSE(second_leases.has_value());

    first_leases.reset();
    EXPECT_EQ(callback_order, std::vector<int>({2, 1}));
    EXPECT_TRUE(second_leases.has_value());
}

TEST(HostStagingBlockPoolTest, BatchWaiterRemainsPendingUntilRelease) {
    HostStagingBlockPool pool(1, 4096);
    auto                 held = pool.tryMallocBatch(1);
    ASSERT_TRUE(held.has_value());

    std::promise<std::optional<HostStagingBlockPool::HostStagingBlockBatch>> result;
    auto                                                                     future = result.get_future();
    pool.requestBatch(1, HostStagingBlockPool::Clock::now() + std::chrono::seconds(1), [&result](auto leases) {
        result.set_value(std::move(leases));
    });

    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    held.reset();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(future.get().has_value());
}

TEST(HostStagingBlockPoolTest, ExpiredBatchWaiterIsDiscardedWhenStagingBecomesAvailable) {
    HostStagingBlockPool pool(1, 4096);
    auto                 held = pool.tryMallocBatch(1);
    ASSERT_TRUE(held.has_value());

    std::promise<std::optional<HostStagingBlockPool::HostStagingBlockBatch>> result;
    auto                                                                     future = result.get_future();
    pool.requestBatch(1, HostStagingBlockPool::Clock::now() + std::chrono::milliseconds(1), [&result](auto leases) {
        result.set_value(std::move(leases));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    held.reset();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(future.get().has_value());
    EXPECT_TRUE(pool.tryMallocBatch(1).has_value());
}

TEST(HostStagingBlockPoolTest, CancelAllCompletesEveryPendingWaiterExactlyOnce) {
    HostStagingBlockPool pool(1, 4096);
    auto                 held = pool.tryMallocBatch(1);
    ASSERT_TRUE(held.has_value());

    std::atomic<size_t> callback_count{0};
    std::atomic<size_t> cancelled_count{0};
    for (size_t waiter = 0; waiter < 2; ++waiter) {
        pool.requestBatch(1, HostStagingBlockPool::Clock::now() + std::chrono::seconds(30), [&](auto result) {
            callback_count.fetch_add(1);
            if (!result.has_value()) {
                cancelled_count.fetch_add(1);
            }
        });
    }

    pool.cancelAllBatchWaiters();
    EXPECT_EQ(callback_count.load(), 2u);
    EXPECT_EQ(cancelled_count.load(), 2u);
    pool.cancelAllBatchWaiters();
    EXPECT_EQ(callback_count.load(), 2u);

    held.reset();
    EXPECT_TRUE(pool.tryMallocBatch(1).has_value());
}

TEST(HostStagingBlockPoolTest, ThrowingCancelCallbackDoesNotSuppressLaterWaiters) {
    HostStagingBlockPool pool(1, 4096);
    auto                 held = pool.tryMallocBatch(1);
    ASSERT_TRUE(held.has_value());

    std::atomic<size_t> throwing_callback_count{0};
    std::atomic<size_t> later_callback_count{0};
    pool.requestBatch(1, HostStagingBlockPool::Clock::now() + std::chrono::seconds(30), [&](auto result) {
        EXPECT_FALSE(result.has_value());
        throwing_callback_count.fetch_add(1);
        throw std::runtime_error("expected cancellation callback failure");
    });
    pool.requestBatch(1, HostStagingBlockPool::Clock::now() + std::chrono::seconds(30), [&](auto result) {
        EXPECT_FALSE(result.has_value());
        later_callback_count.fetch_add(1);
    });

    EXPECT_NO_THROW(pool.cancelAllBatchWaiters());
    EXPECT_EQ(throwing_callback_count.load(), 1u);
    EXPECT_EQ(later_callback_count.load(), 1u);
    EXPECT_NO_THROW(pool.cancelAllBatchWaiters());
    EXPECT_EQ(throwing_callback_count.load(), 1u);
    EXPECT_EQ(later_callback_count.load(), 1u);

    held.reset();
    EXPECT_TRUE(pool.tryMallocBatch(1).has_value());
}

}  // namespace
}  // namespace rtp_llm
