#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <cstring>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(HostStagingBlockPoolTest, UsesCallerProvidedStride) {
    HostStagingBlockPool pool(1, 65, /*try_pin_memory=*/false);

    std::optional<HostStagingBlockPool::HostStagingBlockLease> lease = pool.malloc();
    ASSERT_TRUE(lease.has_value());
    const auto view = lease->blockBuffer(65);
    EXPECT_EQ(view.capacity_bytes, 65u);
}

TEST(HostStagingBlockPoolTest, PageableBackingServesLeasesWhenPinningDisabled) {
    HostStagingBlockPool pool(1, 4096, /*try_pin_memory=*/false);

    auto lease = pool.malloc();
    ASSERT_TRUE(lease.has_value());
    const auto view = lease->blockBuffer(64);
    ASSERT_NE(view.base, nullptr);
    EXPECT_EQ(view.payload_bytes, 64u);
    EXPECT_EQ(view.capacity_bytes, 4096u);
    std::memset(view.base, 0xAB, view.payload_bytes);

    EXPECT_FALSE(pool.malloc().has_value());
    lease.reset();
    EXPECT_TRUE(pool.malloc().has_value());
}

TEST(HostStagingBlockPoolTest, BatchAllocationIsAtomic) {
    HostStagingBlockPool pool(2, 4096, /*try_pin_memory=*/false);

    EXPECT_FALSE(pool.tryMallocBatch(3).has_value());
    auto leases = pool.tryMallocBatch(2);
    ASSERT_TRUE(leases.has_value());
    EXPECT_EQ(leases->size(), 2u);
    EXPECT_FALSE(pool.malloc().has_value());
}

TEST(HostStagingBlockPoolTest, BatchWaitersAreStrictFifo) {
    HostStagingBlockPool pool(2, 4096, /*try_pin_memory=*/false);
    auto held = pool.malloc();
    ASSERT_TRUE(held.has_value());

    std::vector<int> callback_order;
    std::optional<HostStagingBlockPool::HostStagingBlockBatch> first_leases;
    std::optional<HostStagingBlockPool::HostStagingBlockBatch> second_leases;
    pool.requestBatch(2, [&](auto result) {
        ASSERT_TRUE(result.has_value());
        callback_order.push_back(2);
        first_leases.emplace(std::move(*result));
    });
    pool.requestBatch(1, [&](auto result) {
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

TEST(HostStagingBlockPoolTest, CancelHeadWaiterPreservesProgress) {
    HostStagingBlockPool pool(1, 4096, /*try_pin_memory=*/false);
    auto held = pool.malloc();
    ASSERT_TRUE(held.has_value());

    bool first_cancelled = false;
    bool second_ready    = false;
    const auto first_id  = pool.requestBatch(1, [&](auto result) { first_cancelled = !result.has_value(); });
    pool.requestBatch(1, [&](auto result) { second_ready = result.has_value(); });

    EXPECT_TRUE(pool.cancelBatchWaiter(first_id));
    EXPECT_TRUE(first_cancelled);
    EXPECT_FALSE(second_ready);
    held.reset();
    EXPECT_TRUE(second_ready);
}

}  // namespace
}  // namespace rtp_llm
