#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <cstring>

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

}  // namespace
}  // namespace rtp_llm
