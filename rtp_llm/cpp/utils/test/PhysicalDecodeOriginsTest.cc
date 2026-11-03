#include "rtp_llm/cpp/utils/PhysicalDecodeOrigins.h"
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
PhysicalDecodeOrigins::Origin origin() {
    auto parsed = DecodeCacheOrigin::parse("v1." + std::string(32, 'a') + ".7680.7680.8192.8200.8201");
    return std::make_shared<const DecodeCacheOrigin>(*parsed);
}

TEST(PhysicalDecodeOriginsTest, BindRequiresTheSameLiveAllocation) {
    PhysicalDecodeOrigins table(4);
    EXPECT_EQ(table.incarnation(1), uint64_t{0});
    EXPECT_FALSE(table.bind(1, 0, "full", origin()));
    const auto stamp = table.beginAllocation(1);
    ASSERT_NE(stamp, uint64_t{0});
    EXPECT_FALSE(table.get(1, "full"));
    const auto source = origin();
    ASSERT_TRUE(table.bind(1, stamp, "full", source));
    EXPECT_EQ(table.get(1, "full"), source);
    EXPECT_FALSE(table.bind(1, stamp + 1, "full", nullptr));
    EXPECT_EQ(table.get(1, "full"), source);
}

TEST(PhysicalDecodeOriginsTest, ReallocatedOrReusedDonorRejectsLateCompletion) {
    PhysicalDecodeOrigins table(3);
    const auto            first = table.beginAllocation(1);
    ASSERT_TRUE(table.bind(1, first, "full", origin()));
    const auto donor = table.beginAllocation(1);  // same ID, different value
    EXPECT_NE(first, donor);
    EXPECT_FALSE(table.get(1, "full"));
    EXPECT_FALSE(table.bind(1, first, "full", origin()));
    ASSERT_TRUE(table.bind(1, donor, "full", origin()));
    table.release(1);
    EXPECT_EQ(table.incarnation(1), uint64_t{0});
    EXPECT_FALSE(table.get(1, "full"));
    EXPECT_FALSE(table.bind(1, donor, "full", origin()));
    const auto next = table.beginAllocation(1);
    EXPECT_NE(next, donor);
    EXPECT_FALSE(table.bind(1, donor, "full", origin()));
}

TEST(PhysicalDecodeOriginsTest, ClearingSourceDoesNotChangePayloadIncarnation) {
    PhysicalDecodeOrigins table(3);
    const auto            first  = table.beginAllocation(1);
    const auto            second = table.beginAllocation(2);
    ASSERT_TRUE(table.bind(1, first, "full", origin()));
    ASSERT_TRUE(table.bind(2, second, "full", origin()));
    EXPECT_TRUE(table.bind(1, first, "full", nullptr));
    EXPECT_EQ(table.incarnation(1), first);
    EXPECT_FALSE(table.get(1, "full"));
    EXPECT_TRUE(table.get(2, "full"));
}

TEST(PhysicalDecodeOriginsTest, RejectsOutOfRangePhysicalAddresses) {
    PhysicalDecodeOrigins table(3);
    // The pool validates addresses before calling its metadata helper.
    EXPECT_THROW(table.beginAllocation(3), std::out_of_range);
    EXPECT_THROW(table.get(3, "full"), std::out_of_range);
}

TEST(PhysicalDecodeOriginsTest, PackedGroupsShareLifetimeButKeepDifferentSources) {
    PhysicalDecodeOrigins table(3);
    const auto            stamp       = table.beginAllocation(1);
    const auto            full        = origin();
    auto                  state_value = *full;
    state_value.begin                 = 7168;
    state_value.end                   = 7680;
    const auto state                  = std::make_shared<const DecodeCacheOrigin>(state_value);
    ASSERT_TRUE(table.bind(1, stamp, "full", full));
    ASSERT_TRUE(table.bind(1, stamp, "state", state));
    EXPECT_EQ(table.get(1, "full"), full);
    EXPECT_EQ(table.get(1, "state"), state);
    EXPECT_FALSE(table.get(1, "unknown"));
    EXPECT_FALSE(table.bind(1, stamp, "", full));
    EXPECT_FALSE(table.bind(1, stamp, std::string(257, 'x'), full));
    EXPECT_TRUE(table.bind(1, stamp, "full", nullptr));
    EXPECT_FALSE(table.get(1, "full"));
    EXPECT_EQ(table.get(1, "state"), state);
    table.beginAllocation(1);
    EXPECT_FALSE(table.get(1, "state"));
    EXPECT_FALSE(table.bind(1, stamp, "state", state));
}
}  // namespace
}  // namespace rtp_llm
