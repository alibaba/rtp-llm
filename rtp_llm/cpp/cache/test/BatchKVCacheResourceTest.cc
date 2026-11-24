#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {
namespace test {

TEST(BatchKVCacheResourceTest, RetainsFrozenAdmissionAcrossReshapeAndClearsForNewRequest) {
    BatchKVCacheResource resource;
    resource.setMockCachePlan({8, 2, 4, 16, 4, 8});
    resource.resetBatchSize(2);
    ASSERT_TRUE(resource.mockCachePlan());
    EXPECT_EQ(resource.mockCachePlan()->admissionOffsetBlocks(), 8);
    BatchKVCacheResource next;
    next.swap(resource);
    EXPECT_FALSE(resource.mockCachePlan());
    ASSERT_TRUE(next.mockCachePlan());
    std::vector<KVCacheResource> old_rows;
    next.resetAndReturnOldResources(1, old_rows);
    EXPECT_FALSE(next.mockCachePlan());
}
TEST(BatchKVCacheResourcePlanTest, RetryReusesFrozenRemoteOffset) {
    MockCacheHitController controller(0.25, 0.25, 0.5);
    controller.plan(7, [](size_t) { return true; });
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    for (int retry = 0; retry < 3; ++retry) {
        if (!resource.mockCachePlan()) {
            resource.setMockCachePlan(controller.plan(8, [](size_t) { return true; }));
        }
        controller.plan(2, [](size_t) { return true; });
        resource.resetBatchSize(2);
        ASSERT_TRUE(resource.mockCachePlan());
        EXPECT_EQ(resource.mockCachePlan()->admissionOffsetBlocks(), 7u);
        EXPECT_EQ(resource.mockCachePlan()->planned_remote_blocks, 4u);
        EXPECT_EQ(resource.mockCachePlan()->cumulative_planned_remote_blocks, 7u);
    }
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 21u);
    BatchKVCacheResource other;
    resource.swap(other);
    EXPECT_EQ(other.mockCachePlan()->admissionOffsetBlocks(), 7u);
    std::vector<KVCacheResource> old;
    other.resetAndReturnOldResources(1, old);
    EXPECT_FALSE(other.mockCachePlan());
}

}  // namespace test
}  // namespace rtp_llm
