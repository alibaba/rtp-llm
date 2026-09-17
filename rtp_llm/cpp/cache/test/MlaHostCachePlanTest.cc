#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/MlaHostCachePlan.h"

namespace rtp_llm {
namespace {

MlaHostCacheBudget glm53Budget() {
    return {512ULL << 20, 128ULL << 20, 2 * 256 * 528, 2 * 256 * 36, 2, 256, 128, 1, 2051};
}

TEST(MlaHostCachePlanTest, ExpandedTopkAndKernelPageAlignment) {
    const auto b    = glm53Budget();
    const auto plan = planMlaHostCache(b);
    EXPECT_EQ(plan.resident_tokens, 2176);
    EXPECT_NE(plan.resident_tokens % b.allocator_tokens, 0);
    EXPECT_LE(plan.hbm_bytes, b.hbm_bytes);
    EXPECT_LE(plan.host_blocks * b.mla_block_bytes, b.host_bytes);
    EXPECT_GE(plan.hbm_blocks, 2);
    // Host expansion must provide more logical capacity than an all-HBM pool.
    EXPECT_GT(plan.hbm_blocks + plan.host_blocks, b.hbm_bytes / (b.mla_block_bytes + b.auxiliary_block_bytes));
}

TEST(MlaHostCachePlanTest, VerifyBatchRequiresEveryRawSelection) {
    auto b                      = glm53Budget();
    b.query_tokens              = 16;
    b.requested_resident_tokens = 16 * 512;  // Indexer groups alone are insufficient.
    EXPECT_THROW(planMlaHostCache(b), std::invalid_argument);
    b.requested_resident_tokens = 0;
    const auto plan             = planMlaHostCache(b);
    EXPECT_GE(plan.resident_tokens, 16 * 2051);
    EXPECT_EQ(plan.resident_tokens % 128, 0);
    EXPECT_LE(plan.hbm_bytes, b.hbm_bytes);
}

TEST(MlaHostCachePlanTest, AuxiliaryHbmLimitsHostCapacity) {
    auto b                   = glm53Budget();
    b.host_bytes             = 1ULL << 40;
    const auto small_indexer = planMlaHostCache(b);
    b.auxiliary_block_bytes *= 32;
    const auto large_indexer = planMlaHostCache(b);
    EXPECT_LT(large_indexer.host_blocks, small_indexer.host_blocks);
    EXPECT_LE(large_indexer.hbm_bytes, b.hbm_bytes);
    EXPECT_GE(large_indexer.hbm_blocks, 2);
}

TEST(MlaHostCachePlanTest, RejectsInvalidCapacityAndOverflow) {
    auto b      = glm53Budget();
    b.hbm_bytes = 1024;
    EXPECT_THROW(planMlaHostCache(b), std::invalid_argument);
    b            = glm53Budget();
    b.host_bytes = 1;
    EXPECT_THROW(planMlaHostCache(b), std::invalid_argument);
    b              = glm53Budget();
    b.query_tokens = std::numeric_limits<size_t>::max();
    EXPECT_THROW(planMlaHostCache(b), std::invalid_argument);
    b               = glm53Budget();
    b.kernel_tokens = 192;
    EXPECT_THROW(planMlaHostCache(b), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
