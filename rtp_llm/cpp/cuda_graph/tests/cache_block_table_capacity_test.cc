#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"

namespace rtp_llm {
namespace {

void expectInvalidCapacity(int64_t          physical_tokens_per_block,
                           int64_t          kernel_tokens_per_block,
                           std::string_view expected_message) {
    try {
        (void)CacheBlockTableCapacity::fromBlockSizes(
            64, physical_tokens_per_block, kernel_tokens_per_block, 0, "invalid-test");
        FAIL() << "expected invalid cache block capacity";
    } catch (const RTPException& e) {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("context=invalid-test"), std::string::npos) << e.what();
    }
}

TEST(CacheBlockTableCapacityTest, DerivesHeterogeneousPhysicalAndKernelCapacities) {
    const auto subdivided = CacheBlockTableCapacity::fromBlockSizes(
        /*max_seq_len=*/65, /*physical_tokens_per_block=*/16, /*kernel_tokens_per_block=*/4, /*sp_steps=*/2, "full");
    EXPECT_EQ(subdivided.physical_block_table_capacity, 7);
    EXPECT_EQ(subdivided.kernel_block_table_capacity, 28);

    const auto unpartitioned = CacheBlockTableCapacity::fromBlockSizes(
        /*max_seq_len=*/65, /*physical_tokens_per_block=*/8, /*kernel_tokens_per_block=*/8, /*sp_steps=*/2, "linear");
    EXPECT_EQ(unpartitioned.physical_block_table_capacity, 11);
    EXPECT_EQ(unpartitioned.kernel_block_table_capacity, 11);
}

TEST(CacheBlockTableCapacityTest, RejectsInvalidBlockGranularity) {
    expectInvalidCapacity(/*physical_tokens_per_block=*/0, /*kernel_tokens_per_block=*/4, "physical tokens");
    expectInvalidCapacity(/*physical_tokens_per_block=*/16, /*kernel_tokens_per_block=*/0, "kernel tokens");
    expectInvalidCapacity(/*physical_tokens_per_block=*/16, /*kernel_tokens_per_block=*/6, "must be divisible");
}

}  // namespace
}  // namespace rtp_llm
