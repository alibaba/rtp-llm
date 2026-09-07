#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {

TEST(KVCacheTransferPlannerTest, LoadIgnoresSpeculativeDestinationReserve) {
    constexpr size_t source_block_num      = 23;
    constexpr size_t destination_block_num = 30;

    const auto linear_positions = blockPositionsForCacheLoad(destination_block_num,
                                                             source_block_num,
                                                             /*reuse_block_size=*/0,
                                                             /*use_hybrid=*/true,
                                                             /*transfer_tail_blocks=*/true,
                                                             /*tail_block_count=*/1,
                                                             /*hybrid_full_from_begin=*/true);
    EXPECT_EQ(linear_positions, (std::vector<size_t>{22}));

    const auto full_positions = blockPositionsForCacheLoad(destination_block_num,
                                                           source_block_num,
                                                           /*reuse_block_size=*/0,
                                                           /*use_hybrid=*/true,
                                                           /*transfer_tail_blocks=*/false,
                                                           /*tail_block_count=*/0,
                                                           /*hybrid_full_from_begin=*/true);
    ASSERT_EQ(full_positions.size(), source_block_num);
    EXPECT_EQ(full_positions.front(), 0);
    EXPECT_EQ(full_positions.back(), source_block_num - 1);
}

}  // namespace rtp_llm
