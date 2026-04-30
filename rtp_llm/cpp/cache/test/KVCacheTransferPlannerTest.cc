#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {
namespace test {

TEST(KVCacheTransferPlannerTest, HybridTransferPositionsFollowGroupTypePolicy) {
    EXPECT_EQ(blockPositionsForCacheTransfer(/*block_num=*/5,
                                             /*reuse_block_size=*/2,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::FULL,
                                             /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(blockPositionsForCacheTransfer(/*block_num=*/5,
                                             /*reuse_block_size=*/2,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::LINEAR,
                                             /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{4}));
    EXPECT_EQ(blockPositionsForCacheTransfer(/*block_num=*/5,
                                             /*reuse_block_size=*/2,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::SWA,
                                             /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{3, 4}));
}

TEST(KVCacheTransferPlannerTest, NonHybridTransferStartsAfterReuseBlocks) {
    EXPECT_EQ(blockPositionsForCacheTransfer(/*block_num=*/4,
                                             /*reuse_block_size=*/1,
                                             /*use_hybrid=*/false,
                                             CacheGroupType::SWA,
                                             /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{1, 2, 3}));
}

}  // namespace test
}  // namespace rtp_llm
