#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeCacheLoadPlanner.h"

namespace rtp_llm {
namespace test {

TEST(DecodeCacheLoadPlannerTest, UseFullBlockRemoteLoadForMlaSparseOrHybrid) {
    CacheConfig config;
    config.cache_specs.resize(1);
    EXPECT_FALSE(useFullBlockRemoteLoad(config));

    config.use_mla = true;
    EXPECT_TRUE(useFullBlockRemoteLoad(config));

    config.use_mla   = false;
    config.is_sparse = true;
    EXPECT_TRUE(useFullBlockRemoteLoad(config));

    config.is_sparse = false;
    config.cache_specs.resize(2);
    EXPECT_TRUE(useFullBlockRemoteLoad(config));
}

TEST(DecodeCacheLoadPlannerTest, BlockPositionsForRpcHandlesHybridLinearAndFullGroups) {
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/5,
                                   /*reuse_block_size=*/2,
                                   /*use_hybrid=*/true,
                                   CacheGroupType::LINEAR,
                                   /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{4}));
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/1,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   CacheGroupType::LINEAR,
                                   /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{0}));
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/5,
                                   /*reuse_block_size=*/2,
                                   /*use_hybrid=*/true,
                                   CacheGroupType::SWA,
                                   /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{3, 4}));
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/4,
                                   /*reuse_block_size=*/2,
                                   /*use_hybrid=*/true,
                                   CacheGroupType::FULL,
                                   /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{0, 1, 2, 3}));
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/4,
                                   /*reuse_block_size=*/2,
                                   /*use_hybrid=*/true,
                                   CacheGroupType::FULL,
                                   /*hybrid_full_from_begin=*/false),
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(blockPositionsForRpc(/*block_num=*/4,
                                   /*reuse_block_size=*/1,
                                   /*use_hybrid=*/false,
                                   CacheGroupType::FULL,
                                   /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{1, 2, 3}));
}

TEST(DecodeCacheLoadPlannerTest, LayerRegionRequestKeyAddsRegionOnlyForTypedRegion) {
    EXPECT_EQ(layerRegionRequestKey(/*request_id=*/42, /*layer_id=*/3, KVCacheRegionName::DEFAULT), "42-3");
    EXPECT_EQ(layerRegionRequestKey(/*request_id=*/42, /*layer_id=*/3, KVCacheRegionName::SWA_KV),
              "42-3-" + std::to_string(static_cast<int>(KVCacheRegionName::SWA_KV)));
}

}  // namespace test
}  // namespace rtp_llm
