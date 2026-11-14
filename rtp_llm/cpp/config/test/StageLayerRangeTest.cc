#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/StageLayerRange.h"

namespace rtp_llm {

TEST(StageLayerRangeTest, UsesConfiguredCountsAndGlobalLayerIds) {
    ParallelismConfig config;
    config.pp_size               = 3;
    config.pp_stage_layer_counts = {2, 3, 3};

    const StageLayerRange expected[] = {{0, 2}, {2, 3}, {5, 3}};
    for (int64_t stage = 0; stage < 3; ++stage) {
        const auto range = stageLayerRange(config, stage, 8);
        EXPECT_EQ(range.begin, expected[stage].begin);
        EXPECT_EQ(range.size, expected[stage].size);
        EXPECT_EQ(range.end(), expected[stage].end());
    }
}

TEST(StageLayerRangeTest, SingleStageIgnoresPartitionCounts) {
    ParallelismConfig config;
    config.pp_stage_layer_counts = {2, 5};

    const auto range = stageLayerRange(config, 0, 8);
    EXPECT_EQ(range.begin, 0u);
    EXPECT_EQ(range.size, 8u);
    EXPECT_EQ(range.end(), 8u);
}

}  // namespace rtp_llm
