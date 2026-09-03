
#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/PPLayout.h"

using namespace std;

namespace rtp_llm {

namespace {

PPLayout makeLayout(int64_t pp_size, int64_t pp_rank, int64_t total_layers) {
    PPLayout layout;
    layout.pp_size      = pp_size;
    layout.pp_rank      = pp_rank;
    layout.total_layers = total_layers;
    return layout;
}

}  // namespace

TEST(PPLayoutTest, SingleStageDegenerate) {
    auto layout = makeLayout(/*pp_size=*/1, /*pp_rank=*/0, /*total_layers=*/64);
    EXPECT_TRUE(layout.hasEmbedding());
    EXPECT_TRUE(layout.hasLmHead());
    EXPECT_EQ(layout.myLayerRange(), (std::pair<int64_t, int64_t>{0, 64}));
}

TEST(PPLayoutTest, CapabilityFlagsAcrossStages) {
    for (int64_t rank = 0; rank < 4; ++rank) {
        auto layout = makeLayout(4, rank, 64);
        EXPECT_EQ(layout.hasEmbedding(), rank == 0) << "rank " << rank;
        EXPECT_EQ(layout.hasLmHead(), rank == 3) << "rank " << rank;
    }
}

TEST(PPLayoutTest, RingNeighborRanks) {
    // pp=3, dp=1, tp=2: stage0 = ranks {0,1}, stage1 = {2,3}, stage2 = {4,5}.
    auto make = [](int64_t pp_rank, int64_t tp_rank) {
        PPLayout layout;
        layout.pp_size = 3;
        layout.pp_rank = pp_rank;
        layout.dp_size = 1;
        layout.tp_size = 2;
        layout.tp_rank = tp_rank;
        return layout;
    };
    EXPECT_EQ(make(0, 0).laneStride(), 2);
    EXPECT_EQ(make(1, 0).prevRank(), 0);
    EXPECT_EQ(make(1, 0).nextRank(), 4);
    EXPECT_EQ(make(1, 1).prevRank(), 1);
    EXPECT_EQ(make(1, 1).nextRank(), 5);
    EXPECT_EQ(make(0, 0).prevRank(), 4);
    EXPECT_EQ(make(0, 0).nextRank(), 2);
    EXPECT_EQ(make(2, 0).prevRank(), 2);
    EXPECT_EQ(make(2, 0).nextRank(), 0);

    // dp>1: lane offset dp_rank*tp_size must be included; lane of rank 3 is {3, 7}, not {3, 5}.
    PPLayout dp_lane;
    dp_lane.pp_size = 2;
    dp_lane.pp_rank = 0;
    dp_lane.dp_size = 2;
    dp_lane.tp_size = 2;
    dp_lane.dp_rank = 1;
    dp_lane.tp_rank = 1;
    EXPECT_EQ(dp_lane.laneStride(), 4);
    EXPECT_EQ(dp_lane.rankOfStage(0), 3);
    EXPECT_EQ(dp_lane.rankOfStage(1), 7);
    EXPECT_EQ(dp_lane.prevRank(), 7);
    EXPECT_EQ(dp_lane.nextRank(), 7);

    PPLayout single = makeLayout(1, 0, 8);
    EXPECT_EQ(single.prevRank(), 0);
    EXPECT_EQ(single.nextRank(), 0);
}

namespace {

PPLayout makeMaterializedLayout(std::vector<int64_t> counts, int64_t pp_rank) {
    PPLayout layout;
    layout.pp_size = static_cast<int64_t>(counts.size());
    layout.pp_rank = pp_rank;
    for (const auto c : counts) {
        layout.total_layers += c;
    }
    layout.layer_counts = std::move(counts);
    return layout;
}

}  // namespace

TEST(PPLayoutTest, MaterializedShapeSpecializedCounts) {
    auto layout = makeMaterializedLayout({4, 12, 12}, 1);
    EXPECT_EQ(layout.myLayerRange(), (std::pair<int64_t, int64_t>{4, 16}));
    EXPECT_EQ(layout.layerRangeOf(2), (std::pair<int64_t, int64_t>{16, 28}));
}

TEST(PPLayoutTest, MaterializedSingleStage) {
    auto layout = makeMaterializedLayout({64}, 0);
    EXPECT_EQ(layout.myLayerRange(), (std::pair<int64_t, int64_t>{0, 64}));
}

TEST(PPLayoutTest, MaterializedRejectsInconsistentCounts) {
    auto layout    = makeMaterializedLayout({8, 8}, 0);
    layout.pp_size = 3;  // size/pp_size mismatch
    EXPECT_THROW(layout.layerRangeOf(0), std::exception);
    auto ok = makeMaterializedLayout({8, 8}, 0);
    EXPECT_THROW(ok.layerRangeOf(2), std::exception);
    EXPECT_THROW(ok.layerRangeOf(-1), std::exception);
}

TEST(PPLayoutTest, FromParallelismConfig) {
    ParallelismConfig pc;
    pc.pp_size               = 3;
    pc.pp_rank               = 2;
    pc.dp_size               = 2;
    pc.tp_size               = 4;
    pc.dp_rank               = 1;
    pc.tp_rank               = 3;
    pc.pp_stage_layer_counts = {5, 4, 4};
    const auto layout        = PPLayout::fromParallelismConfig(pc, 13);
    EXPECT_EQ(layout.pp_size, 3);
    EXPECT_EQ(layout.pp_rank, 2);
    EXPECT_EQ(layout.dp_size, 2);
    EXPECT_EQ(layout.tp_size, 4);
    EXPECT_EQ(layout.dp_rank, 1);
    EXPECT_EQ(layout.tp_rank, 3);
    EXPECT_EQ(layout.myLayerRange(), (std::pair<int64_t, int64_t>{9, 13}));

    // Empty counts -> even-split fallback.
    ParallelismConfig bare;
    bare.pp_size           = 4;
    bare.pp_rank           = 0;
    const auto bare_layout = PPLayout::fromParallelismConfig(bare, 65);
    EXPECT_EQ(bare_layout.layerRangeOf(0), (std::pair<int64_t, int64_t>{0, 17}));
}

}  // namespace rtp_llm
