
#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/RankLayout.h"

namespace rtp_llm {

namespace {

// Independent oracle: the canonical world-rank formula the Python authority
// (rank_layout.py) defines, written in a different algebraic form so the C++
// compose slice is checked against the source of truth, not against itself.
int64_t oracleLaneStride(int64_t dp_size, int64_t tp_size) {
    return dp_size * tp_size;
}

int64_t oracleStageRank(int64_t dp_rank, int64_t tp_rank, int64_t tp_size) {
    return dp_rank * tp_size + tp_rank;
}

int64_t oracleWorldRank(int64_t pp_stage, int64_t dp_rank, int64_t tp_rank, int64_t dp_size, int64_t tp_size) {
    return (pp_stage * dp_size + dp_rank) * tp_size + tp_rank;
}

RankLayout makeLattice(int64_t dp_size, int64_t tp_size, int64_t dp_rank, int64_t tp_rank) {
    RankLayout layout;
    layout.dp_size = dp_size;
    layout.tp_size = tp_size;
    layout.dp_rank = dp_rank;
    layout.tp_rank = tp_rank;
    return layout;
}

RankLayout makePp(int64_t pp_size, int64_t pp_rank) {
    RankLayout layout;
    layout.pp_size = pp_size;
    layout.pp_rank = pp_rank;
    return layout;
}

RankLayout makeMaterialized(std::vector<int64_t> counts, int64_t pp_rank) {
    RankLayout layout;
    layout.pp_size      = static_cast<int64_t>(counts.size());
    layout.pp_rank      = pp_rank;
    layout.layer_counts = std::move(counts);
    return layout;
}

}  // namespace

TEST(RankLayoutTest, ComposeSliceMatchesOracleExhaustive) {
    for (int64_t pp_size = 1; pp_size <= 3; ++pp_size) {
        for (int64_t dp_size = 1; dp_size <= 3; ++dp_size) {
            for (int64_t tp_size = 1; tp_size <= 3; ++tp_size) {
                for (int64_t dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
                    for (int64_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
                        const auto layout = makeLattice(dp_size, tp_size, dp_rank, tp_rank);
                        EXPECT_EQ(layout.laneStride(), oracleLaneStride(dp_size, tp_size));
                        EXPECT_EQ(layout.stageRank(), oracleStageRank(dp_rank, tp_rank, tp_size));
                        for (int64_t pp_stage = 0; pp_stage < pp_size; ++pp_stage) {
                            EXPECT_EQ(layout.worldRankOf(pp_stage),
                                      oracleWorldRank(pp_stage, dp_rank, tp_rank, dp_size, tp_size));
                        }
                    }
                }
            }
        }
    }
}

TEST(RankLayoutTest, WorldRankOfStageZeroIsStageRank) {
    const auto layout = makeLattice(/*dp_size=*/2, /*tp_size=*/4, /*dp_rank=*/1, /*tp_rank=*/3);
    EXPECT_EQ(layout.worldRankOf(0), layout.stageRank());
    EXPECT_EQ(layout.stageRank(), 1 * 4 + 3);
}

TEST(RankLayoutTest, SingleStageDegenerate) {
    auto layout = makePp(/*pp_size=*/1, /*pp_rank=*/0);
    EXPECT_TRUE(layout.hasEmbedding());
    EXPECT_TRUE(layout.hasLmHead());
    EXPECT_EQ(layout.myLayerRange(/*total_layers=*/64), (std::pair<int64_t, int64_t>{0, 64}));
}

TEST(RankLayoutTest, CapabilityFlagsAcrossStages) {
    for (int64_t rank = 0; rank < 4; ++rank) {
        auto layout = makePp(4, rank);
        EXPECT_EQ(layout.hasEmbedding(), rank == 0) << "rank " << rank;
        EXPECT_EQ(layout.hasLmHead(), rank == 3) << "rank " << rank;
    }
}

TEST(RankLayoutTest, RingNeighborRanks) {
    // pp=3, dp=1, tp=2: stage0 = ranks {0,1}, stage1 = {2,3}, stage2 = {4,5}.
    auto make = [](int64_t pp_rank, int64_t tp_rank) {
        RankLayout layout;
        layout.pp_size = 3;
        layout.pp_rank = pp_rank;
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
    RankLayout dp_lane;
    dp_lane.pp_size = 2;
    dp_lane.pp_rank = 0;
    dp_lane.dp_size = 2;
    dp_lane.tp_size = 2;
    dp_lane.dp_rank = 1;
    dp_lane.tp_rank = 1;
    EXPECT_EQ(dp_lane.laneStride(), 4);
    EXPECT_EQ(dp_lane.worldRankOf(0), 3);
    EXPECT_EQ(dp_lane.worldRankOf(1), 7);
    EXPECT_EQ(dp_lane.prevRank(), 7);
    EXPECT_EQ(dp_lane.nextRank(), 7);

    auto single = makePp(1, 0);
    EXPECT_EQ(single.prevRank(), 0);
    EXPECT_EQ(single.nextRank(), 0);
}

TEST(RankLayoutTest, MaterializedShapeSpecializedCounts) {
    auto layout = makeMaterialized({4, 12, 12}, 1);
    EXPECT_EQ(layout.myLayerRange(/*total_layers=*/28), (std::pair<int64_t, int64_t>{4, 16}));
    EXPECT_EQ(layout.layerRangeOf(2, /*total_layers=*/28), (std::pair<int64_t, int64_t>{16, 28}));
}

TEST(RankLayoutTest, MaterializedSingleStage) {
    auto layout = makeMaterialized({64}, 0);
    EXPECT_EQ(layout.myLayerRange(/*total_layers=*/64), (std::pair<int64_t, int64_t>{0, 64}));
}

TEST(RankLayoutTest, MaterializedRejectsInconsistentCounts) {
    auto layout    = makeMaterialized({8, 8}, 0);
    layout.pp_size = 3;  // size/pp_size mismatch
    EXPECT_THROW(layout.layerRangeOf(0, /*total_layers=*/16), std::exception);
    auto ok = makeMaterialized({8, 8}, 0);
    EXPECT_THROW(ok.layerRangeOf(2, /*total_layers=*/16), std::exception);
    EXPECT_THROW(ok.layerRangeOf(-1, /*total_layers=*/16), std::exception);
}

TEST(RankLayoutTest, FromParallelismConfig) {
    ParallelismConfig pc;
    pc.pp_size               = 3;
    pc.pp_rank               = 2;
    pc.dp_size               = 2;
    pc.tp_size               = 4;
    pc.dp_rank               = 1;
    pc.tp_rank               = 3;
    pc.pp_stage_layer_counts = {5, 4, 4};
    const auto layout        = RankLayout::fromParallelismConfig(pc);
    EXPECT_EQ(layout.pp_size, 3);
    EXPECT_EQ(layout.pp_rank, 2);
    EXPECT_EQ(layout.dp_size, 2);
    EXPECT_EQ(layout.tp_size, 4);
    EXPECT_EQ(layout.dp_rank, 1);
    EXPECT_EQ(layout.tp_rank, 3);
    EXPECT_EQ(layout.laneStride(), 8);
    EXPECT_EQ(layout.stageRank(), 7);
    EXPECT_EQ(layout.worldRankOf(2), 2 * 8 + 7);
    EXPECT_EQ(layout.myLayerRange(/*total_layers=*/13), (std::pair<int64_t, int64_t>{9, 13}));

    // Empty counts -> even-split fallback driven by the total_layers argument.
    ParallelismConfig bare;
    bare.pp_size           = 4;
    bare.pp_rank           = 0;
    const auto bare_layout = RankLayout::fromParallelismConfig(bare);
    EXPECT_EQ(bare_layout.layerRangeOf(0, /*total_layers=*/65), (std::pair<int64_t, int64_t>{0, 17}));
}

}  // namespace rtp_llm
