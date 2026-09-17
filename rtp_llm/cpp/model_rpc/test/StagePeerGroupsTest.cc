#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/StagePeerGroups.h"

namespace rtp_llm {
namespace {

ParallelismConfig makePpConfig(int64_t pp_size, int64_t tp_size, const std::vector<int64_t>& counts) {
    ParallelismConfig pc;
    pc.pp_size               = pp_size;
    pc.tp_size               = tp_size;
    pc.pp_stage_layer_counts = counts;
    return pc;
}

std::vector<std::string> makeWorkers(size_t n) {
    std::vector<std::string> workers;
    for (size_t i = 0; i < n; ++i) {
        workers.push_back("10.0.0." + std::to_string(i + 1) + ":100:200");
    }
    return workers;
}

}  // namespace

TEST(StagePeerGroups, SingleStageProducesNoGroups) {
    EXPECT_TRUE(buildStagePeerGroups(makePpConfig(1, 2, {}), makeWorkers(2)).empty());
}

TEST(StagePeerGroups, PpSlicesRangesAndPeers) {
    const auto groups = buildStagePeerGroups(makePpConfig(4, 2, {3, 3, 2, 2}), makeWorkers(8));
    ASSERT_EQ(groups.size(), 4u);
    const std::vector<std::pair<uint32_t, uint32_t>> expected = {{0, 3}, {3, 3}, {6, 2}, {8, 2}};
    for (size_t stage = 0; stage < groups.size(); ++stage) {
        EXPECT_EQ(groups[stage].range.begin, expected[stage].first) << "stage=" << stage;
        EXPECT_EQ(groups[stage].range.size, expected[stage].second) << "stage=" << stage;
        EXPECT_EQ(groups[stage].is_last_stage, stage + 1 == groups.size()) << "stage=" << stage;
        ASSERT_EQ(groups[stage].peer_addrs.size(), 2u) << "stage=" << stage;
        EXPECT_EQ(groups[stage].peer_addrs[0], "10.0.0." + std::to_string(stage * 2 + 1) + ":100:200");
        EXPECT_EQ(groups[stage].peer_addrs[1], "10.0.0." + std::to_string(stage * 2 + 2) + ":100:200");
    }
}

TEST(StagePeerGroups, MismatchedWorkerCountThrows) {
    EXPECT_THROW(buildStagePeerGroups(makePpConfig(4, 2, {2, 2, 2, 2}), makeWorkers(7)), std::exception);
}

TEST(StagePeerGroups, MissingLayerPartitionThrows) {
    EXPECT_THROW(buildStagePeerGroups(makePpConfig(2, 1, {}), makeWorkers(2)), std::exception);
}

TEST(StagePeerGroups, ValidateAcceptsTilingGroups) {
    const auto groups = buildStagePeerGroups(makePpConfig(4, 2, {3, 3, 2, 2}), makeWorkers(8));
    EXPECT_NO_THROW(validateStagePeerGroups(groups, 10));
}

TEST(StagePeerGroups, ValidateRejectsGap) {
    std::vector<StagePeerGroup> groups = {{{0, 3}, {"a"}, false}, {{4, 3}, {"b"}, true}};
    EXPECT_THROW(validateStagePeerGroups(groups, 7), std::exception);
}

TEST(StagePeerGroups, ValidateRejectsTotalMismatch) {
    std::vector<StagePeerGroup> groups = {{{0, 3}, {"a"}, false}, {{3, 3}, {"b"}, true}};
    EXPECT_THROW(validateStagePeerGroups(groups, 10), std::exception);
}

TEST(StagePeerGroups, ValidateRejectsWrongLastStageFlag) {
    std::vector<StagePeerGroup> groups = {{{0, 3}, {"a"}, true}, {{3, 3}, {"b"}, false}};
    EXPECT_THROW(validateStagePeerGroups(groups, 6), std::exception);
}

TEST(StagePeerGroups, PlanSlicesSymmetricTpCopiesWholeBlock) {
    const auto slices = planStagePeerSlices(2, 2, 1, false);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 1u);
    EXPECT_EQ(slices[0].local_piece_count, 1);
    EXPECT_EQ(slices[0].local_piece_id, 0);
    EXPECT_EQ(slices[0].peer_piece_count, 1);
    EXPECT_EQ(slices[0].peer_piece_id, 0);
}

TEST(StagePeerGroups, PlanSlicesFinerPrefillTpAssemblesPeerSlices) {
    const auto slices = planStagePeerSlices(4, 2, 1, false);
    ASSERT_EQ(slices.size(), 2u);
    EXPECT_EQ(slices[0].peer_index, 2u);
    EXPECT_EQ(slices[0].local_piece_count, 2);
    EXPECT_EQ(slices[0].local_piece_id, 0);
    EXPECT_EQ(slices[0].peer_piece_count, 1);
    EXPECT_EQ(slices[1].peer_index, 3u);
    EXPECT_EQ(slices[1].local_piece_id, 1);
}

TEST(StagePeerGroups, PlanSlicesFinerDecodeTpReadsSubSlice) {
    const auto slices = planStagePeerSlices(1, 2, 1, false);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 0u);
    EXPECT_EQ(slices[0].local_piece_count, 1);
    EXPECT_EQ(slices[0].peer_piece_count, 2);
    EXPECT_EQ(slices[0].peer_piece_id, 1);
}

TEST(StagePeerGroups, PlanSlicesReplicatedKvUsesSingleWholePeer) {
    const auto slices = planStagePeerSlices(1, 2, 1, true);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 0u);
    EXPECT_EQ(slices[0].local_piece_count, 1);
    EXPECT_EQ(slices[0].peer_piece_count, 1);
    EXPECT_EQ(slices[0].peer_piece_id, 0);
}

TEST(StagePeerGroups, PlanSlicesRejectsUnsupportedTpRatio) {
    EXPECT_THROW(planStagePeerSlices(3, 2, 0, false), std::exception);
}

TEST(StagePeerGroups, PlanSlicesRejectsOutOfRangeLane) {
    EXPECT_THROW(planStagePeerSlices(2, 2, 2, false), std::exception);
}

namespace {

StageGroupLoadParams makeGroupParams(int  peer_count,
                                     int  cp_size            = 1,
                                     bool prefill_cp_enabled = false,
                                     int  decode_tp_size     = 1,
                                     int  decode_tp_rank     = 0,
                                     bool replicated         = false,
                                     bool opaque             = false) {
    return {peer_count, cp_size, prefill_cp_enabled, decode_tp_size, decode_tp_rank, replicated, opaque};
}

}  // namespace

// Mode A reads whole blocks: one read per group peer, no cutting on either side.
TEST(StagePeerGroups, GroupLoadShardedMlaEnumeratesPeers) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 2, true, 2, 1, /*replicated=*/true));
    EXPECT_TRUE(plan.error.empty());
    EXPECT_TRUE(plan.page_level_rr);
    ASSERT_EQ(plan.loads.size(), 2u);
    EXPECT_EQ(plan.loads[0].peer_index, 0u);
    EXPECT_EQ(plan.loads[0].local_piece_count, 1);
    EXPECT_EQ(plan.loads[0].local_piece_id, 0);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 1);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 0);
    EXPECT_EQ(plan.loads[1].peer_index, 1u);
    EXPECT_EQ(plan.loads[1].local_piece_id, 0);
}

TEST(StagePeerGroups, GroupLoadShardedOpaqueAllowed) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 2, true, 2, 1, /*replicated=*/false, /*opaque=*/true));
    EXPECT_TRUE(plan.error.empty());
    EXPECT_TRUE(plan.page_level_rr);
    ASSERT_EQ(plan.loads.size(), 2u);
    EXPECT_EQ(plan.loads[0].peer_index, 0u);
    EXPECT_EQ(plan.loads[1].peer_index, 1u);
}

TEST(StagePeerGroups, GroupLoadShardedRejectsPlainMha) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 2, true, 2, 1, /*replicated=*/false, /*opaque=*/false));
    EXPECT_TRUE(plan.loads.empty());
    EXPECT_NE(plan.error.find("only supported for MLA or opaque"), std::string::npos);
}

TEST(StagePeerGroups, GroupLoadShardedRejectsPeerCountMismatch) {
    const auto plan = planStageGroupLoads(makeGroupParams(4, 2, true, 2, 1, /*replicated=*/true));
    EXPECT_TRUE(plan.loads.empty());
    EXPECT_NE(plan.error.find("requires stage peer group size"), std::string::npos);
}

// Mode B mirrors the flat CP-full tuple: one peer per lane, tp_d source slicing.
TEST(StagePeerGroups, GroupLoadFullReplicationPicksPeerByLane) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 1, true, 4, 3));
    EXPECT_TRUE(plan.error.empty());
    EXPECT_FALSE(plan.page_level_rr);
    ASSERT_EQ(plan.loads.size(), 1u);
    EXPECT_EQ(plan.loads[0].peer_index, 1u);
    EXPECT_EQ(plan.loads[0].local_piece_count, 1);
    EXPECT_EQ(plan.loads[0].local_piece_id, 0);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 4);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 3);
}

TEST(StagePeerGroups, GroupLoadFullReplicationBalancesPeersAcrossLanes) {
    const auto lane0 = planStageGroupLoads(makeGroupParams(2, 1, true, 4, 0));
    const auto lane1 = planStageGroupLoads(makeGroupParams(2, 1, true, 4, 1));
    ASSERT_EQ(lane0.loads.size(), 1u);
    ASSERT_EQ(lane1.loads.size(), 1u);
    EXPECT_EQ(lane0.loads[0].peer_index, 0u);
    EXPECT_EQ(lane1.loads[0].peer_index, 1u);
}

// Replicated KV keeps the same lane slicing as MHA in full-replication mode.
TEST(StagePeerGroups, GroupLoadFullReplicationKeepsReplicatedLaneSlicing) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 1, true, 4, 3, /*replicated=*/true));
    ASSERT_EQ(plan.loads.size(), 1u);
    EXPECT_EQ(plan.loads[0].peer_index, 1u);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 4);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 3);
}

TEST(StagePeerGroups, GroupLoadSymmetricTpMatchesFlatTuple) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 1, false, 2, 1));
    EXPECT_TRUE(plan.error.empty());
    EXPECT_FALSE(plan.page_level_rr);
    ASSERT_EQ(plan.loads.size(), 1u);
    EXPECT_EQ(plan.loads[0].peer_index, 1u);
    EXPECT_EQ(plan.loads[0].local_piece_count, 1);
    EXPECT_EQ(plan.loads[0].local_piece_id, 0);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 1);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 0);
}

TEST(StagePeerGroups, GroupLoadFinerPrefillTpAssemblesPeerSlices) {
    const auto plan = planStageGroupLoads(makeGroupParams(4, 1, false, 2, 1));
    ASSERT_EQ(plan.loads.size(), 2u);
    EXPECT_EQ(plan.loads[0].peer_index, 2u);
    EXPECT_EQ(plan.loads[0].local_piece_count, 2);
    EXPECT_EQ(plan.loads[0].local_piece_id, 0);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 1);
    EXPECT_EQ(plan.loads[1].peer_index, 3u);
    EXPECT_EQ(plan.loads[1].local_piece_id, 1);
}

TEST(StagePeerGroups, GroupLoadFinerDecodeTpReadsSubSlice) {
    const auto plan = planStageGroupLoads(makeGroupParams(1, 1, false, 2, 1));
    ASSERT_EQ(plan.loads.size(), 1u);
    EXPECT_EQ(plan.loads[0].peer_index, 0u);
    EXPECT_EQ(plan.loads[0].local_piece_count, 1);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 2);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 1);
}

TEST(StagePeerGroups, GroupLoadReplicatedKvUsesSingleWholePeer) {
    const auto plan = planStageGroupLoads(makeGroupParams(2, 1, false, 2, 1, /*replicated=*/true));
    ASSERT_EQ(plan.loads.size(), 1u);
    EXPECT_EQ(plan.loads[0].peer_index, 1u);
    EXPECT_EQ(plan.loads[0].peer_piece_count, 1);
    EXPECT_EQ(plan.loads[0].peer_piece_id, 0);
}

TEST(StagePeerGroups, GroupLoadEmptyPeerGroupRejected) {
    const auto plan = planStageGroupLoads(makeGroupParams(0, 1, false, 2, 0));
    EXPECT_TRUE(plan.loads.empty());
    EXPECT_FALSE(plan.error.empty());
}

}  // namespace rtp_llm
