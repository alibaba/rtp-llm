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
    std::vector<StagePeerGroup> groups = {{{0, 3}, {"a"}}, {{4, 3}, {"b"}}};
    EXPECT_THROW(validateStagePeerGroups(groups, 7), std::exception);
}

TEST(StagePeerGroups, ValidateRejectsTotalMismatch) {
    std::vector<StagePeerGroup> groups = {{{0, 3}, {"a"}}, {{3, 3}, {"b"}}};
    EXPECT_THROW(validateStagePeerGroups(groups, 10), std::exception);
}

TEST(StagePeerGroups, PlanSlicesSymmetricTpCopiesWholeBlock) {
    const auto slices = planStagePeerSlices(2, 2, 1, false);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 1u);
    EXPECT_EQ(slices[0].dst_partition_count, 1);
    EXPECT_EQ(slices[0].dst_partition_id, 0);
    EXPECT_EQ(slices[0].src_partition_count, 1);
    EXPECT_EQ(slices[0].src_partition_id, 0);
}

TEST(StagePeerGroups, PlanSlicesFinerPrefillTpAssemblesPeerSlices) {
    const auto slices = planStagePeerSlices(4, 2, 1, false);
    ASSERT_EQ(slices.size(), 2u);
    EXPECT_EQ(slices[0].peer_index, 2u);
    EXPECT_EQ(slices[0].dst_partition_count, 2);
    EXPECT_EQ(slices[0].dst_partition_id, 0);
    EXPECT_EQ(slices[0].src_partition_count, 1);
    EXPECT_EQ(slices[1].peer_index, 3u);
    EXPECT_EQ(slices[1].dst_partition_id, 1);
}

TEST(StagePeerGroups, PlanSlicesFinerDecodeTpReadsSubSlice) {
    const auto slices = planStagePeerSlices(1, 2, 1, false);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 0u);
    EXPECT_EQ(slices[0].dst_partition_count, 1);
    EXPECT_EQ(slices[0].src_partition_count, 2);
    EXPECT_EQ(slices[0].src_partition_id, 1);
}

TEST(StagePeerGroups, PlanSlicesReplicatedKvUsesSingleWholePeer) {
    const auto slices = planStagePeerSlices(1, 2, 1, true);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].peer_index, 0u);
    EXPECT_EQ(slices[0].dst_partition_count, 1);
    EXPECT_EQ(slices[0].src_partition_count, 1);
    EXPECT_EQ(slices[0].src_partition_id, 0);
}

TEST(StagePeerGroups, PlanSlicesRejectsUnsupportedTpRatio) {
    EXPECT_THROW(planStagePeerSlices(3, 2, 0, false), std::exception);
}

TEST(StagePeerGroups, PlanSlicesRejectsOutOfRangeLane) {
    EXPECT_THROW(planStagePeerSlices(2, 2, 2, false), std::exception);
}

}  // namespace rtp_llm
