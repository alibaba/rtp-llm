#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkWorkload.h"

#include <algorithm>
#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {

TEST(TransferBenchmarkWorkloadTest, VisitsWorkingSetLargerThanConcurrency) {
    constexpr size_t kWaveWidth  = 3;
    constexpr size_t kDirections = 2;
    constexpr size_t kWorkingSet = 11;
    std::set<size_t> visited;
    size_t           operations = 0;
    for (size_t begin = 0; begin < kWorkingSet; begin += kWaveWidth) {
        for (const auto& op : scheduleTransferWave(
                 kWorkingSet * kDirections, kDirections, 0, kWorkingSet, false, 42, begin, kWaveWidth)) {
            visited.insert(op.working_set_index);
            ++operations;
        }
    }
    EXPECT_EQ(operations, kWorkingSet * kDirections);
    EXPECT_EQ(visited.size(), kWorkingSet);
}

TEST(TransferBenchmarkWorkloadTest, AdjacentDirectionsShareLogicalCoordinate) {
    constexpr size_t    kWaveWidth = 4;
    std::vector<size_t> first_direction_index(13, 99);
    std::vector<size_t> second_direction_index(13, 98);
    for (size_t begin = 0; begin < 13; begin += kWaveWidth) {
        for (const auto& op : scheduleTransferWave(26, 2, 17, 7, true, 123, begin, kWaveWidth)) {
            const size_t coordinate_offset = op.logical_coordinate - 17;
            if (op.direction_index == 0) {
                first_direction_index[coordinate_offset] = op.working_set_index;
            } else {
                second_direction_index[coordinate_offset] = op.working_set_index;
            }
        }
    }
    EXPECT_EQ(first_direction_index, second_direction_index);
}

TEST(TransferBenchmarkWorkloadTest, OperationCountIsGlobalAcrossWaves) {
    constexpr size_t kRequested = 37;
    size_t           attempted  = 0;
    for (size_t begin = 0; begin < 19; begin += 8) {
        attempted += scheduleTransferWave(kRequested, 2, 0, 32, false, 42, begin, 8).size();
    }
    EXPECT_EQ(attempted, kRequested);
}

TEST(TransferBenchmarkWorkloadTest, EachWaveAssignsUniqueReusableLanes) {
    constexpr size_t kWaveWidth = 8;
    for (size_t begin = 0; begin < 21; begin += kWaveWidth) {
        const auto       operations = scheduleTransferWave(42, 2, 0, 32, false, 42, begin, kWaveWidth);
        std::set<size_t> lanes;
        for (const auto& op : operations) {
            if (op.direction_index == 0) {
                lanes.insert(op.lane_index);
            }
            EXPECT_LT(op.lane_index, kWaveWidth);
        }
        EXPECT_EQ(lanes.size(), std::min(kWaveWidth, size_t{21} - begin));
    }
}

TEST(TransferBenchmarkWorkloadTest, RandomPermutationKeepsEndpointsUniqueWithinWave) {
    constexpr size_t kWaveWidth  = 8;
    constexpr size_t kWorkingSet = 11;
    for (size_t begin = 0; begin < 22; begin += kWaveWidth) {
        const auto       operations = scheduleTransferWave(44, 2, 0, kWorkingSet, true, 123, begin, kWaveWidth);
        std::set<size_t> working_set_indices;
        for (const auto& op : operations) {
            if (op.direction_index == 0) {
                working_set_indices.insert(op.working_set_index);
            }
        }
        EXPECT_EQ(working_set_indices.size(), std::min(kWaveWidth, size_t{22} - begin));
    }
}

}  // namespace rtp_llm::benchmark
