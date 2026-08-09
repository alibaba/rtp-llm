#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkWorkload.h"

#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {

TEST(TransferBenchmarkWorkloadTest, VisitsWorkingSetLargerThanConcurrency) {
    constexpr size_t kWorkers    = 3;
    constexpr size_t kDirections = 2;
    constexpr size_t kWorkingSet = 11;
    std::set<size_t> visited;
    size_t           operations = 0;
    for (size_t worker = 0; worker < kWorkers; ++worker) {
        for (const auto& op : scheduleTransferWorker(
                 kWorkingSet * kDirections, kDirections, 0, kWorkingSet, false, 42, kWorkers, worker)) {
            visited.insert(op.working_set_index);
            ++operations;
        }
    }
    EXPECT_EQ(operations, kWorkingSet * kDirections);
    EXPECT_EQ(visited.size(), kWorkingSet);
}

TEST(TransferBenchmarkWorkloadTest, AdjacentDirectionsShareLogicalCoordinate) {
    constexpr size_t    kWorkers = 4;
    std::vector<size_t> first_direction_index(13, 99);
    std::vector<size_t> second_direction_index(13, 98);
    for (size_t worker = 0; worker < kWorkers; ++worker) {
        for (const auto& op : scheduleTransferWorker(26, 2, 17, 7, true, 123, kWorkers, worker)) {
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

TEST(TransferBenchmarkWorkloadTest, OperationCountIsGlobalAcrossWorkers) {
    constexpr size_t kRequested = 37;
    size_t           attempted  = 0;
    for (size_t worker = 0; worker < 8; ++worker) {
        attempted += scheduleTransferWorker(kRequested, 2, 0, 32, false, 42, 8, worker).size();
    }
    EXPECT_EQ(attempted, kRequested);
}

}  // namespace rtp_llm::benchmark
