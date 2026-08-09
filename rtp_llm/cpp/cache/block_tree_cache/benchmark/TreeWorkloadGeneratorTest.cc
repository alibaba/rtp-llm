#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <numeric>
#include <set>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
namespace {

StatefulPathConfig testConfig() {
    StatefulPathConfig config;
    config.max_path_length         = 64;
    config.initial_min_path_length = 16;
    config.initial_max_path_length = 32;
    config.append_length           = 8;
    config.inserts_per_match       = 4;
    config.active_path_limit       = 64;
    config.continuation_ratio      = 0.7;
    config.fork_ratio              = 0.2;
    config.fork_reuse_min_ratio    = 0.25;
    config.fork_reuse_max_ratio    = 0.9;
    config.hot_path_ratio          = 0.0;
    return config;
}

PathKeys sequentialPath(int64_t first, size_t length) {
    PathKeys path(length);
    std::iota(path.begin(), path.end(), first);
    return path;
}

bool hasPrefix(const PathKeys& path, const PathKeys& prefix, size_t prefix_length) {
    return path.size() >= prefix_length && prefix.size() >= prefix_length
           && std::equal(path.begin(), path.begin() + prefix_length, prefix.begin());
}

void expectIncrementalInserts(const StatefulPathOperation& operation, const StatefulPathConfig& config) {
    ASSERT_EQ(operation.insert_paths.size(), config.inserts_per_match);
    SharedPath previous = operation.match_path;
    for (const auto& insert_path : operation.insert_paths) {
        ASSERT_TRUE(insert_path);
        EXPECT_EQ(insert_path->size(), previous->size() + config.append_length);
        EXPECT_TRUE(hasPrefix(*insert_path, *previous, previous->size()));
        EXPECT_LE(insert_path->size(), config.max_path_length);
        previous = insert_path;
    }
}

TEST(TreeWorkloadGeneratorTest, TopologyHasExactNodeCountAndBoundedVariedPaths) {
    auto                  config = testConfig();
    TreeWorkloadGenerator generator(42, 256, 4, config);
    const auto            metadata = generator.generateTopology();

    std::set<PathKeys> unique_nodes;
    std::set<size_t>   path_lengths;
    const auto&        tree_paths = generator.treePaths();
    for (const auto& path : tree_paths) {
        EXPECT_GE(path.size(), config.initial_min_path_length);
        EXPECT_LE(path.size(), config.initial_max_path_length);
        path_lengths.insert(path.size());
        for (size_t depth = 1; depth <= path.size(); ++depth) {
            unique_nodes.emplace(path.begin(), path.begin() + depth);
        }
    }
    size_t expected_leaf_count = 0;
    for (const auto& candidate : tree_paths) {
        const bool has_child = std::any_of(tree_paths.begin(), tree_paths.end(), [&](const PathKeys& other) {
            return other.size() > candidate.size() && hasPrefix(other, candidate, candidate.size());
        });
        expected_leaf_count += has_child ? 0 : 1;
    }

    EXPECT_EQ(unique_nodes.size(), 256);
    EXPECT_EQ(metadata.actual_node_count, 256);
    EXPECT_EQ(metadata.leaf_count, expected_leaf_count);
    EXPECT_GT(path_lengths.size(), 1);
    EXPECT_LE(metadata.max_depth, config.initial_max_path_length);
}

TEST(StatefulPathSessionTest, ContinuationMatchesWholeCandidateThenAppends) {
    auto config                 = testConfig();
    config.continuation_ratio   = 1.0;
    config.fork_ratio           = 0.0;
    const PathKeys      initial = sequentialPath(1, 16);
    StatefulPathSession session(7, 3, {initial}, 0, 1, config);

    const auto operation = session.nextOperation();

    EXPECT_EQ(operation.scenario, PathScenario::CONTINUATION);
    EXPECT_EQ(*operation.match_path, initial);
    EXPECT_EQ(operation.planned_reuse_prefix_length, initial.size());
    EXPECT_EQ(operation.planned_new_node_count, config.append_length * config.inserts_per_match);
    expectIncrementalInserts(operation, config);
}

TEST(StatefulPathSessionTest, ForkReusesPrefixAndDivergesBeforeAppending) {
    auto config                 = testConfig();
    config.continuation_ratio   = 0.0;
    config.fork_ratio           = 1.0;
    config.fork_reuse_min_ratio = 0.5;
    config.fork_reuse_max_ratio = 0.5;
    const PathKeys      initial = sequentialPath(1, 32);
    StatefulPathSession session(11, 4, {initial}, 0, 1, config);

    const auto operation = session.nextOperation();

    EXPECT_EQ(operation.scenario, PathScenario::FORK);
    EXPECT_GT(operation.planned_reuse_prefix_length, 0);
    EXPECT_LT(operation.planned_reuse_prefix_length, operation.match_path->size());
    EXPECT_TRUE(hasPrefix(*operation.match_path, initial, operation.planned_reuse_prefix_length));
    EXPECT_NE((*operation.match_path)[operation.planned_reuse_prefix_length],
              initial[operation.planned_reuse_prefix_length]);
    expectIncrementalInserts(operation, config);
}

TEST(StatefulPathSessionTest, ColdRequestUsesANewRootPathBeforeAppending) {
    auto config                 = testConfig();
    config.continuation_ratio   = 0.0;
    config.fork_ratio           = 0.0;
    const PathKeys      initial = sequentialPath(1, 32);
    StatefulPathSession session(13, 5, {initial}, 0, 1, config);

    const auto operation = session.nextOperation();

    EXPECT_EQ(operation.scenario, PathScenario::COLD);
    EXPECT_EQ(operation.planned_reuse_prefix_length, 0);
    EXPECT_NE(operation.match_path->front(), initial.front());
    expectIncrementalInserts(operation, config);
}

TEST(StatefulPathSessionTest, NormalCandidatesAreSelectedWithoutReplacementPerEpoch) {
    auto config                = testConfig();
    config.continuation_ratio  = 1.0;
    config.fork_ratio          = 0.0;
    config.hot_path_ratio      = 0.0;
    const PathKeys      first  = sequentialPath(1, 16);
    const PathKeys      second = sequentialPath(101, 16);
    StatefulPathSession session(17, 6, {first, second}, 0, 1, config);

    const auto first_operation  = session.nextOperation();
    const auto second_operation = session.nextOperation();

    EXPECT_NE(*first_operation.match_path, *second_operation.match_path);
    EXPECT_TRUE(*first_operation.match_path == first || *first_operation.match_path == second);
    EXPECT_TRUE(*second_operation.match_path == first || *second_operation.match_path == second);
}

TEST(StatefulPathSessionTest, SameSeedAndAdvanceProduceTheSameTrace) {
    auto                config = testConfig();
    const PathKeys      first  = sequentialPath(1, 16);
    const PathKeys      second = sequentialPath(101, 24);
    StatefulPathSession a(19, 7, {first, second}, 0, 1, config);
    StatefulPathSession b(19, 7, {first, second}, 0, 1, config);

    for (size_t i = 0; i < 20; ++i) {
        const auto left  = a.nextOperation();
        const auto right = b.nextOperation();
        EXPECT_EQ(left.scenario, right.scenario);
        EXPECT_EQ(*left.match_path, *right.match_path);
        ASSERT_EQ(left.insert_paths.size(), right.insert_paths.size());
        for (size_t j = 0; j < left.insert_paths.size(); ++j) {
            EXPECT_EQ(*left.insert_paths[j], *right.insert_paths[j]);
        }
    }
}

}  // namespace
}  // namespace rtp_llm::benchmark
