#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <set>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
namespace {

TEST(TreeWorkloadGeneratorTest, MixedTraceIsDeterministicAndStructurallyValid) {
    const auto            config = OnlineTreeWorkloadConfig::smokeTestConfig();
    TreeWorkloadGenerator first(42, config);
    TreeWorkloadGenerator second(42, config);

    const auto first_topology  = first.generateTopology();
    const auto second_topology = second.generateTopology();
    first.generateTrace();
    second.generateTrace();

    EXPECT_EQ(first_topology.actual_node_count, config.shared_base_nodes + config.background_tree_nodes);
    EXPECT_EQ(first_topology.actual_node_count, second_topology.actual_node_count);
    ASSERT_EQ(first.trace().size(), config.operation_trace_count);
    ASSERT_EQ(first.trace().size(), second.trace().size());
    EXPECT_EQ(first.traceHash(), second.traceHash());
    EXPECT_EQ(first.workloadDefinitionHash(), second.workloadDefinitionHash());
    EXPECT_GT(first.baseRequestCount(), 0u);
    EXPECT_GT(first.continuationRequestCount(), 0u);

    std::set<size_t> families_with_continuation;
    for (size_t i = 0; i < first.trace().size(); ++i) {
        const auto& request = first.trace()[i];
        const auto& copy    = second.trace()[i];
        EXPECT_EQ(request.request_id, i);
        EXPECT_EQ(request.path, copy.path);
        EXPECT_EQ(request.family_id, copy.family_id);
        EXPECT_EQ(request.epoch_id, copy.epoch_id);
        EXPECT_EQ(request.generation, copy.generation);
        EXPECT_EQ(request.predecessor_id, copy.predecessor_id);
        EXPECT_EQ(request.is_continuation, copy.is_continuation);
        EXPECT_EQ(request.path.size(), request.input_blocks);
        EXPECT_LT(request.planned_reuse_blocks, request.path.size());

        if (!request.is_continuation) {
            EXPECT_EQ(request.generation, 0u);
            EXPECT_EQ(request.predecessor_id, -1);
            continue;
        }

        ASSERT_GE(request.predecessor_id, 0);
        ASSERT_LT(static_cast<size_t>(request.predecessor_id), i);
        const auto& parent = first.trace()[static_cast<size_t>(request.predecessor_id)];
        EXPECT_EQ(request.family_id, parent.family_id);
        EXPECT_EQ(request.epoch_id, parent.epoch_id);
        EXPECT_EQ(request.generation, parent.generation + 1);
        EXPECT_EQ(request.planned_reuse_blocks, parent.path.size());
        ASSERT_GT(request.path.size(), parent.path.size());
        EXPECT_TRUE(std::equal(parent.path.begin(), parent.path.end(), request.path.begin()));
        families_with_continuation.insert(request.family_id);
    }
    EXPECT_EQ(families_with_continuation.size(), config.logical_concurrency);
}

TEST(TreeWorkloadGeneratorTest, GeneratedKeySpaceDoesNotCollide) {
    auto config                  = OnlineTreeWorkloadConfig::smokeTestConfig();
    config.tokens_per_block      = 1;
    config.logical_concurrency   = 2;
    config.shared_base_nodes     = 12;
    config.background_tree_nodes = 12;
    config.operation_trace_count = 64;
    config.length_buckets_tokens = {4, 8, 12};
    config.length_weights        = {1, 1, 1};
    config.hit_rates_percent     = {0, 50, 99};

    TreeWorkloadGenerator generator(7, config);
    generator.generateTopology();
    generator.generateTrace();

    std::set<int64_t> topology_keys;
    for (const auto& path : generator.topologyPaths()) {
        for (const int64_t key : path) {
            EXPECT_TRUE(topology_keys.insert(key).second);
        }
    }
    EXPECT_EQ(topology_keys.size(), config.shared_base_nodes + config.background_tree_nodes);

    std::set<int64_t> generated_keys;
    for (const auto& request : generator.trace()) {
        size_t generated_begin = request.planned_reuse_blocks;
        if (request.is_continuation) {
            const auto& parent = generator.trace()[static_cast<size_t>(request.predecessor_id)];
            generated_begin    = parent.path.size();
        }
        for (size_t i = generated_begin; i < request.path.size(); ++i) {
            EXPECT_EQ(topology_keys.count(request.path[i]), 0u);
            EXPECT_TRUE(generated_keys.insert(request.path[i]).second);
        }
    }
}

}  // namespace
}  // namespace rtp_llm::benchmark
