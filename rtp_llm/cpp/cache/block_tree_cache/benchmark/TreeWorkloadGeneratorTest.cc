#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <chrono>
#include <set>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/OnlineTreeScheduler.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm::benchmark {
namespace {

class RequestBlockCleanupCache: public OnlineCacheApi {
public:
    MatchOutcome match(const PathKeys& path) override {
        MatchOutcome outcome;
        outcome.actual_matched_depth      = path.size();
        outcome.matched_device_blocks     = 1;
        outcome.joined_target_block_count = 1;
        return outcome;
    }

    void materializeRequestBlocks(MatchOutcome& outcome) override {
        ++materialize_calls;
        outcome.request_blocks = {{11, 12, 13}};
    }

    bool allocateLoadTargets(const MatchOutcome&, PreparedRequestResources&) override {
        return true;
    }

    bool allocateSuffixBlocks(size_t, PreparedRequestResources&) override {
        return suffix_allocation_succeeds;
    }

    bool commitLoad(const std::shared_ptr<LoadAsyncContext>&, PreparedRequestResources&) override {
        return false;
    }

    void
    publishInsert(const PathKeys&, size_t, PreparedRequestResources&, std::vector<BlockIndicesType>& blocks) override {
        releaseRequestBlocks(blocks);
    }

    void releaseRequestBlocks(std::vector<BlockIndicesType>& blocks) override {
        released_request_blocks = blocks;
        blocks.clear();
    }

    void rollback(PreparedRequestResources&, std::vector<BlockIndicesType>& blocks) override {
        releaseRequestBlocks(blocks);
    }

    bool                          suffix_allocation_succeeds{false};
    size_t                        materialize_calls{0};
    std::vector<BlockIndicesType> released_request_blocks;
};

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

TEST(OnlineTreeSchedulerTest, AdmissionFailureReleasesTransferredRequestBlocks) {
    RequestBlockCleanupCache cache;
    OnlineTreeWorkloadConfig config;
    config.admission_allocation_retry_limit = 0;
    OnlineTreeScheduler scheduler(cache, config);

    OnlineRequestContext context;
    context.path          = {1};
    context.target_tokens = 1;

    EXPECT_EQ(scheduler.admit(context), OnlineTreeScheduler::AdmitResult::OK);
    EXPECT_EQ(context.state, OnlineRequestState::FINISHED);
    EXPECT_TRUE(context.request_blocks.empty());
    EXPECT_EQ(cache.materialize_calls, 1u);
    EXPECT_EQ(cache.released_request_blocks, (std::vector<BlockIndicesType>{{11, 12, 13}}));
}

TEST(OnlineTreeSchedulerTest, HeldRequestBlockPeakIncludesDeviceSources) {
    RequestBlockCleanupCache cache;
    cache.suffix_allocation_succeeds = true;
    OnlineTreeWorkloadConfig config;
    config.logical_concurrency              = 1;
    config.active_token_budget              = 1;
    config.forward_sleep_ms                 = 0;
    config.admission_allocation_retry_limit = 0;
    OnlineTreeScheduler scheduler(cache, config);

    OnlineRequestDescriptor descriptor;
    descriptor.path          = {1};
    descriptor.input_blocks  = 1;
    descriptor.target_tokens = 1;
    size_t  next_trace_index = 0;
    int64_t measured_ns      = 0;
    scheduler.runPhase({descriptor}, next_trace_index, std::chrono::milliseconds(10), measured_ns);

    EXPECT_EQ(scheduler.metrics().held_request_blocks_peak, 3u);
    EXPECT_EQ(cache.materialize_calls, 1u);
    EXPECT_EQ(cache.released_request_blocks, (std::vector<BlockIndicesType>{{11, 12, 13}}));
}

TEST(BlockTreeCacheAdapterTest, MaterializesReadyRefsOutsideMatchAndReleasesThem) {
    using namespace rtp_llm::block_tree_cache_test;
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 2;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    auto adapter = makeBlockTreeCacheAdapterForTest(*environment->cache);
    auto outcome = adapter->match(environment->keys);
    EXPECT_TRUE(outcome.request_blocks.empty());
    EXPECT_FALSE(outcome.matched_device_resources.empty());
    for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
        for (const BlockIdxType block : environment->blocksForDevicePool(pool_id)) {
            EXPECT_EQ(environment->device_pools[pool_id]->refCount(block), 2u);
        }
    }

    adapter->materializeRequestBlocks(outcome);
    EXPECT_TRUE(outcome.matched_device_resources.empty());
    size_t held_blocks = 0;
    for (const auto& blocks : outcome.request_blocks) {
        held_blocks += blocks.size();
    }
    EXPECT_EQ(held_blocks, options.path_length * environment->device_pools.size());

    adapter->releaseRequestBlocks(outcome.request_blocks);
    EXPECT_TRUE(outcome.request_blocks.empty());
    for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
        for (const BlockIdxType block : environment->blocksForDevicePool(pool_id)) {
            EXPECT_EQ(environment->device_pools[pool_id]->refCount(block), 1u);
        }
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

}  // namespace
}  // namespace rtp_llm::benchmark
