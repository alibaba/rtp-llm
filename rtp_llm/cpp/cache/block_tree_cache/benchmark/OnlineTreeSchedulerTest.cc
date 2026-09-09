#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <chrono>
#include <set>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/OnlineTreeScheduler.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

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

TEST(BenchmarkFixtureTest, SwaWindowUsesTokenGeometry) {
    // Cover windows spanning two blocks and fitting within a single block.
    for (const size_t tokens_per_block : {128u, 1024u}) {
        SCOPED_TRACE(tokens_per_block);
        const auto topology = BenchmarkFixture::createTopology(
            {{"swa", rtp_llm::CacheGroupType::SWA}}, {64}, tokens_per_block, {}, {129});
        auto group = BenchmarkFixture::createSWAGroupSet({}, nullptr, nullptr, 0, topology, {0}, 129, tokens_per_block);
        const size_t window_blocks = tokens_per_block == 128 ? 2 : 1;
        EXPECT_EQ(group->computeReuseBlockCount(10), window_blocks);
        auto validator = group->createMatchValidator();
        EXPECT_FALSE(validator->validate(GroupSetResource{}));
        GroupSetResource resource;
        resource.host_block = 1;
        for (size_t block = 1; block <= window_blocks; ++block) {
            EXPECT_EQ(validator->validate(resource), block == window_blocks);
        }
    }
}

}  // namespace
}  // namespace rtp_llm::benchmark
