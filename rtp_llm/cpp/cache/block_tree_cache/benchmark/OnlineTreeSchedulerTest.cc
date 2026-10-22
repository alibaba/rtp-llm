#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>

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
        outcome.request_blocks     = {{11, 12}, {}, {13}};
        outcome.request_group_tags = {"swa", "empty", "full"};
    }

    bool allocateLoadTargets(const MatchOutcome&, PreparedRequestResources&) override {
        return true;
    }

    bool allocateSuffixBlocks(size_t, PreparedRequestResources& out) override {
        if (hold_prepared_on_failure) {
            out.suffix_blocks    = {{99}};
            out.suffix_allocated = true;
        }
        return suffix_allocation_succeeds;
    }

    bool commitLoad(const std::shared_ptr<LoadAsyncContext>&, PreparedRequestResources&) override {
        return false;
    }

    void publishInsert(const PathKeys&,
                       size_t,
                       PreparedRequestResources&,
                       std::vector<BlockIndicesType>& blocks,
                       std::vector<std::string>&      tags) override {
        releaseRequestBlocks(blocks, tags);
    }

    void releaseRequestBlocks(std::vector<BlockIndicesType>& blocks, std::vector<std::string>& tags) override {
        released_request_blocks = blocks;
        released_request_group_tags = tags;
        blocks.clear();
        tags.clear();
    }

    void rollback(PreparedRequestResources&      prepared,
                  std::vector<BlockIndicesType>& blocks,
                  std::vector<std::string>&      tags) override {
        ++rollback_calls;
        prepared = PreparedRequestResources{};
        releaseRequestBlocks(blocks, tags);
    }

    bool                          suffix_allocation_succeeds{false};
    bool                          hold_prepared_on_failure{false};
    size_t                        rollback_calls{0};
    size_t                        materialize_calls{0};
    std::vector<BlockIndicesType> released_request_blocks;
    std::vector<std::string>      released_request_group_tags;
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
    EXPECT_TRUE(context.request_group_tags.empty());
    EXPECT_EQ(cache.materialize_calls, 1u);
    EXPECT_EQ(cache.released_request_group_tags, (std::vector<std::string>{"swa", "empty", "full"}));
    EXPECT_EQ(cache.released_request_blocks, (std::vector<BlockIndicesType>{{11, 12}, {}, {13}}));
}

TEST(OnlineTreeSchedulerTest, AdmissionRollbackKeepsRequestTagsPairedWithEmptyRows) {
    RequestBlockCleanupCache cache;
    cache.hold_prepared_on_failure = true;
    OnlineTreeWorkloadConfig config;
    config.admission_allocation_retry_limit = 0;
    OnlineTreeScheduler  scheduler(cache, config);
    OnlineRequestContext context;
    context.path          = {1};
    context.target_tokens = 1;

    EXPECT_EQ(scheduler.admit(context), OnlineTreeScheduler::AdmitResult::OK);
    EXPECT_EQ(cache.rollback_calls, 1u);
    EXPECT_EQ(cache.released_request_group_tags, (std::vector<std::string>{"swa", "empty", "full"}));
    EXPECT_EQ(cache.released_request_blocks, (std::vector<BlockIndicesType>{{11, 12}, {}, {13}}));
    EXPECT_TRUE(context.request_blocks.empty());
    EXPECT_TRUE(context.request_group_tags.empty());
    EXPECT_FALSE(context.prepared.holdsBlocks());
    EXPECT_EQ(context.state, OnlineRequestState::FINISHED);
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
    EXPECT_EQ(cache.released_request_group_tags, (std::vector<std::string>{"swa", "empty", "full"}));
    EXPECT_EQ(cache.released_request_blocks, (std::vector<BlockIndicesType>{{11, 12}, {}, {13}}));
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
    EXPECT_TRUE(outcome.request_group_tags.empty());
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

    std::reverse(outcome.request_blocks.begin(), outcome.request_blocks.end());
    std::reverse(outcome.request_group_tags.begin(), outcome.request_group_tags.end());
    adapter->releaseRequestBlocks(outcome.request_blocks, outcome.request_group_tags);
    EXPECT_TRUE(outcome.request_blocks.empty());
    EXPECT_TRUE(outcome.request_group_tags.empty());
    for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
        for (const BlockIdxType block : environment->blocksForDevicePool(pool_id)) {
            EXPECT_EQ(environment->device_pools[pool_id]->refCount(block), 1u);
        }
    }
    // A subset row is identified by its tag, even when its pool is not row zero.
    const auto& last_group = environment->groups.back();
    const auto& last_pool  = last_group->devicePools().back();
    auto        extra      = last_pool->malloc(1);
    ASSERT_TRUE(extra.has_value());
    last_pool->incRef(*extra);
    std::vector<BlockIndicesType> subset_blocks = {{extra->front()}};
    std::vector<std::string>      subset_tags   = {last_group->groupTags().back()};
    EXPECT_EQ(last_pool->refCount(extra->front()), 1u);
    adapter->releaseRequestBlocks(subset_blocks, subset_tags);
    EXPECT_FALSE(last_pool->isAllocated(extra->front()));
    EXPECT_TRUE(subset_blocks.empty());
    EXPECT_TRUE(subset_tags.empty());
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeCacheAdapterTest, InvalidRequestIdentityDoesNotPartiallyReleaseLedger) {
    using namespace rtp_llm::block_tree_cache_test;
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    auto adapter = makeBlockTreeCacheAdapterForTest(*environment->cache);

    const auto& first_pool  = environment->device_pools[0];
    const auto& second_pool = environment->device_pools[1];
    auto        first       = first_pool->malloc(1);
    auto        second      = second_pool->malloc(1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    first_pool->incRef(*first);
    second_pool->incRef(*second);

    std::vector<BlockIndicesType> request_blocks  = {{first->front()}, {second->front()}};
    const auto                    original_blocks = request_blocks;
    std::vector<std::string>      request_tags    = {"group0", "unknown"};
    const auto                    unknown_tags    = request_tags;

    EXPECT_THROW(adapter->releaseRequestBlocks(request_blocks, request_tags), std::runtime_error);
    EXPECT_EQ(request_blocks, original_blocks);
    EXPECT_EQ(request_tags, unknown_tags);
    EXPECT_EQ(first_pool->refCount(first->front()), 1u);
    EXPECT_EQ(second_pool->refCount(second->front()), 1u);

    auto prepared_block = first_pool->malloc(1);
    ASSERT_TRUE(prepared_block.has_value());
    first_pool->incRef(*prepared_block);
    PreparedRequestResources prepared;
    prepared.suffix_blocks              = {{prepared_block->front()}, {}};
    prepared.suffix_allocated           = true;
    const auto original_prepared_blocks = prepared.suffix_blocks;
    request_tags                        = {"group0", "group0"};
    const auto duplicate_tags           = request_tags;

    EXPECT_THROW(adapter->rollback(prepared, request_blocks, request_tags), std::runtime_error);
    EXPECT_TRUE(prepared.holdsBlocks());
    EXPECT_EQ(prepared.suffix_blocks, original_prepared_blocks);
    EXPECT_EQ(request_blocks, original_blocks);
    EXPECT_EQ(request_tags, duplicate_tags);
    EXPECT_EQ(first_pool->refCount(first->front()), 1u);
    EXPECT_EQ(second_pool->refCount(second->front()), 1u);
    EXPECT_EQ(first_pool->refCount(prepared_block->front()), 1u);

    request_tags = {"group0", "group1"};
    adapter->rollback(prepared, request_blocks, request_tags);
    EXPECT_FALSE(prepared.holdsBlocks());
    EXPECT_TRUE(request_blocks.empty());
    EXPECT_TRUE(request_tags.empty());
    EXPECT_FALSE(first_pool->isAllocated(first->front()));
    EXPECT_FALSE(second_pool->isAllocated(second->front()));
    EXPECT_FALSE(first_pool->isAllocated(prepared_block->front()));
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BenchmarkFixtureTest, SwaWindowUsesTokenGeometry) {
    // Cover windows spanning two blocks and fitting within a single block.
    for (const size_t tokens_per_block : {128u, 1024u}) {
        SCOPED_TRACE(tokens_per_block);
        const auto topology = BenchmarkFixture::createTopology(
            {{"swa", rtp_llm::CacheGroupType::SWA}}, {64}, tokens_per_block, {}, {129});
        auto group =
            BenchmarkFixture::createSWAGroupSet({}, nullptr, nullptr, 0, topology, {"swa"}, 129, tokens_per_block);
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
