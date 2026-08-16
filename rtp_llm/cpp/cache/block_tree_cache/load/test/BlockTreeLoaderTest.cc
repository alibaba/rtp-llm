#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::FullSWAEnvironment;
using block_tree_cache_test::FullSWAEnvironmentOptions;
using block_tree_cache_test::cudaAvailable;
using block_tree_cache_test::releaseDeviceBlocksAndNotify;

TEST(BlockTreeLoaderTest, HostLoadInstallsAllocatorBoundDeviceTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 1u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::HOST), 1u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::DISK), 0u);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks, BlockRefType::REQUEST);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    std::shared_ptr<AsyncContext> context = load_context;
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    environment->expectPayloads();

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, DiskTransferFailureInstallsNoLoadTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    environment->demoteAll(Tier::HOST);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DISK));

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(true);
    environment->scripted_per_rank_transfer_engine->enqueue(false);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 2u);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const size_t group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups[group_set_id]->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            pool->incRef(blocks, BlockRefType::REQUEST);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    load_context->waitDone();
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedBatchCount(), 2u);

    const auto resources = environment->resourcesForPathNode(0);
    ASSERT_EQ(resources.size(), 2u);
    for (const GroupSetResource& resource : resources) {
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (const auto& [pool, block] : request_targets) {
        EXPECT_EQ(pool->refCount(block), 1u);
        pool->decRef(block, BlockRefType::REQUEST);
        EXPECT_EQ(pool->freeBlocksNum(), options.usable_device_blocks);
    }

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, LoadStateMachineRejectsDuplicateTransitionAndRestoresSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    constexpr size_t  group_set_id = 0;
    GroupSetResource& resource = find_result.back()->group_set_resources[group_set_id];

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING));
    environment->cache->evictor_.refreshCandidate(find_result.back(), group_set_id);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    environment->cache->evictor_.refreshCandidate(find_result.back(), group_set_id);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, ChangeTransferStateDoesNotOverwriteForeignTransferState) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    GroupSetResource& resource = find_result.back()->group_set_resources[0];
    resource.transfer_state    = GroupSetTransferState::DEMOTING;
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), 0, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);

    resource.transfer_state = GroupSetTransferState::IDLE;
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

}  // namespace
}  // namespace rtp_llm
