#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::FullSWAEnvironment;
using block_tree_cache_test::FullSWAEnvironmentOptions;
using block_tree_cache_test::cudaAvailable;
using block_tree_cache_test::insertGroupSetResources;
using block_tree_cache_test::makeHostPool;
using block_tree_cache_test::makeStructuralDevicePool;
using block_tree_cache_test::releaseDeviceBlocks;

TEST(BlockTreeLoaderTest, HostLoadInstallsAllocatorBoundDeviceTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 2;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::HOST), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::DISK), 0u);

    for (const TransferDescriptor& desc : load_context->loadDescs()) {
        const auto source_pool = environment->groups.at(desc.group_set_id)->hostPool();
        ASSERT_NE(source_pool, nullptr);
        for (BlockIdxType block : desc.source_blocks) {
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
        }
    }
    for (const auto& source_pool : environment->host_pools) {
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::CACHE), options.path_length);
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::LOAD), options.path_length);
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
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
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    environment->expectPayloads();

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, HostOnlyPolicyReadsHostCopyWhenDeviceCopyAlsoExists) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 2;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    std::vector<std::vector<GroupSetResource>> source_resources(
        options.path_length, std::vector<GroupSetResource>(environment->groups.size()));
    for (size_t path_index = 0; path_index < options.path_length; ++path_index) {
        for (size_t group_set_id = 0; group_set_id < environment->groups.size(); ++group_set_id) {
            source_resources[path_index][group_set_id].device_blocks =
                environment->request_blocks[group_set_id][path_index];
        }
    }
    environment->cache->insert(environment->keys, source_resources, Tier::HOST);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));

    BlockTreeMatchPolicy policy;
    policy.enable_device = false;
    policy.enable_host   = true;
    policy.enable_disk   = false;
    policy.enable_remote = false;

    // Rolling back a HOST-only match must re-admit the resource's real top
    // (DEVICE) candidate, not create a stale HOST-heap entry.
    BlockTreeMatchResult aborted_result  = environment->cache->match(environment->keys, policy);
    auto                 aborted_context = std::dynamic_pointer_cast<LoadAsyncContext>(aborted_result.async_context);
    ASSERT_NE(aborted_context, nullptr);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 0u);
    EXPECT_TRUE(environment->cache->abortPendingLoad(aborted_context));
    aborted_result.async_context.reset();
    aborted_context.reset();
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);

    BlockTreeMatchResult result = environment->cache->match(environment->keys, policy);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), options.path_length);
    EXPECT_EQ(load_context->matchedBlocks(Tier::HOST), options.path_length);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& desc = load_context->loadDescs()[desc_index];
        EXPECT_EQ(desc.source_tier, Tier::HOST);
        EXPECT_FALSE(desc.install_target_in_cache);
        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    load_context->waitDone();
    ASSERT_TRUE(load_context->success());
    for (size_t path_index = 0; path_index < options.path_length; ++path_index) {
        for (const GroupSetResource& resource : environment->resourcesForPathNode(path_index)) {
            EXPECT_TRUE(resource.hasTier(Tier::DEVICE));
            EXPECT_TRUE(resource.hasTier(Tier::HOST));
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        }
    }
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);
    for (const auto& source_pool : environment->host_pools) {
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::CACHE), options.path_length);
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    }

    result.async_context.reset();
    load_context.reset();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->releaseRequestRefs();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, MatchRefreshesOnlyReusedSuffixForEachGroup) {
    constexpr size_t path_length = 4;
    auto             full        = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(0)}, makeHostPool(1, path_length), nullptr);
    auto swa    = std::make_shared<SWAGroupSet>(/*sliding_window_size=*/2,
                                             /*seq_size_per_block=*/1,
                                             std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(1)},
                                             makeHostPool(1, path_length),
                                             nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(2)}, makeHostPool(1, path_length), nullptr);
    std::vector<GroupSetPtr> groups = {full, swa, linear};

    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_host_cache   = true;
    config.enable_disk_cache   = false;
    auto cache                 = block_tree_cache_test::makeBlockTreeCacheForTest(groups, config);
    ASSERT_NE(cache, nullptr);

    const CacheKeysType                        keys = {100, 200, 300, 400};
    std::vector<std::vector<GroupSetResource>> resources(path_length, std::vector<GroupSetResource>(groups.size()));
    for (size_t path_index = 0; path_index < path_length; ++path_index) {
        for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
            resources[path_index][group_set_id].host_block =
                groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_NE(resources[path_index][group_set_id].host_block, NULL_BLOCK_IDX);
        }
    }
    ASSERT_TRUE(insertGroupSetResources(*cache, keys, resources));

    const std::vector<TreeNode*> path = cache->tree()->findNode(keys);
    ASSERT_EQ(path.size(), path_length);
    std::vector<std::vector<uint64_t>> access_before(path.size());
    for (size_t node_index = 0; node_index < path.size(); ++node_index) {
        access_before[node_index].reserve(groups.size());
        for (const GroupSetResource& resource : path[node_index]->group_set_resources) {
            EXPECT_EQ(resource.candidate_meta.hit_count, 0u);
            access_before[node_index].push_back(resource.candidate_meta.last_access_seq);
        }
    }

    BlockTreeMatchResult result = cache->match(keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), path_length);

    const std::vector<size_t> reused_suffix_starts = {0, 2, 3};
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        for (size_t node_index = 0; node_index < path.size(); ++node_index) {
            const CandidateMeta& meta = path[node_index]->group_set_resources[group_set_id].candidate_meta;
            if (node_index < reused_suffix_starts[group_set_id]) {
                EXPECT_EQ(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.hit_count, 0u);
            } else {
                EXPECT_GT(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.hit_count, 1u);
            }
        }
    }

    result.async_context.reset();
    load_context.reset();
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

    BlockTreeMatchResult result       = environment->cache->match(environment->keys);
    auto                 load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 2u);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups[group_set_id]->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            pool->incRef(blocks);
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
        pool->decRef(block);
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
    GroupSetResource& resource     = find_result.back()->group_set_resources[group_set_id];

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING));
    environment->cache->evictor_.suspendCandidate(find_result.back(), group_set_id, Tier::HOST);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    environment->cache->evictor_.admitCandidate(find_result.back(), group_set_id, Tier::HOST);
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
