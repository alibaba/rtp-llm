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
    ASSERT_TRUE(environment->allSlotsAtTier(Tier::HOST));

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_blocks, 0u);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_blocks, 2u);
    EXPECT_EQ(result.host_load_blocks, 2u);
    EXPECT_EQ(result.disk_load_blocks, 0u);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t item_index = 0; item_index < result.load_ticket->itemCount(); ++item_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = result.load_ticket->groupSetId(item_index);
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks, BlockRefType::REQUEST);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        ASSERT_TRUE(result.load_ticket->bindTargetDeviceBlocks(item_index, std::move(targets)));
    }

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    environment->expectPayloads();

    result.load_ticket.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        pool->decRef(block, BlockRefType::REQUEST);
    }
    environment->cache->onBlocksReleased();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, LoadStateMachineRejectsDuplicateBeginAndRestoresSource) {
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
    BlockTreeFindResult find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_NE(find_result.matched_node, nullptr);

    constexpr size_t                group_set_id  = 0;
    const GroupSetPtr&              group         = environment->groups.at(group_set_id);
    GroupSetResource&               slot          = find_result.matched_node->group_set_resources[group_set_id];
    const std::vector<BlockIdxType> source_blocks = group->getBlocks(slot, Tier::HOST);

    ASSERT_TRUE(
        environment->cache->loader_.reserveLoad(find_result.matched_node, group_set_id, Tier::HOST, source_blocks));
    EXPECT_EQ(slot.transfer_state, GroupSetTransferState::LOAD_PENDING);
    ASSERT_TRUE(environment->cache->loader_.beginLoad(find_result.matched_node, group_set_id, Tier::HOST));
    EXPECT_EQ(slot.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(environment->cache->loader_.beginLoad(find_result.matched_node, group_set_id, Tier::HOST));
    EXPECT_EQ(slot.transfer_state, GroupSetTransferState::LOADING);

    ASSERT_TRUE(environment->cache->loader_.finishLoad(find_result.matched_node, group_set_id, Tier::HOST, false));
    EXPECT_EQ(slot.transfer_state, GroupSetTransferState::IDLE);

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, FinishLoadDoesNotOverwriteForeignTransferState) {
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
    BlockTreeFindResult find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_NE(find_result.matched_node, nullptr);

    GroupSetResource& slot = find_result.matched_node->group_set_resources[0];
    slot.transfer_state    = GroupSetTransferState::DEMOTING;
    EXPECT_FALSE(environment->cache->loader_.finishLoad(find_result.matched_node, 0, Tier::HOST, false));
    EXPECT_EQ(slot.transfer_state, GroupSetTransferState::DEMOTING);

    slot.transfer_state = GroupSetTransferState::IDLE;
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

}  // namespace
}  // namespace rtp_llm
