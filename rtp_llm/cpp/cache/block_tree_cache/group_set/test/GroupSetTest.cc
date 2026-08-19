#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;

GroupBase makeGroupBase(std::vector<int> layer_ids, bool reusable = true) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = reusable;
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), 64, 16);
}

TEST(GroupSetTest, StoresOrderedTopologyMembershipAndLogicalPayload) {
    const auto topology = makeTestTopology({makeGroupBase({0, 1}), makeGroupBase({0})});
    const auto pool_a   = makeTestDevicePool({{80, 16}, {96, 24}}, 4, "group_set_a");
    const auto pool_b   = makeTestDevicePool({{128, 32}}, 4, "group_set_b");

    const auto group_set = makeTestGroupSet(3, topology, {1, 0}, {pool_b, pool_a});

    EXPECT_EQ(group_set->groupSetId(), 3u);
    EXPECT_EQ(group_set->groupIds(), (std::vector<size_t>{1, 0}));
    EXPECT_EQ(&group_set->groupAt(0), &topology->groupById(1));
    EXPECT_EQ(&group_set->groupAt(1), &topology->groupById(0));
    EXPECT_EQ(group_set->devicePools(), (std::vector<DeviceBlockPoolPtr>{pool_b, pool_a}));
    EXPECT_EQ(group_set->payloadBytes(), 3u * (64u + 16u));
    EXPECT_EQ(group_set->groupType(), CacheGroupType::FULL);
}

TEST(GroupSetTest, KeepsTopologyAliveAfterCallerReleasesOwnership) {
    auto                               topology      = makeTestTopology({makeGroupBase({0})});
    std::weak_ptr<const CacheTopology> weak_topology = topology;
    const auto                         pool          = makeTestDevicePool({{64, 16}}, 4, "group_set_topology_lifetime");
    auto                               group         = makeTestGroupSet(0, topology, {0}, {pool});
    const GroupBase*                   expected_group = &topology->groupById(0);

    topology.reset();

    EXPECT_FALSE(weak_topology.expired());
    EXPECT_EQ(&group->groupAt(0), expected_group);
}

TEST(GroupSetTest, MapsOneLogicalDeviceResourceAcrossMembers) {
    const auto topology = makeTestTopology({makeGroupBase({0}), makeGroupBase({1})});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 2, "group_set_binding_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 2, "group_set_binding_b");
    auto       group    = makeTestGroupSet(0, topology, {0, 1}, {pool_a, pool_b});

    const auto block_a = pool_a->malloc();
    const auto block_b = pool_b->malloc();
    ASSERT_TRUE(block_a.has_value());
    ASSERT_TRUE(block_b.has_value());
    TreeNode node;
    node.group_set_resources.resize(1);
    node.group_set_resources[0].setBlocks(Tier::DEVICE, {*block_a, *block_b});
    const MultiNodeResource resource{0, Tier::DEVICE, {{&node, {*block_a, *block_b}}}};

    EXPECT_TRUE(group->areBlockToNodeMapsEmpty());
    group->mapDeviceBlocksToTreeNode(resource);
    EXPECT_EQ(group->findTreeNodeByDeviceBlock(/*member_group_id=*/0, *block_a), &node);
    EXPECT_EQ(group->findTreeNodeByDeviceBlock(/*member_group_id=*/1, *block_b), &node);
    EXPECT_FALSE(group->areBlockToNodeMapsEmpty());

    group->unmapDeviceBlocksFromTreeNode(resource);
    node.group_set_resources[0].evictFromTier(Tier::DEVICE);
    EXPECT_EQ(group->findTreeNodeByDeviceBlock(/*member_group_id=*/0, *block_a), nullptr);
    EXPECT_EQ(group->findTreeNodeByDeviceBlock(/*member_group_id=*/1, *block_b), nullptr);
    EXPECT_TRUE(group->areBlockToNodeMapsEmpty());

    pool_a->free(*block_a);
    pool_b->free(*block_b);
}

TEST(GroupSetTest, ReusedDeviceBlockIdCanBeRemapped) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 1, "group_set_reused_block");
    auto       group    = makeTestGroupSet(0, topology, {0}, {pool});

    const auto first_block = pool->malloc();
    ASSERT_TRUE(first_block.has_value());
    TreeNode first_node;
    first_node.group_set_resources.resize(1);
    first_node.group_set_resources[0].setBlocks(Tier::DEVICE, {*first_block});
    const MultiNodeResource first_resource{0, Tier::DEVICE, {{&first_node, {*first_block}}}};
    group->mapDeviceBlocksToTreeNode(first_resource);
    group->unmapDeviceBlocksFromTreeNode(first_resource);
    first_node.group_set_resources[0].evictFromTier(Tier::DEVICE);
    pool->free(*first_block);

    const auto second_block = pool->malloc();
    ASSERT_TRUE(second_block.has_value());
    ASSERT_EQ(*second_block, *first_block);
    TreeNode second_node;
    second_node.group_set_resources.resize(1);
    second_node.group_set_resources[0].setBlocks(Tier::DEVICE, {*second_block});
    const MultiNodeResource second_resource{0, Tier::DEVICE, {{&second_node, {*second_block}}}};
    group->mapDeviceBlocksToTreeNode(second_resource);

    EXPECT_EQ(group->findTreeNodeByDeviceBlock(0, *second_block), &second_node);

    group->unmapDeviceBlocksFromTreeNode(second_resource);
    second_node.group_set_resources[0].evictFromTier(Tier::DEVICE);
    pool->free(*second_block);
}

TEST(GroupSetTest, DeviceBlockMappingDoesNotChangeReferenceCounts) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 2, "group_set_mapping_ref_count");
    auto       group    = makeTestGroupSet(0, topology, {0}, {pool});
    const auto block    = pool->malloc();
    ASSERT_TRUE(block.has_value());
    pool->incRef(*block, BlockRefType::REQUEST);

    TreeNode node;
    node.group_set_resources.resize(1);
    node.group_set_resources[0].setBlocks(Tier::DEVICE, {*block});
    const MultiNodeResource resource{0, Tier::DEVICE, {{&node, {*block}}}};
    group->mapDeviceBlocksToTreeNode(resource);
    EXPECT_EQ(pool->refCount(*block), 1u);

    group->unmapDeviceBlocksFromTreeNode(resource);
    EXPECT_EQ(pool->refCount(*block), 1u);
    node.group_set_resources[0].evictFromTier(Tier::DEVICE);
    pool->decRef(*block, BlockRefType::REQUEST);
}

}  // namespace
}  // namespace rtp_llm
