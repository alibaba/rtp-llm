#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

namespace rtp_llm {
namespace {

std::vector<GroupSetPtr> makeGroupSets(size_t count) {
    std::vector<GroupSetPtr> group_sets;
    group_sets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        group_sets.push_back(std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr));
    }
    block_tree_cache_test::prepareGroupSetsForTest(group_sets);
    return group_sets;
}

std::vector<GroupSetPtr> makeFullSwaGroupSets() {
    std::vector<GroupSetPtr> group_sets{
        std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            nullptr,
            nullptr),
        std::make_shared<SWAGroupSet>(128,
                                      64,
                                      std::vector<DeviceBlockPoolPtr>{
                                          block_tree_cache_test::makeStructuralDevicePool(1)},
                                      nullptr,
                                      nullptr),
    };
    block_tree_cache_test::prepareGroupSetsForTest(group_sets);
    return group_sets;
}

// Helper: create complete Device resources for every GroupSet on every node.
std::vector<std::vector<GroupSetResource>> make2DResources(int group_count, int path_len, BlockIdxType start_block) {
    std::vector<std::vector<GroupSetResource>> resources(static_cast<size_t>(path_len));
    for (int i = 0; i < path_len; ++i) {
        resources[i].resize(static_cast<size_t>(group_count));
        for (int group_set_id = 0; group_set_id < group_count; ++group_set_id) {
            resources[i][group_set_id].device_blocks = {static_cast<BlockIdxType>(start_block + i)};
        }
    }
    return resources;
}

// Helper: create 2D resources with one empty GroupSetResource per node.
std::vector<std::vector<GroupSetResource>> makeEmpty2DResources(int path_len) {
    return std::vector<std::vector<GroupSetResource>>(static_cast<size_t>(path_len), std::vector<GroupSetResource>(1));
}

TreeNode* insertAndGetNode(BlockTree&                                        tree,
                           const CacheKeysType&                              cache_keys,
                           const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTreeInsertResult result = tree.insertNode(cache_keys, resources, /*collect_path=*/false);
    return result.inserted_nodes.back();
}

TEST(BlockTreeTest, EmptyTreeFindReturnsEmpty) {
    BlockTree tree(makeGroupSets(1));
    auto      result = tree.findNode({100, 200, 300});
    EXPECT_TRUE(result.empty());
}

TEST(BlockTreeTest, OwnsReusableGroupLocations) {
    auto topology = block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(),
                                                                  block_transfer_engine_test::makeTestGroupBase(),
                                                                  block_transfer_engine_test::makeTestGroupBase()});
    std::vector<GroupSetPtr> group_sets{
        block_transfer_engine_test::makeTestGroupSet(0, topology, {0, 2}, {}),
        block_transfer_engine_test::makeTestGroupSet(1, topology, {1}, {}),
    };

    BlockTree tree(std::move(group_sets));

    EXPECT_EQ(tree.reusableGroupCount(), 3u);

    const auto* group_0 = tree.reusableGroupLocation(0);
    ASSERT_NE(group_0, nullptr);
    EXPECT_EQ(group_0->group_set_id, 0u);
    EXPECT_EQ(group_0->member_group_id, 0u);

    const auto* group_2 = tree.reusableGroupLocation(2);
    ASSERT_NE(group_2, nullptr);
    EXPECT_EQ(group_2->group_set_id, 0u);
    EXPECT_EQ(group_2->member_group_id, 1u);

    const auto* group_1 = tree.reusableGroupLocation(1);
    ASSERT_NE(group_1, nullptr);
    EXPECT_EQ(group_1->group_set_id, 1u);
    EXPECT_EQ(group_1->member_group_id, 0u);

    EXPECT_EQ(tree.reusableGroupLocation(3), nullptr);
}

TEST(BlockTreeTest, InsertSinglePath) {
    BlockTree     tree(makeGroupSets(1));
    CacheKeysType keys      = {100, 200, 300};
    auto          resources = make2DResources(1, 3, 42);

    TreeNode* leaf = insertAndGetNode(tree, keys, resources);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->cache_key, 300);
    EXPECT_EQ(leaf->group_set_resources[0].device_blocks[0], 44);  // start_block + 2
    EXPECT_EQ(tree.size(), 3u);  // 3 nodes created (not counting root)

    // Verify tree structure and per-node resources
    auto* root = tree.root();
    EXPECT_EQ(root->children.size(), 1u);
    auto* a = root->children.at(100);
    EXPECT_EQ(a->cache_key, 100);
    EXPECT_EQ(a->group_set_resources[0].device_blocks[0], 42);  // start_block + 0
    auto* b = a->children.at(200);
    EXPECT_EQ(b->cache_key, 200);
    EXPECT_EQ(b->group_set_resources[0].device_blocks[0], 43);  // start_block + 1
    auto* c = b->children.at(300);
    EXPECT_EQ(c->cache_key, 300);
    EXPECT_EQ(c, leaf);
}

TEST(BlockTreeTest, InsertForkPath) {
    BlockTree tree(makeGroupSets(1));

    // Insert root → 100 → 200 → 300
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 1), /*collect_path=*/false);
    // Insert root → 100 → 200 → 400 (fork at 200)
    tree.insertNode({100, 200, 400}, make2DResources(1, 3, 10), /*collect_path=*/false);
    // Insert root → 100 → 500 (fork at 100)
    tree.insertNode({100, 500}, make2DResources(1, 2, 20), /*collect_path=*/false);

    EXPECT_EQ(tree.size(), 5u);  // 100, 200, 300, 400, 500

    auto* root = tree.root();
    auto* n100 = root->children.at(100);
    EXPECT_EQ(n100->children.size(), 2u);  // 200, 500
    auto* n200 = n100->children.at(200);
    EXPECT_EQ(n200->children.size(), 2u);  // 300, 400
}

TEST(BlockTreeTest, FindExistingPath) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42), /*collect_path=*/false);

    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result.back()->cache_key, 300);
}

TEST(BlockTreeTest, FindPartialMatch) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42), /*collect_path=*/false);

    // Search for a longer path — only first 2 match
    auto result = tree.findNode({100, 200, 999});
    EXPECT_EQ(result.size(), 2u);
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result.back()->cache_key, 200);
}

TEST(BlockTreeTest, FindAllowsLoadingNode) {
    BlockTree tree(makeGroupSets(2));
    insertAndGetNode(tree, {100, 200, 300}, make2DResources(2, 3, 42));

    TreeNode* loading_node                              = tree.root()->children.at(100)->children.at(200);
    loading_node->group_set_resources[1].transfer_state = GroupSetTransferState::LOADING;

    const auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result.back(), tree.root()->children.at(100)->children.at(200)->children.at(300));
}

TEST(BlockTreeTest, FindTraversesBusyNodeAndItsDescendants) {
    // Transfer-state gating moved to the per-group MatchValidators: the tree
    // walk is purely topological and must not truncate at busy resources.
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        BlockTree tree(makeGroupSets(1));
        tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42), /*collect_path=*/false);

        TreeNode* busy_node                              = tree.root()->children.at(100)->children.at(200);
        busy_node->group_set_resources[0].transfer_state = state;

        const auto result = tree.findNode({100, 200, 300});
        ASSERT_EQ(result.size(), 3u);
        EXPECT_EQ(result[1], busy_node);
        EXPECT_EQ(result.back(), busy_node->children.at(300));
    }
}

TEST(BlockTreeTest, FindEmptyKeys) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100}, make2DResources(1, 1, 1), /*collect_path=*/false);

    auto result = tree.findNode({});
    EXPECT_TRUE(result.empty());
}

TEST(BlockTreeTest, RemoveLeafNode) {
    BlockTree tree(makeGroupSets(1));
    auto      resources = make2DResources(1, 3, 42);
    resources.back()[0].device_blocks.clear();
    tree.insertNode({100, 200, 300}, resources, /*collect_path=*/false);
    EXPECT_EQ(tree.size(), 3u);

    auto result = tree.findNode({100, 200, 300});
    ASSERT_FALSE(result.empty());
    TreeNode* surviving_ancestor = tree.removeNodeAndEmptyAncestors(result.back());
    EXPECT_EQ(surviving_ancestor->cache_key, 200);
    EXPECT_EQ(tree.size(), 2u);

    auto result2 = tree.findNode({100, 200, 300});
    EXPECT_EQ(result2.size(), 2u);
    EXPECT_EQ(result2.back()->cache_key, 200);
}

TEST(BlockTreeTest, IsRemovableRequiresLeafWithRemovableGroupResources) {
    BlockTree                                  tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(2));
    TreeNode* leaf = insertAndGetNode(tree, {100, 200}, resources);
    ASSERT_NE(leaf, nullptr);

    EXPECT_FALSE(tree.isRemovable(leaf->parent));
    EXPECT_TRUE(tree.isRemovable(leaf));

    leaf->group_set_resources[1].transfer_state = GroupSetTransferState::LOAD_PENDING;
    EXPECT_FALSE(tree.isRemovable(leaf));
}

TEST(BlockTreeTest, RemoveNodeAndEmptyAncestors) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* leaf = insertAndGetNode(tree, {100, 200, 300}, makeEmpty2DResources(3));
    EXPECT_EQ(tree.size(), 3u);

    TreeNode* surviving_ancestor = tree.removeNodeAndEmptyAncestors(leaf);
    EXPECT_EQ(surviving_ancestor, tree.root());
    EXPECT_EQ(tree.size(), 0u);
}

TEST(BlockTreeTest, RemovingNonTailNodeUpdatesMovedNodeIndex) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* node200 = insertAndGetNode(tree, {100, 200}, makeEmpty2DResources(2));
    TreeNode* node300 = insertAndGetNode(tree, {100, 300}, makeEmpty2DResources(2));
    TreeNode* node100 = node200->parent;

    ASSERT_NE(node100, nullptr);
    ASSERT_EQ(node100, node300->parent);
    EXPECT_EQ(node100->index, 0u);
    EXPECT_EQ(node200->index, 1u);
    EXPECT_EQ(node300->index, 2u);

    EXPECT_EQ(tree.removeNodeAndEmptyAncestors(node200), node100);

    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(node300->index, 1u);
    EXPECT_EQ(node100->children.at(300), node300);
}

TEST(BlockTreeTest, RemoveNodeAndEmptyAncestorsStopsAtData) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100}, make2DResources(1, 1, 10), /*collect_path=*/false);
    auto resources = make2DResources(1, 2, 20);
    resources[1][0].device_blocks.clear();
    TreeNode* leaf = insertAndGetNode(tree, {100, 200}, resources);

    TreeNode* surviving_ancestor = tree.removeNodeAndEmptyAncestors(leaf);

    auto result = tree.findNode({100});
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(surviving_ancestor, result.back());
    EXPECT_EQ(tree.size(), 1u);
    auto check = tree.findNode({100});
    EXPECT_EQ(check.size(), 1u);
}

TEST(BlockTreeTest, RemoveNodeAndEmptyAncestorsReturnsFirstSurvivorAfterPruning) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100}, make2DResources(1, 1, 10), /*collect_path=*/false);
    TreeNode* leaf = insertAndGetNode(tree, {100, 200, 300}, makeEmpty2DResources(3));
    ASSERT_NE(leaf, nullptr);

    TreeNode* surviving_ancestor = tree.removeNodeAndEmptyAncestors(leaf);

    auto node100 = tree.findNode({100});
    ASSERT_FALSE(node100.empty());
    EXPECT_EQ(surviving_ancestor, node100.back());
    EXPECT_EQ(surviving_ancestor->cache_key, 100);
    EXPECT_EQ(tree.size(), 1u);
}

TEST(BlockTreeTest, RepeatedInsertDoesNotDuplicate) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200}, make2DResources(1, 2, 1), /*collect_path=*/false);
    EXPECT_EQ(tree.size(), 2u);

    // Insert same path again — should reuse existing nodes
    tree.insertNode({100, 200}, make2DResources(1, 2, 50), /*collect_path=*/false);
    EXPECT_EQ(tree.size(), 2u);

    // After Bug 3 fix: existing nodes are NOT overwritten.
    // Only newly created nodes get group_set_resources assigned.
    auto result = tree.findNode({100, 200});
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result.back()->group_set_resources[0].device_blocks[0], 2);  // original value (1+1)
}

TEST(BlockTreeTest, InsertEmptyKeys) {
    BlockTree                   tree(makeGroupSets(1));
    const BlockTreeInsertResult result = tree.insertNode({}, {}, /*collect_path=*/false);
    EXPECT_TRUE(result.inserted_nodes.empty());
    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_EQ(result.accepted_resource_count, 0u);
    EXPECT_EQ(tree.size(), 0u);
}

TEST(BlockTreeTest, MultipleGroupSets) {
    BlockTree tree(makeGroupSets(3));  // 3 group sets

    // Create 2D resources for a single node with 3 group sets.
    std::vector<std::vector<GroupSetResource>> resources(1);
    resources[0].resize(3);
    resources[0][0].device_blocks = {10};
    resources[0][1].device_blocks = {20};
    resources[0][2].device_blocks = {30};

    TreeNode* leaf = insertAndGetNode(tree, {100}, resources);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->group_set_resources.size(), 3u);
    EXPECT_EQ(leaf->group_set_resources[0].device_blocks[0], 10);
    EXPECT_EQ(leaf->group_set_resources[1].device_blocks.size(), 1u);
    EXPECT_EQ(leaf->group_set_resources[2].device_blocks[0], 30);
}

// UT-1: Verify insertNode does not overwrite existing node's group_set_resources (Bug 3 fix)
TEST(BlockTreeTest, InsertDoesNotOverwriteExistingNodeResources) {
    BlockTree tree(makeGroupSets(1));

    // First insert: 100 -> 200, with device_blocks={42, 43}
    tree.insertNode({100, 200}, make2DResources(1, 2, 42), /*collect_path=*/false);

    // Second insert: 100 -> 200 -> 300, with device_blocks={99, 100, 101}
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 99), /*collect_path=*/false);

    // Verify: nodes 100 and 200 retain original values, only 300 gets new value
    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0]->group_set_resources[0].device_blocks[0], 42);   // 100 unchanged
    EXPECT_EQ(result[1]->group_set_resources[0].device_blocks[0], 43);   // 200 unchanged
    EXPECT_EQ(result[2]->group_set_resources[0].device_blocks[0], 101);  // 300 new (99+2)
}

TEST(BlockTreeTest, InsertResultContainsCompletePathForDuplicateInsert) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200}, make2DResources(1, 2, 10), /*collect_path=*/false);
    const std::vector<TreeNode*> existing_path = tree.findNode({100, 200});

    const BlockTreeInsertResult result = tree.insertNode({100, 200}, make2DResources(1, 2, 20), /*collect_path=*/true);

    EXPECT_TRUE(result.inserted_nodes.empty());
    EXPECT_TRUE(result.adopted_nodes.empty());
    ASSERT_EQ(result.path.size(), 2u);
    EXPECT_EQ(result.path[0], existing_path[0]);
    EXPECT_EQ(result.path[1], existing_path[1]);
}

TEST(BlockTreeTest, InsertDoesNotCollectPathWhenDisabled) {
    BlockTree tree(makeGroupSets(1));

    const BlockTreeInsertResult result = tree.insertNode({100, 200}, make2DResources(1, 2, 10), /*collect_path=*/false);

    EXPECT_TRUE(result.path.empty());
}

TEST(BlockTreeTest, InsertFillsOnlyCompleteEmptyIdleGroupsOnExistingNode) {
    BlockTree                                  tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {10};
    original[0][1].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);

    std::vector<std::vector<GroupSetResource>> replacement(1, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks = {20};
    replacement[0][1].device_blocks = {30};

    const BlockTreeInsertResult result = tree.insertNode({100}, replacement, /*collect_path=*/false);
    EXPECT_TRUE(result.inserted_nodes.empty());
    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].first, node);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{1}));
    EXPECT_EQ(result.accepted_resource_count, 1u);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{30}));
}

TEST(BlockTreeTest, InsertAggregatesAdoptedGroupSetsPerNode) {
    BlockTree                                  tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {NULL_BLOCK_IDX};
    original[0][1].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);

    std::vector<std::vector<GroupSetResource>> replacement(1, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks    = {20};
    replacement[0][1].device_blocks    = {30};
    const BlockTreeInsertResult result = tree.insertNode({100}, replacement, /*collect_path=*/false);

    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].first, node);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(result.accepted_resource_count, 2u);
}

TEST(BlockTreeTest, InsertSkipsBusyEmptyGroupOnExistingNode) {
    BlockTree                                  tree(makeGroupSets(1));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(1));
    original[0][0].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    const BlockTreeInsertResult result = tree.insertNode({100}, make2DResources(1, 1, 20), /*collect_path=*/false);
    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_EQ(result.accepted_resource_count, 0u);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{NULL_BLOCK_IDX}));
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
}

TEST(BlockTreeTest, InsertHardStopsAtBusyFullGroup) {
    BlockTree tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {NULL_BLOCK_IDX};
    original[0][1].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOADING;

    std::vector<std::vector<GroupSetResource>> replacement(2, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks    = {20};
    replacement[0][1].device_blocks    = {30};
    replacement[1][0].device_blocks    = {21};
    replacement[1][1].device_blocks    = {31};
    const BlockTreeInsertResult result = tree.insertNode({100, 200}, replacement, /*collect_path=*/false);

    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_TRUE(result.inserted_nodes.empty());
    EXPECT_TRUE(node->children.empty());
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{NULL_BLOCK_IDX}));
    EXPECT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{NULL_BLOCK_IDX}));
}

TEST(BlockTreeTest, InsertHardStopsAtExistingHostFullNode) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* host_node = insertAndGetNode(tree, {100, 200}, make2DResources(1, 2, 10));
    ASSERT_NE(host_node, nullptr);

    GroupSetResource& host_resource = host_node->group_set_resources[0];
    const MultiNodeResource device_resource{0, Tier::DEVICE, {{host_node, host_resource.device_blocks}}};
    tree.groupSets()[0]->unmapDeviceBlocksFromTreeNode(device_resource);
    tree.groupSets()[0]->unreferenceBlocks(device_resource, BlockRefType::BLOCK_CACHE);
    host_resource.evictFromTier(Tier::DEVICE);
    host_resource.host_block = 7;

    const BlockTreeInsertResult result =
        tree.insertNode({100, 200, 300}, make2DResources(1, 3, 20), /*collect_path=*/true);

    ASSERT_EQ(result.path.size(), 2u);
    EXPECT_EQ(result.path[0], host_node->parent);
    EXPECT_EQ(result.path[1], host_node);
    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_TRUE(result.inserted_nodes.empty());
    EXPECT_TRUE(host_node->children.empty());
    EXPECT_EQ(host_resource.getTopTier(), Tier::HOST);
    EXPECT_EQ(host_resource.host_block, 7);
}

TEST(BlockTreeTest, InsertAdoptsIdleEmptyFullNodeBeforeAddingSuffix) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* empty_node = insertAndGetNode(tree, {100, 200}, make2DResources(1, 2, 10));
    ASSERT_NE(empty_node, nullptr);

    GroupSetResource& empty_resource = empty_node->group_set_resources[0];
    const MultiNodeResource device_resource{0, Tier::DEVICE, {{empty_node, empty_resource.device_blocks}}};
    tree.groupSets()[0]->unmapDeviceBlocksFromTreeNode(device_resource);
    tree.groupSets()[0]->unreferenceBlocks(device_resource, BlockRefType::BLOCK_CACHE);
    empty_resource.evictFromTier(Tier::DEVICE);

    const BlockTreeInsertResult result =
        tree.insertNode({100, 200, 300}, make2DResources(1, 3, 20), /*collect_path=*/false);

    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].first, empty_node);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{0}));
    ASSERT_EQ(result.inserted_nodes.size(), 1u);
    EXPECT_EQ(result.inserted_nodes[0]->cache_key, 300);
    EXPECT_EQ(empty_resource.device_blocks, (BlockIndicesType{21}));
}

TEST(BlockTreeTest, InsertAcceptsHostAndDiskIncomingResources) {
    for (Tier tier : {Tier::HOST, Tier::DISK}) {
        BlockTree       tree(makeGroupSets(1));
        GroupSetResource resource;
        if (tier == Tier::HOST) {
            resource.host_block = 7;
        } else {
            resource.disk_slot = 8;
        }

        const BlockTreeInsertResult result = tree.insertNode({100}, {{resource}}, /*collect_path=*/false);
        ASSERT_EQ(result.inserted_nodes.size(), 1u);
        EXPECT_EQ(result.inserted_nodes[0]->group_set_resources[0].getTopTier(), tier);
        EXPECT_EQ(result.accepted_resource_count, 1u);
    }
}

TEST(BlockTreeTest, LowerTierInsertDoesNotUseDeviceHardStop) {
    BlockTree       tree(makeGroupSets(1));
    GroupSetResource existing_host;
    existing_host.host_block = 7;
    TreeNode* node = insertAndGetNode(tree, {100}, {{existing_host}});
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOADING;

    GroupSetResource incoming_parent;
    incoming_parent.host_block = 8;
    GroupSetResource incoming_child;
    incoming_child.host_block = 9;
    const BlockTreeInsertResult result =
        tree.insertNode({100, 200}, {{incoming_parent}, {incoming_child}}, /*collect_path=*/false);

    ASSERT_EQ(result.inserted_nodes.size(), 1u);
    EXPECT_EQ(result.inserted_nodes[0]->cache_key, 200);
    EXPECT_EQ(result.inserted_nodes[0]->group_set_resources[0].getTopTier(), Tier::HOST);
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
}

TEST(BlockTreeTest, InsertReusesExistingDeviceFullNodeWithoutOverwritingIt) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100}, make2DResources(1, 1, 10), /*collect_path=*/false);

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {30};
    resources[1][0].device_blocks = {20};

    const BlockTreeInsertResult result = tree.insertNode({100, 200}, resources, /*collect_path=*/false);

    ASSERT_EQ(result.inserted_nodes.size(), 1u);
    EXPECT_EQ(result.inserted_nodes[0]->cache_key, 200);
    EXPECT_EQ(result.accepted_resource_count, 1u);
    const auto path = tree.findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
}

TEST(BlockTreeTest, BusySwaResourceDoesNotGateFullSuffixInsertion) {
    BlockTree tree(makeFullSwaGroupSets());
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {10};
    original[0][1].device_blocks = {20};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);
    node->group_set_resources[1].transfer_state = GroupSetTransferState::DEMOTING;

    std::vector<std::vector<GroupSetResource>> replacement(2, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks = {30};
    replacement[0][1].device_blocks = {40};
    replacement[1][0].device_blocks = {31};
    replacement[1][1].device_blocks = {41};

    const BlockTreeInsertResult result = tree.insertNode({100, 200}, replacement, /*collect_path=*/false);

    ASSERT_EQ(result.inserted_nodes.size(), 1u);
    EXPECT_EQ(result.inserted_nodes[0]->cache_key, 200);
    EXPECT_EQ(node->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
}

TEST(BlockTreeTest, RemoveNodeAndEmptyAncestorsStopsAtBusyEmptyNode) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* node = insertAndGetNode(tree, {100}, makeEmpty2DResources(1));
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOAD_PENDING;

    EXPECT_EQ(tree.removeNodeAndEmptyAncestors(node), node);
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(node->parent, tree.root());
}

}  // namespace
}  // namespace rtp_llm
