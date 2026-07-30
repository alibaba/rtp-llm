#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

namespace rtp_llm {
namespace {

std::vector<GroupSetPtr> makeGroupSets(size_t count) {
    std::vector<GroupSetPtr> group_sets;
    group_sets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        group_sets.push_back(std::make_shared<FullGroupSet>());
    }
    block_tree_cache_test::prepareGroupSetsForTest(group_sets);
    return group_sets;
}

// Helper: create 2D resources — each node gets one GroupSetResource with incrementing block_idx.
// resources[i][0].device_blocks = {start_block + i}
std::vector<std::vector<GroupSetResource>> make2DResources(int group_count, int path_len, BlockIdxType start_block) {
    std::vector<std::vector<GroupSetResource>> resources(static_cast<size_t>(path_len));
    for (int i = 0; i < path_len; ++i) {
        resources[i].resize(static_cast<size_t>(group_count));
        resources[i][0].device_blocks = {static_cast<BlockIdxType>(start_block + i)};
    }
    return resources;
}

// Helper: create 2D resources with one empty GroupSetResource per node.
std::vector<std::vector<GroupSetResource>> makeEmpty2DResources(int path_len) {
    return std::vector<std::vector<GroupSetResource>>(static_cast<size_t>(path_len),
                                                       std::vector<GroupSetResource>(1));
}

TreeNode* insertAndGetNode(BlockTree&                                        tree,
                           const CacheKeysType&                              cache_keys,
                           const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTreeInsertResult result = tree.insertNode(cache_keys, resources);
    return result.inserted_nodes.back();
}

TEST(BlockTreeTest, EmptyTreeFindReturnsEmpty) {
    BlockTree tree(makeGroupSets(1));
    auto      result = tree.findNode({100, 200, 300});
    EXPECT_TRUE(result.empty());
}

TEST(BlockTreeTest, InsertSinglePath) {
    BlockTree tree(makeGroupSets(1));
    CacheKeysType keys      = {100, 200, 300};
    auto          resources = make2DResources(1, 3, 42);

    TreeNode* leaf = insertAndGetNode(tree, keys, resources);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->cache_key, 300);
    EXPECT_EQ(leaf->group_set_resources[0].device_blocks[0], 44);  // start_block + 2
    EXPECT_EQ(tree.nodes().size(), 3u);                               // 3 nodes created (not counting root)

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
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 1));
    // Insert root → 100 → 200 → 400 (fork at 200)
    tree.insertNode({100, 200, 400}, make2DResources(1, 3, 10));
    // Insert root → 100 → 500 (fork at 100)
    tree.insertNode({100, 500}, make2DResources(1, 2, 20));

    EXPECT_EQ(tree.nodes().size(), 5u);  // 100, 200, 300, 400, 500

    auto* root = tree.root();
    auto* n100 = root->children.at(100);
    EXPECT_EQ(n100->children.size(), 2u);  // 200, 500
    auto* n200 = n100->children.at(200);
    EXPECT_EQ(n200->children.size(), 2u);  // 300, 400
}

TEST(BlockTreeTest, FindExistingPath) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42));

    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result.back()->cache_key, 300);
}

TEST(BlockTreeTest, FindPartialMatch) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42));

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
        tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42));

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
    tree.insertNode({100}, make2DResources(1, 1, 1));

    auto result = tree.findNode({});
    EXPECT_TRUE(result.empty());
}

TEST(BlockTreeTest, RemoveLeafNode) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 42));
    EXPECT_EQ(tree.nodes().size(), 3u);

    // Find and remove leaf node (300)
    auto result = tree.findNode({100, 200, 300});
    ASSERT_FALSE(result.empty());
    tree.removeNode(result.back());
    EXPECT_EQ(tree.nodes().size(), 2u);

    // Node 300 should no longer be findable
    auto result2 = tree.findNode({100, 200, 300});
    EXPECT_EQ(result2.size(), 2u);
    EXPECT_EQ(result2.back()->cache_key, 200);
}

TEST(BlockTreeTest, RemoveEmptyAncestors) {
    BlockTree tree(makeGroupSets(1));
    // Insert root → 100 → 200 with empty group_set_resources (no data)
    tree.insertNode({100, 200}, makeEmpty2DResources(2));
    // Insert root → 100 → 200 → 300 with data only on the new leaf.
    auto leaf_resources = makeEmpty2DResources(3);
    leaf_resources[2][0].device_blocks = {42};
    TreeNode* leaf                     = insertAndGetNode(tree, {100, 200, 300}, leaf_resources);
    EXPECT_EQ(tree.nodes().size(), 3u);

    TreeNode* first_empty_ancestor = leaf->parent;

    // Remove the leaf, then let removeEmptyAncestors prune 200 and 100.
    tree.removeNode(leaf);
    EXPECT_EQ(tree.nodes().size(), 2u);

    TreeNode* surviving_ancestor = tree.removeEmptyAncestors(first_empty_ancestor, {0});
    EXPECT_EQ(surviving_ancestor, tree.root());
    EXPECT_EQ(tree.nodes().size(), 0u);
}

TEST(BlockTreeTest, RemoveEmptyAncestorsStopsAtData) {
    BlockTree tree(makeGroupSets(1));
    // Insert 100 with data
    tree.insertNode({100}, make2DResources(1, 1, 10));
    // Insert 100 → 200 with data (100 already exists, only 200 is new)
    TreeNode* leaf = insertAndGetNode(tree, {100, 200}, make2DResources(1, 2, 20));

    // Remove leaf 200
    tree.removeNode(leaf);

    // removeEmptyAncestors from 100's position: 100 has data → stops
    auto                result             = tree.findNode({100});
    ASSERT_FALSE(result.empty());
    std::vector<size_t> reusable_groups    = {0};
    TreeNode*           surviving_ancestor = tree.removeEmptyAncestors(result.back(), reusable_groups);

    // 100 should still be in the tree (it has data in group 0)
    EXPECT_EQ(surviving_ancestor, result.back());
    EXPECT_EQ(tree.nodes().size(), 1u);
    auto check = tree.findNode({100});
    EXPECT_EQ(check.size(), 1u);
}

TEST(BlockTreeTest, RemoveEmptyAncestorsReturnsFirstSurvivorAfterPruning) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100}, make2DResources(1, 1, 10));
    TreeNode* leaf = insertAndGetNode(tree, {100, 200, 300}, makeEmpty2DResources(3));
    ASSERT_NE(leaf, nullptr);
    TreeNode* first_empty_ancestor = leaf->parent;

    tree.removeNode(leaf);
    TreeNode* surviving_ancestor = tree.removeEmptyAncestors(first_empty_ancestor, {0});

    auto node100 = tree.findNode({100});
    ASSERT_FALSE(node100.empty());
    EXPECT_EQ(surviving_ancestor, node100.back());
    EXPECT_EQ(surviving_ancestor->cache_key, 100);
    EXPECT_EQ(tree.nodes().size(), 1u);
}

TEST(BlockTreeTest, RepeatedInsertDoesNotDuplicate) {
    BlockTree tree(makeGroupSets(1));
    tree.insertNode({100, 200}, make2DResources(1, 2, 1));
    EXPECT_EQ(tree.nodes().size(), 2u);

    // Insert same path again — should reuse existing nodes
    tree.insertNode({100, 200}, make2DResources(1, 2, 50));
    EXPECT_EQ(tree.nodes().size(), 2u);

    // After Bug 3 fix: existing nodes are NOT overwritten.
    // Only newly created nodes get group_set_resources assigned.
    auto result = tree.findNode({100, 200});
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result.back()->group_set_resources[0].device_blocks[0], 2);  // original value (1+1)
}

TEST(BlockTreeTest, InsertEmptyKeys) {
    BlockTree tree(makeGroupSets(1));
    const BlockTreeInsertResult result = tree.insertNode({}, {});
    EXPECT_TRUE(result.inserted_nodes.empty());
    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_EQ(tree.nodes().size(), 0u);
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
    tree.insertNode({100, 200}, make2DResources(1, 2, 42));

    // Second insert: 100 -> 200 -> 300, with device_blocks={99, 100, 101}
    tree.insertNode({100, 200, 300}, make2DResources(1, 3, 99));

    // Verify: nodes 100 and 200 retain original values, only 300 gets new value
    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0]->group_set_resources[0].device_blocks[0], 42);   // 100 unchanged
    EXPECT_EQ(result[1]->group_set_resources[0].device_blocks[0], 43);   // 200 unchanged
    EXPECT_EQ(result[2]->group_set_resources[0].device_blocks[0], 101);  // 300 new (99+2)
}

TEST(BlockTreeTest, InsertFillsOnlyCompleteEmptyIdleGroupsOnExistingNode) {
    BlockTree tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {10};
    original[0][1].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);

    std::vector<std::vector<GroupSetResource>> replacement(1, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks = {20};
    replacement[0][1].device_blocks = {30};

    const BlockTreeInsertResult result = tree.insertNode({100}, replacement);
    EXPECT_TRUE(result.inserted_nodes.empty());
    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].first, node);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{1}));
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{30}));
}

TEST(BlockTreeTest, InsertAggregatesAdoptedGroupSetsPerNode) {
    BlockTree tree(makeGroupSets(2));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(2));
    original[0][0].device_blocks = {NULL_BLOCK_IDX};
    original[0][1].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);

    std::vector<std::vector<GroupSetResource>> replacement(1, std::vector<GroupSetResource>(2));
    replacement[0][0].device_blocks = {20};
    replacement[0][1].device_blocks = {30};
    const BlockTreeInsertResult result = tree.insertNode({100}, replacement);

    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].first, node);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{0, 1}));
}

TEST(BlockTreeTest, InsertSkipsBusyEmptyGroupOnExistingNode) {
    BlockTree tree(makeGroupSets(1));
    std::vector<std::vector<GroupSetResource>> original(1, std::vector<GroupSetResource>(1));
    original[0][0].device_blocks = {NULL_BLOCK_IDX};
    TreeNode* node               = insertAndGetNode(tree, {100}, original);
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    const BlockTreeInsertResult result = tree.insertNode({100}, make2DResources(1, 1, 20));
    EXPECT_TRUE(result.adopted_nodes.empty());
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{NULL_BLOCK_IDX}));
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
}

TEST(BlockTreeTest, InsertSkipsBusyGroupButFillsOtherIdleGroupAndAddsSuffix) {
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
    const BlockTreeInsertResult result = tree.insertNode({100, 200}, replacement);

    ASSERT_EQ(result.adopted_nodes.size(), 1u);
    EXPECT_EQ(result.adopted_nodes[0].second, (std::vector<size_t>{1}));
    ASSERT_EQ(result.inserted_nodes.size(), 1u);
    EXPECT_EQ(result.inserted_nodes[0]->cache_key, 200);
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{NULL_BLOCK_IDX}));
    EXPECT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{30}));
    EXPECT_EQ(result.inserted_nodes[0]->group_set_resources[0].device_blocks, (BlockIndicesType{21}));
    EXPECT_EQ(result.inserted_nodes[0]->group_set_resources[1].device_blocks, (BlockIndicesType{31}));
}

TEST(BlockTreeTest, RemoveEmptyAncestorsStopsAtBusyEmptyNode) {
    BlockTree tree(makeGroupSets(1));
    TreeNode* node = insertAndGetNode(tree, {100}, makeEmpty2DResources(1));
    ASSERT_NE(node, nullptr);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOAD_PENDING;

    EXPECT_EQ(tree.removeEmptyAncestors(node, {0}), node);
    EXPECT_EQ(tree.nodes().size(), 1u);
    EXPECT_EQ(node->parent, tree.root());
}

}  // namespace
}  // namespace rtp_llm
