#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

namespace rtp_llm {
namespace {

// Helper: create 2D slots — each node gets a single-group slot with incrementing block_idx.
// slots[i][0].device_blocks = {start_block + i}
std::vector<std::vector<GroupSlot>> make2DSlots(int group_count, int path_len, BlockIdxType start_block) {
    std::vector<std::vector<GroupSlot>> slots(static_cast<size_t>(path_len));
    for (int i = 0; i < path_len; ++i) {
        slots[i].resize(static_cast<size_t>(group_count));
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(start_block + i)};
    }
    return slots;
}

// Helper: create 2D slots with empty inner vectors (no data assigned to nodes).
std::vector<std::vector<GroupSlot>> makeEmpty2DSlots(int path_len) {
    return std::vector<std::vector<GroupSlot>>(static_cast<size_t>(path_len));
}

TEST(BlockTreeTest, EmptyTreeFindReturnsEmpty) {
    BlockTree tree(1);
    auto      result = tree.findNode({100, 200, 300});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.path.empty());
}

TEST(BlockTreeTest, InsertSinglePath) {
    BlockTree     tree(1);
    CacheKeysType keys  = {100, 200, 300};
    auto          slots = make2DSlots(1, 3, 42);

    TreeNode* leaf = tree.insertNode(nullptr, keys, slots);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->cache_key, 300);
    EXPECT_EQ(leaf->group_slots[0].device_blocks[0], 44);  // start_block + 2
    EXPECT_EQ(tree.nodeCount(), 3u);                       // 3 nodes created (not counting root)

    // Verify tree structure and per-node slots
    auto* root = tree.root();
    EXPECT_EQ(root->children.size(), 1u);
    auto* a = root->children.at(100);
    EXPECT_EQ(a->cache_key, 100);
    EXPECT_EQ(a->group_slots[0].device_blocks[0], 42);  // start_block + 0
    auto* b = a->children.at(200);
    EXPECT_EQ(b->cache_key, 200);
    EXPECT_EQ(b->group_slots[0].device_blocks[0], 43);  // start_block + 1
    auto* c = b->children.at(300);
    EXPECT_EQ(c->cache_key, 300);
    EXPECT_EQ(c, leaf);
}

TEST(BlockTreeTest, InsertForkPath) {
    BlockTree tree(1);

    // Insert root → 100 → 200 → 300
    tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 1));
    // Insert root → 100 → 200 → 400 (fork at 200)
    tree.insertNode(nullptr, {100, 200, 400}, make2DSlots(1, 3, 10));
    // Insert root → 100 → 500 (fork at 100)
    tree.insertNode(nullptr, {100, 500}, make2DSlots(1, 2, 20));

    EXPECT_EQ(tree.nodeCount(), 5u);  // 100, 200, 300, 400, 500

    auto* root = tree.root();
    auto* n100 = root->children.at(100);
    EXPECT_EQ(n100->children.size(), 2u);  // 200, 500
    auto* n200 = n100->children.at(200);
    EXPECT_EQ(n200->children.size(), 2u);  // 300, 400
}

TEST(BlockTreeTest, FindExistingPath) {
    BlockTree tree(1);
    tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 42));

    auto result = tree.findNode({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 300);
    EXPECT_EQ(result.path.size(), 3u);
}

TEST(BlockTreeTest, FindPartialMatch) {
    BlockTree tree(1);
    tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 42));

    // Search for a longer path — only first 2 match
    auto result = tree.findNode({100, 200, 999});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 200);
}

TEST(BlockTreeTest, FindEmptyKeys) {
    BlockTree tree(1);
    tree.insertNode(nullptr, {100}, make2DSlots(1, 1, 1));

    auto result = tree.findNode({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
}

TEST(BlockTreeTest, RemoveLeafNode) {
    BlockTree tree(1);
    tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 42));
    EXPECT_EQ(tree.nodeCount(), 3u);

    // Find and remove leaf node (300)
    auto result = tree.findNode({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    tree.removeNode(result.matched_node);
    EXPECT_EQ(tree.nodeCount(), 2u);

    // Node 300 should no longer be findable
    auto result2 = tree.findNode({100, 200, 300});
    EXPECT_EQ(result2.matched_blocks, 2u);
    EXPECT_EQ(result2.matched_node->cache_key, 200);
}

TEST(BlockTreeTest, RemoveEmptyAncestors) {
    BlockTree tree(1);
    // Insert root → 100 → 200 with empty group_slots (no data)
    tree.insertNode(nullptr, {100, 200}, makeEmpty2DSlots(2));
    // Insert root → 100 → 200 → 300 with data (existing 100, 200 not overwritten)
    TreeNode* leaf = tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 42));
    EXPECT_EQ(tree.nodeCount(), 3u);

    // Remove the leaf
    tree.removeNode(leaf);
    EXPECT_EQ(tree.nodeCount(), 2u);

    // Find and remove 200 (now has no children after 300 was removed)
    auto      result  = tree.findNode({100, 200});
    TreeNode* node200 = result.matched_node;
    ASSERT_NE(node200, nullptr);
    tree.removeNode(node200);
    EXPECT_EQ(tree.nodeCount(), 1u);

    // Find and remove 100 (now has no children)
    auto      result100 = tree.findNode({100});
    TreeNode* node100   = result100.matched_node;
    ASSERT_NE(node100, nullptr);
    tree.removeNode(node100);
    EXPECT_EQ(tree.nodeCount(), 0u);
}

TEST(BlockTreeTest, RemoveEmptyAncestorsStopsAtData) {
    BlockTree tree(1);
    // Insert 100 with data
    tree.insertNode(nullptr, {100}, make2DSlots(1, 1, 10));
    // Insert 100 → 200 with data (100 already exists, only 200 is new)
    TreeNode* leaf = tree.insertNode(nullptr, {100, 200}, make2DSlots(1, 2, 20));

    // Remove leaf 200
    tree.removeNode(leaf);

    // removeEmptyAncestors from 100's position: 100 has data → stops
    auto             result          = tree.findNode({100});
    std::vector<int> reusable_groups = {0};
    tree.removeEmptyAncestors(result.matched_node, reusable_groups);

    // 100 should still be in the tree (it has data in group 0)
    EXPECT_EQ(tree.nodeCount(), 1u);
    auto check = tree.findNode({100});
    EXPECT_EQ(check.matched_blocks, 1u);
}

TEST(BlockTreeTest, RepeatedInsertDoesNotDuplicate) {
    BlockTree tree(1);
    tree.insertNode(nullptr, {100, 200}, make2DSlots(1, 2, 1));
    EXPECT_EQ(tree.nodeCount(), 2u);

    // Insert same path again — should reuse existing nodes
    tree.insertNode(nullptr, {100, 200}, make2DSlots(1, 2, 50));
    EXPECT_EQ(tree.nodeCount(), 2u);

    // After Bug 3 fix: existing nodes are NOT overwritten.
    // Only newly created nodes get group_slots assigned.
    auto result = tree.findNode({100, 200});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->group_slots[0].device_blocks[0], 2);  // original value (1+1)
}

TEST(BlockTreeTest, InsertEmptyKeys) {
    BlockTree tree(1);
    TreeNode* node = tree.insertNode(nullptr, {}, {});
    EXPECT_EQ(node, tree.root());
    EXPECT_EQ(tree.nodeCount(), 0u);
}

TEST(BlockTreeTest, InsertWithParent) {
    BlockTree tree(1);
    // First insert: root → 100 → 200
    tree.insertNode(nullptr, {100, 200}, make2DSlots(1, 2, 10));

    // Find node 200 and use it as parent to insert 300
    auto      find   = tree.findNode({100, 200});
    TreeNode* parent = find.matched_node;
    ASSERT_NE(parent, nullptr);

    TreeNode* leaf = tree.insertNode(parent, {300}, make2DSlots(1, 1, 50));
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->cache_key, 300);
    EXPECT_EQ(leaf->group_slots[0].device_blocks[0], 50);
    EXPECT_EQ(tree.nodeCount(), 3u);

    // Verify full path is findable
    auto result = tree.findNode({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
}

TEST(BlockTreeTest, MultipleGroups) {
    BlockTree tree(3);  // 3 component groups

    // Create 2D slots for a single node with 3 groups
    std::vector<std::vector<GroupSlot>> slots(1);
    slots[0].resize(3);
    slots[0][0].device_blocks = {10};
    slots[0][1].device_blocks = {20, 21};
    slots[0][2].device_blocks = {30};

    TreeNode* leaf = tree.insertNode(nullptr, {100}, slots);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->group_slots.size(), 3u);
    EXPECT_EQ(leaf->group_slots[0].device_blocks[0], 10);
    EXPECT_EQ(leaf->group_slots[1].device_blocks.size(), 2u);
    EXPECT_EQ(leaf->group_slots[2].device_blocks[0], 30);
}

// UT-1: Verify insertNode does not overwrite existing node's group_slots (Bug 3 fix)
TEST(BlockTreeTest, InsertDoesNotOverwriteExistingNodeSlots) {
    BlockTree tree(1);

    // First insert: 100 -> 200, with device_blocks={42, 43}
    tree.insertNode(nullptr, {100, 200}, make2DSlots(1, 2, 42));

    // Second insert: 100 -> 200 -> 300, with device_blocks={99, 100, 101}
    tree.insertNode(nullptr, {100, 200, 300}, make2DSlots(1, 3, 99));

    // Verify: nodes 100 and 200 retain original values, only 300 gets new value
    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.path.size(), 3u);
    EXPECT_EQ(result.path[0]->group_slots[0].device_blocks[0], 42);   // 100 unchanged
    EXPECT_EQ(result.path[1]->group_slots[0].device_blocks[0], 43);   // 200 unchanged
    EXPECT_EQ(result.path[2]->group_slots[0].device_blocks[0], 101);  // 300 new (99+2)
}

}  // namespace
}  // namespace rtp_llm
