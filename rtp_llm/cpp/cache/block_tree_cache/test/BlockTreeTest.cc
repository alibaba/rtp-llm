#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

namespace rtp_llm {
namespace {

// Helper: create a vector of GroupSlots (one group) with a device block.
std::vector<GroupSlot> makeGroupSlots(int group_count, BlockIdxType block_idx) {
    std::vector<GroupSlot> slots(static_cast<size_t>(group_count));
    slots[0].device_blocks = {block_idx};
    return slots;
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
    auto          slots = makeGroupSlots(1, 42);

    TreeNode* leaf = tree.insertNode(keys, slots);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->cache_key, 300);
    EXPECT_EQ(leaf->group_slots[0].device_blocks[0], 42);
    EXPECT_EQ(tree.nodeCount(), 3u);  // 3 nodes created (not counting root)

    // Verify tree structure
    auto* root = tree.root();
    EXPECT_EQ(root->children.size(), 1u);
    auto* a = root->children.at(100);
    EXPECT_EQ(a->cache_key, 100);
    auto* b = a->children.at(200);
    EXPECT_EQ(b->cache_key, 200);
    auto* c = b->children.at(300);
    EXPECT_EQ(c->cache_key, 300);
    EXPECT_EQ(c, leaf);
}

TEST(BlockTreeTest, InsertForkPath) {
    BlockTree tree(1);

    // Insert root → 100 → 200 → 300
    tree.insertNode({100, 200, 300}, makeGroupSlots(1, 1));
    // Insert root → 100 → 200 → 400 (fork at 200)
    tree.insertNode({100, 200, 400}, makeGroupSlots(1, 2));
    // Insert root → 100 → 500 (fork at 100)
    tree.insertNode({100, 500}, makeGroupSlots(1, 3));

    EXPECT_EQ(tree.nodeCount(), 5u);  // 100, 200, 300, 400, 500

    auto* root = tree.root();
    auto* n100 = root->children.at(100);
    EXPECT_EQ(n100->children.size(), 2u);  // 200, 500
    auto* n200 = n100->children.at(200);
    EXPECT_EQ(n200->children.size(), 2u);  // 300, 400
}

TEST(BlockTreeTest, FindExistingPath) {
    BlockTree tree(1);
    tree.insertNode({100, 200, 300}, makeGroupSlots(1, 42));

    auto result = tree.findNode({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 300);
    EXPECT_EQ(result.path.size(), 3u);
}

TEST(BlockTreeTest, FindPartialMatch) {
    BlockTree tree(1);
    tree.insertNode({100, 200, 300}, makeGroupSlots(1, 42));

    // Search for a longer path — only first 2 match
    auto result = tree.findNode({100, 200, 999});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 200);
}

TEST(BlockTreeTest, FindEmptyKeys) {
    BlockTree tree(1);
    tree.insertNode({100}, makeGroupSlots(1, 1));

    auto result = tree.findNode({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
}

TEST(BlockTreeTest, RemoveLeafNode) {
    BlockTree tree(1);
    tree.insertNode({100, 200, 300}, makeGroupSlots(1, 42));
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
    // Insert root → 100 → 200 → 300, all with empty group_slots except leaf
    tree.insertNode({100, 200}, {});  // intermediate nodes have default (empty) slots
    TreeNode* leaf = tree.insertNode({100, 200, 300}, makeGroupSlots(1, 42));
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
    // Insert 100 → 200, where 100 has data
    tree.insertNode({100}, makeGroupSlots(1, 10));
    TreeNode* leaf = tree.insertNode({100, 200}, makeGroupSlots(1, 20));

    // Remove leaf 200
    tree.removeNode(leaf);

    // removeEmptyAncestors from 200's position: 100 has data → stops
    auto             result          = tree.findNode({100});
    std::vector<int> reusable_groups = {0};
    tree.removeEmptyAncestors(result.matched_node, reusable_groups);

    // 100 should still be in the tree
    EXPECT_EQ(tree.nodeCount(), 1u);
    auto check = tree.findNode({100});
    EXPECT_EQ(check.matched_blocks, 1u);
}

TEST(BlockTreeTest, RepeatedInsertDoesNotDuplicate) {
    BlockTree tree(1);
    tree.insertNode({100, 200}, makeGroupSlots(1, 1));
    EXPECT_EQ(tree.nodeCount(), 2u);

    // Insert same path again — should reuse existing nodes
    tree.insertNode({100, 200}, makeGroupSlots(1, 2));
    EXPECT_EQ(tree.nodeCount(), 2u);

    // After Bug 3 fix: existing nodes are NOT overwritten.
    // Only newly created nodes get group_slots assigned.
    auto result = tree.findNode({100, 200});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->group_slots[0].device_blocks[0], 1);
}

TEST(BlockTreeTest, InsertEmptyKeys) {
    BlockTree tree(1);
    TreeNode* node = tree.insertNode({}, {});
    EXPECT_EQ(node, tree.root());
    EXPECT_EQ(tree.nodeCount(), 0u);
}

TEST(BlockTreeTest, MultipleGroups) {
    BlockTree tree(3);  // 3 component groups

    std::vector<GroupSlot> slots(3);
    slots[0].device_blocks = {10};
    slots[1].device_blocks = {20, 21};
    slots[2].device_blocks = {30};

    TreeNode* leaf = tree.insertNode({100}, slots);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->group_slots.size(), 3u);
    EXPECT_EQ(leaf->group_slots[0].device_blocks[0], 10);
    EXPECT_EQ(leaf->group_slots[1].device_blocks.size(), 2u);
    EXPECT_EQ(leaf->group_slots[2].device_blocks[0], 30);
}

// UT-1: Verify insertNode does not overwrite existing node's group_slots (Bug 3 fix)
TEST(BlockTreeTest, InsertDoesNotOverwriteExistingNodeSlots) {
    BlockTree tree(1);

    // First insert: 100 -> 200, with device_blocks={42}
    std::vector<GroupSlot> slots1(1);
    slots1[0].device_blocks = {42};
    tree.insertNode({100, 200}, slots1);

    // Second insert: 100 -> 200 -> 300, with device_blocks={99}
    std::vector<GroupSlot> slots2(1);
    slots2[0].device_blocks = {99};
    tree.insertNode({100, 200, 300}, slots2);

    // Verify: nodes 100 and 200 retain original {42}, only 300 gets {99}
    auto result = tree.findNode({100, 200, 300});
    ASSERT_EQ(result.path.size(), 3u);
    EXPECT_EQ(result.path[0]->group_slots[0].device_blocks[0], 42);  // 100 unchanged
    EXPECT_EQ(result.path[1]->group_slots[0].device_blocks[0], 42);  // 200 unchanged
    EXPECT_EQ(result.path[2]->group_slots[0].device_blocks[0], 99);  // 300 new
}

// UT-3: Verify removeEmptyAncestors only considers REUSABLE groups (Bug 2 fix)
TEST(BlockTreeTest, RemoveEmptyAncestorsIgnoresNonReusableGroups) {
    BlockTree tree(2);

    // Build: root -> A(100) -> B(200)
    std::vector<GroupSlot> slots(2);
    slots[0].device_blocks = {NULL_BLOCK_IDX};  // group 0 (REUSABLE) empty
    slots[1].device_blocks = {42};              // group 1 (NON_REUSABLE) has data
    tree.insertNode({100, 200}, slots);

    auto find = tree.findNode({100, 200});
    ASSERT_EQ(find.path.size(), 2u);

    // Remove B
    tree.removeNode(find.path[1]);

    // removeEmptyAncestors with only REUSABLE group IDs = {0}
    // A's group 0 is empty -> should be deleted (ignoring group 1)
    tree.removeEmptyAncestors(find.path[0], {0});

    EXPECT_EQ(tree.nodeCount(), 0u);
}

}  // namespace
}  // namespace rtp_llm
