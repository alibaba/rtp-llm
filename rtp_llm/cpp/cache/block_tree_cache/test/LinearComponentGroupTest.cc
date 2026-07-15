#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"

namespace rtp_llm {
namespace {

class LinearComponentGroupTest: public ::testing::Test {
protected:
    void SetUp() override {
        group_                     = std::make_shared<LinearComponentGroup>();
        group_->component_group_id = 0;
    }

    TreeNode* makeNode(CacheKeyType key, int group_count = 1) {
        auto* node      = new TreeNode();
        node->cache_key = key;
        node->group_slots.resize(static_cast<size_t>(group_count));
        return node;
    }

    void setDeviceBlock(TreeNode* node, int gid, BlockIdxType block) {
        node->group_slots[static_cast<size_t>(gid)].device_blocks = {block};
    }

    std::shared_ptr<LinearComponentGroup> group_;
};

TEST_F(LinearComponentGroupTest, DriveEvictionAnyNode) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);

    // Both nodes can be added (Any-node heap)
    group_->tryAddToDeviceHeap(a);
    group_->tryAddToDeviceHeap(b);
    EXPECT_EQ(group_->device_heap->size(), 2u);

    auto result = group_->driveEviction(1, Tier::DEVICE);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->node, nullptr);
    EXPECT_EQ(result->source_tier, Tier::DEVICE);

    delete a;
    delete b;
}

TEST_F(LinearComponentGroupTest, EvictFromTierDevice) {
    auto* node = makeNode(100);
    setDeviceBlock(node, 0, 42);

    group_->tryAddToDeviceHeap(node);
    group_->evictFromTier(node, node->group_slots[0], Tier::DEVICE);

    EXPECT_FALSE(node->group_slots[0].has_value(Tier::DEVICE));
    EXPECT_FALSE(node->group_slots[0].in_device_heap);

    delete node;
}

TEST_F(LinearComponentGroupTest, MatchValidatorHasData) {
    auto validator = group_->createMatchValidator();

    auto* node_with = makeNode(100);
    setDeviceBlock(node_with, 0, 42);
    EXPECT_TRUE(validator->validate(node_with, node_with->group_slots[0]));

    auto* node_empty = makeNode(200);
    EXPECT_FALSE(validator->validate(node_empty, node_empty->group_slots[0]));

    delete node_with;
    delete node_empty;
}

TEST_F(LinearComponentGroupTest, CommitInsertData) {
    auto*                     node   = makeNode(100);
    std::vector<BlockIdxType> blocks = {55};

    group_->commitInsertData(node, node->group_slots[0], blocks);
    EXPECT_EQ(node->group_slots[0].device_blocks[0], 55);

    delete node;
}

TEST_F(LinearComponentGroupTest, DriveEvictionEmptyHeap) {
    auto result = group_->driveEviction(1, Tier::DEVICE);
    EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace rtp_llm
