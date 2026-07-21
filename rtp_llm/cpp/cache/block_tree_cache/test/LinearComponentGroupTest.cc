#include <gtest/gtest.h>

#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

class LinearComponentGroupTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "linear_component_group_test");
        ASSERT_NE(pool_, nullptr);
        group_                     = std::make_shared<LinearComponentGroup>();
        group_->component_group_id = 0;
        group_->setDevicePools({pool_}, {"tag_0"});
    }

    void TearDown() override {
        for (const auto block : held_blocks_) {
            pool_->decRef(block, BlockRefType::REQUEST);
        }
    }

    TreeNode* makeNode(CacheKeyType key, int group_count = 1) {
        auto* node      = new TreeNode();
        node->cache_key = key;
        node->group_slots.resize(static_cast<size_t>(group_count));
        return node;
    }

    BlockIdxType setDeviceBlock(TreeNode* node, int gid) {
        const auto block = pool_->malloc();
        EXPECT_TRUE(block.has_value());
        if (!block.has_value()) {
            return NULL_BLOCK_IDX;
        }
        pool_->incRef(block.value(), BlockRefType::REQUEST);
        held_blocks_.insert(block.value());
        node->group_slots[static_cast<size_t>(gid)].device_blocks = {block.value()};
        return block.value();
    }

    DeviceBlockPoolPtr                    pool_;
    std::unordered_set<BlockIdxType>      held_blocks_;
    std::shared_ptr<LinearComponentGroup> group_;
};

TEST_F(LinearComponentGroupTest, AnyNodeWithDataIsSlotEvictable) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);

    // LINEAR has no leaf/topology requirement; both nodes are eligible.
    EXPECT_TRUE(group_->isSlotEvictable(*a, Tier::DEVICE));
    EXPECT_TRUE(group_->isSlotEvictable(*b, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(LinearComponentGroupTest, EvictFromTierDevice) {
    auto* node = makeNode(100);
    setDeviceBlock(node, 0);

    group_->evictFromTier(node, node->group_slots[0], Tier::DEVICE);

    // Device blocks are cleared; heap ownership no longer lives on the group.
    EXPECT_FALSE(node->group_slots[0].has_value(Tier::DEVICE));

    delete node;
}

TEST_F(LinearComponentGroupTest, MatchValidatorHasData) {
    auto validator = group_->createMatchValidator();

    auto* node_with = makeNode(100);
    setDeviceBlock(node_with, 0);
    EXPECT_TRUE(validator->validate(node_with, node_with->group_slots[0]));

    auto* node_empty = makeNode(200);
    EXPECT_FALSE(validator->validate(node_empty, node_empty->group_slots[0]));

    delete node_with;
    delete node_empty;
}

TEST_F(LinearComponentGroupTest, EmptySlotIsNotEvictable) {
    auto* node = makeNode(100);
    EXPECT_FALSE(group_->isSlotEvictable(*node, Tier::DEVICE));
    EXPECT_FALSE(group_->isSlotEvictable(*node, Tier::HOST));
    delete node;
}

}  // namespace
}  // namespace rtp_llm
