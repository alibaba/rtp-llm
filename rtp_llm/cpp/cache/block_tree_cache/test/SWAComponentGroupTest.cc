#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"

namespace rtp_llm {
namespace {

class SWAComponentGroupTest: public ::testing::Test {
protected:
    void SetUp() override {
        group_ = std::make_shared<SWAComponentGroup>(
            /*sliding_window_size=*/128,
            /*seq_size_per_block=*/64);
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

    void setHostBlock(TreeNode* node, int gid, BlockIdxType block) {
        node->group_slots[static_cast<size_t>(gid)].host_block = block;
    }

    std::shared_ptr<SWAComponentGroup> group_;
};

TEST_F(SWAComponentGroupTest, AnyNodeHeapMiddleNode) {
    // SWA allows any node with data to be in heap (not just leaves)
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);

    // Both A and B can be added to heap (even though A has child B)
    group_->tryAddToDeviceHeap(a);
    group_->tryAddToDeviceHeap(b);

    EXPECT_TRUE(a->group_slots[0].in_device_heap);
    EXPECT_TRUE(b->group_slots[0].in_device_heap);
    EXPECT_EQ(group_->device_heap->size(), 2u);

    delete a;
    delete b;
}

TEST_F(SWAComponentGroupTest, WindowValidatorConnectedPath) {
    auto  validator     = group_->createMatchValidator();
    auto* swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());

    // Path: root → A (has data) → B (has data) → C (has data)
    auto* a = makeNode(100);
    auto* b = makeNode(200);
    auto* c = makeNode(300);
    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);
    setDeviceBlock(c, 0, 30);

    EXPECT_TRUE(validator->validate(a, a->group_slots[0]));
    EXPECT_TRUE(validator->validate(b, b->group_slots[0]));
    EXPECT_TRUE(validator->validate(c, c->group_slots[0]));

    EXPECT_TRUE(swa_validator->connectedToRoot());
    // 3 blocks * 64 tokens = 192
    EXPECT_EQ(swa_validator->accumulatedLength(), 192u);

    delete a;
    delete b;
    delete c;
}

TEST_F(SWAComponentGroupTest, WindowValidatorGapResetsCount) {
    auto  validator     = group_->createMatchValidator();
    auto* swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());

    // Path: root → A (has data) → B (no data) → C (has data)
    auto* a = makeNode(100);
    auto* b = makeNode(200);
    auto* c = makeNode(300);
    setDeviceBlock(a, 0, 10);
    // B has no SWA data
    setDeviceBlock(c, 0, 30);

    EXPECT_TRUE(validator->validate(a, a->group_slots[0]));
    EXPECT_TRUE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 64u);

    // B: no data → gap
    EXPECT_TRUE(validator->validate(b, b->group_slots[0]));
    EXPECT_FALSE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 0u);

    // C: has data → start counting from gap
    EXPECT_TRUE(validator->validate(c, c->group_slots[0]));
    EXPECT_EQ(swa_validator->accumulatedLength(), 64u);

    delete a;
    delete b;
    delete c;
}

TEST_F(SWAComponentGroupTest, WindowValidatorMultitierNoReset) {
    auto validator = group_->createMatchValidator();

    // B has no device data but has host data → should not reset
    auto* a = makeNode(100);
    auto* b = makeNode(200);
    auto* c = makeNode(300);
    setDeviceBlock(a, 0, 10);
    // B: host data only
    setDeviceBlock(b, 0, NULL_BLOCK_IDX);
    b->group_slots[0].host_block = 15;
    setDeviceBlock(c, 0, 30);

    EXPECT_TRUE(validator->validate(a, a->group_slots[0]));
    // B has host data → has_any_value() is true → no reset
    EXPECT_TRUE(validator->validate(b, b->group_slots[0]));
    EXPECT_TRUE(validator->validate(c, c->group_slots[0]));

    auto* swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());
    EXPECT_TRUE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 192u);

    delete a;
    delete b;
    delete c;
}

TEST_F(SWAComponentGroupTest, IndependentEvictionDoesNotAffectFull) {
    // SWA eviction only affects SWA group data, not Full group
    auto* node                         = makeNode(100, 2);  // 2 groups: 0=Full, 1=SWA
    node->group_slots[0].device_blocks = {42};              // Full data
    node->group_slots[1].device_blocks = {99};              // SWA data

    group_->component_group_id = 1;  // SWA group
    group_->evictFromTier(node, node->group_slots[1], Tier::DEVICE);

    // SWA data cleared
    EXPECT_FALSE(node->group_slots[1].has_device_value());
    // Full data intact
    EXPECT_TRUE(node->group_slots[0].has_device_value());
    EXPECT_EQ(node->group_slots[0].device_blocks[0], 42);

    delete node;
}

TEST_F(SWAComponentGroupTest, DriveEvictionFromAnyNode) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);

    group_->tryAddToDeviceHeap(a);
    group_->tryAddToDeviceHeap(b);

    // Both nodes are candidates (not just leaves)
    auto result = group_->driveEviction(1, Tier::DEVICE);
    ASSERT_TRUE(result.has_value());
    // Either A or B could be popped (depends on LRU ordering)
    EXPECT_NE(result->node, nullptr);

    delete a;
    delete b;
}

TEST_F(SWAComponentGroupTest, SlidingWindowConfig) {
    EXPECT_EQ(group_->slidingWindowSize(), 128);
    EXPECT_EQ(group_->seqSizePerBlock(), 64);
}

}  // namespace
}  // namespace rtp_llm
