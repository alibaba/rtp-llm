#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {
namespace {

class FullComponentGroupTest: public ::testing::Test {
protected:
    void SetUp() override {
        group_                     = std::make_shared<FullComponentGroup>();
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

    void setDiskSlot(TreeNode* node, int gid, BlockIdxType slot) {
        node->group_slots[static_cast<size_t>(gid)].disk_slot = slot;
    }

    std::shared_ptr<FullComponentGroup> group_;
};

TEST_F(FullComponentGroupTest, DeviceLeafDetection) {
    // Create: root → A → B → C (C is leaf)
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    auto* c          = makeNode(300);
    a->children[200] = b;
    b->parent        = a;
    b->children[300] = c;
    c->parent        = b;

    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);
    setDeviceBlock(c, 0, 30);

    // C is DeviceLeaf (no children with device value)
    EXPECT_TRUE(group_->isLeafAtTier(c, 0, Tier::DEVICE));
    // B is NOT DeviceLeaf (child C has device value)
    EXPECT_FALSE(group_->isLeafAtTier(b, 0, Tier::DEVICE));
    // A is NOT DeviceLeaf (child B has device value)
    EXPECT_FALSE(group_->isLeafAtTier(a, 0, Tier::DEVICE));

    delete a;
    delete b;
    delete c;
}

TEST_F(FullComponentGroupTest, DeviceLeafAfterChildEviction) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0, 10);
    setDeviceBlock(b, 0, 20);

    EXPECT_FALSE(group_->isLeafAtTier(a, 0, Tier::DEVICE));

    // Evict B's device data
    setDeviceBlock(b, 0, NULL_BLOCK_IDX);

    // Now A should be DeviceLeaf
    EXPECT_TRUE(group_->isLeafAtTier(a, 0, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(FullComponentGroupTest, DriveEvictionDevice) {
    auto* a = makeNode(100);
    setDeviceBlock(a, 0, 42);

    // A is a DeviceLeaf (no children)
    group_->tryAddToDeviceHeap(a);
    EXPECT_TRUE(a->group_slots[0].in_device_heap);
    EXPECT_EQ(group_->device_heap->size(), 1u);

    auto result = group_->driveEviction(1, Tier::DEVICE);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->node, a);
    EXPECT_EQ(result->source_tier, Tier::DEVICE);
    EXPECT_EQ(result->target_tier, Tier::HOST);  // REUSABLE → demote to host
    EXPECT_EQ(result->blocks_to_release.size(), 1u);
    EXPECT_EQ(result->blocks_to_release[0], 42);

    delete a;
}

TEST_F(FullComponentGroupTest, EvictFromTierDevice) {
    auto* a = makeNode(100);
    setDeviceBlock(a, 0, 42);
    group_->tryAddToDeviceHeap(a);

    group_->evictFromTier(a, a->group_slots[0], Tier::DEVICE);

    // Device blocks should be cleared
    EXPECT_FALSE(a->group_slots[0].has_device_value());
    EXPECT_FALSE(a->group_slots[0].in_device_heap);

    delete a;
}

TEST_F(FullComponentGroupTest, EvictFromTierHost) {
    auto* a = makeNode(100);
    setHostBlock(a, 0, 15);

    group_->evictFromTier(a, a->group_slots[0], Tier::HOST);

    EXPECT_FALSE(a->group_slots[0].has_host_value());
    EXPECT_FALSE(a->group_slots[0].in_host_heap);

    delete a;
}

TEST_F(FullComponentGroupTest, EvictFromTierDisk) {
    auto* a = makeNode(100);
    setDiskSlot(a, 0, 8);

    group_->evictFromTier(a, a->group_slots[0], Tier::DISK);

    EXPECT_FALSE(a->group_slots[0].has_disk_value());
    EXPECT_FALSE(a->group_slots[0].in_disk_heap);

    delete a;
}

TEST_F(FullComponentGroupTest, MatchValidatorFullPathValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setDeviceBlock(node, 0, 42);

    EXPECT_TRUE(validator->validate(node, node->group_slots[0]));

    delete node;
}

TEST_F(FullComponentGroupTest, MatchValidatorHostDataValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setHostBlock(node, 0, 15);

    EXPECT_TRUE(validator->validate(node, node->group_slots[0]));

    delete node;
}

TEST_F(FullComponentGroupTest, MatchValidatorEmptyInvalid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    // No data in any tier

    EXPECT_FALSE(validator->validate(node, node->group_slots[0]));

    delete node;
}

TEST_F(FullComponentGroupTest, BuildTransferD2H) {
    auto* node = makeNode(100);
    setDeviceBlock(node, 0, 42);

    auto desc = group_->buildTransfer(node, TransferType::DEVICE_TO_HOST);
    EXPECT_EQ(desc.source_tier, Tier::DEVICE);
    EXPECT_EQ(desc.target_tier, Tier::HOST);
    EXPECT_EQ(desc.component_group_id, 0);
    EXPECT_EQ(desc.nodes.size(), 1u);
    EXPECT_EQ(desc.source_blocks.size(), 1u);
    EXPECT_EQ(desc.source_blocks[0][0], 42);

    delete node;
}

TEST_F(FullComponentGroupTest, CommitInsertData) {
    auto*                     node   = makeNode(100);
    std::vector<BlockIdxType> blocks = {10, 20, 30};

    group_->commitInsertData(node, node->group_slots[0], blocks);

    EXPECT_EQ(node->group_slots[0].device_blocks.size(), 3u);
    EXPECT_EQ(node->group_slots[0].device_blocks[0], 10);
    EXPECT_EQ(node->group_slots[0].device_blocks[1], 20);
    EXPECT_EQ(node->group_slots[0].device_blocks[2], 30);

    delete node;
}

TEST_F(FullComponentGroupTest, HostLeafDetection) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    // A: evicted from device, has host data
    setHostBlock(a, 0, 15);
    // B: evicted from device, has host data
    setHostBlock(b, 0, 25);

    // B is HostLeaf (no child with host value)
    EXPECT_TRUE(group_->isLeafAtTier(b, 0, Tier::HOST));
    // A is NOT HostLeaf (child B has host value)
    EXPECT_FALSE(group_->isLeafAtTier(a, 0, Tier::HOST));

    delete a;
    delete b;
}

TEST_F(FullComponentGroupTest, TryAddToHostHeap) {
    auto* a = makeNode(100);
    // Evicted from device, has host data
    setHostBlock(a, 0, 15);

    group_->tryAddToHostHeap(a);
    EXPECT_TRUE(a->group_slots[0].in_host_heap);
    EXPECT_EQ(group_->host_heap->size(), 1u);

    delete a;
}

TEST_F(FullComponentGroupTest, DriveEvictionHost) {
    auto* a = makeNode(100);
    setHostBlock(a, 0, 15);

    group_->tryAddToHostHeap(a);
    auto result = group_->driveEviction(1, Tier::HOST);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source_tier, Tier::HOST);
    EXPECT_EQ(result->target_tier, Tier::DISK);

    delete a;
}

TEST_F(FullComponentGroupTest, DriveEvictionEmptyHeap) {
    auto result = group_->driveEviction(1, Tier::DEVICE);
    EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace rtp_llm
