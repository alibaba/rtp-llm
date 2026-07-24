#include <gtest/gtest.h>

#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::TestGroupSpec;

class FullGroupSetTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "full_group_set_test");
        ASSERT_NE(pool_, nullptr);
        group_                     = std::make_shared<FullGroupSet>();
        TestGroupSpec spec;
        spec.tag                   = "tag_0";
        spec.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
        spec.kv_block_stride_bytes = 1;
        group_->initialize(0, makeTestTopology({std::move(spec)}), {0}, {pool_});
    }

    void TearDown() override {
        for (const auto block : held_blocks_) {
            pool_->decRef(block, BlockRefType::REQUEST);
        }
    }

    TreeNode* makeNode(CacheKeyType key, int group_count = 1) {
        auto* node      = new TreeNode();
        node->cache_key = key;
        node->group_set_resources.resize(static_cast<size_t>(group_count));
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
        node->group_set_resources[static_cast<size_t>(gid)].device_blocks = {block.value()};
        return block.value();
    }

    void clearDeviceBlock(TreeNode* node, int gid) {
        node->group_set_resources[static_cast<size_t>(gid)].device_blocks = {NULL_BLOCK_IDX};
    }

    void setHostBlock(TreeNode* node, int gid, BlockIdxType block) {
        node->group_set_resources[static_cast<size_t>(gid)].host_block = block;
    }

    void setDiskSlot(TreeNode* node, int gid, BlockIdxType slot) {
        node->group_set_resources[static_cast<size_t>(gid)].disk_slot = slot;
    }

    DeviceBlockPoolPtr                  pool_;
    std::unordered_set<BlockIdxType>    held_blocks_;
    std::shared_ptr<FullGroupSet> group_;
};

TEST_F(FullGroupSetTest, DeviceLeafDetection) {
    // Create: root -> A -> B -> C (C is leaf)
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    auto* c          = makeNode(300);
    a->children[200] = b;
    b->parent        = a;
    b->children[300] = c;
    c->parent        = b;

    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);
    setDeviceBlock(c, 0);

    // C is DeviceLeaf (no children with device value)
    EXPECT_TRUE(group_->isLeafAtTier(c, Tier::DEVICE));
    // B is NOT DeviceLeaf (child C has device value)
    EXPECT_FALSE(group_->isLeafAtTier(b, Tier::DEVICE));
    // A is NOT DeviceLeaf (child B has device value)
    EXPECT_FALSE(group_->isLeafAtTier(a, Tier::DEVICE));

    delete a;
    delete b;
    delete c;
}

TEST_F(FullGroupSetTest, DeviceLeafAfterChildEviction) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);

    EXPECT_FALSE(group_->isLeafAtTier(a, Tier::DEVICE));

    // Evict B's device data
    clearDeviceBlock(b, 0);

    // Now A should be DeviceLeaf
    EXPECT_TRUE(group_->isLeafAtTier(a, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, DeviceCandidateEligibility) {
    // FULL: only a leaf holding device data is candidate-eligible; a parent whose
    // child still holds device data is not (evicting it would break the prefix).
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;
    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);

    EXPECT_TRUE(group_->isSlotEvictable(*b, Tier::DEVICE));
    EXPECT_FALSE(group_->isSlotEvictable(*a, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, EvictFromTierDevice) {
    auto* a = makeNode(100);
    setDeviceBlock(a, 0);

    group_->evictFromTier(a, a->group_set_resources[0], Tier::DEVICE);

    // Device blocks should be cleared
    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::DEVICE));

    delete a;
}

TEST_F(FullGroupSetTest, EvictFromTierHost) {
    auto* a = makeNode(100);
    setHostBlock(a, 0, 15);

    group_->evictFromTier(a, a->group_set_resources[0], Tier::HOST);

    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::HOST));

    delete a;
}

TEST_F(FullGroupSetTest, EvictFromTierDisk) {
    auto* a = makeNode(100);
    setDiskSlot(a, 0, 8);

    group_->evictFromTier(a, a->group_set_resources[0], Tier::DISK);

    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::DISK));

    delete a;
}

TEST_F(FullGroupSetTest, MatchValidatorFullPathValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setDeviceBlock(node, 0);

    EXPECT_TRUE(validator->validate(node, node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, MatchValidatorHostDataValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setHostBlock(node, 0, 15);

    EXPECT_TRUE(validator->validate(node, node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, MatchValidatorEmptyInvalid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    // No data in any tier

    EXPECT_FALSE(validator->validate(node, node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, BuildTransferD2H) {
    auto*              node         = makeNode(100);
    const BlockIdxType device_block = setDeviceBlock(node, 0);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);

    TransferDescriptor desc = group_->buildTransfer(node, TransferType::DEVICE_TO_HOST);
    EXPECT_EQ(desc.source_tier, Tier::DEVICE);
    EXPECT_EQ(desc.target_tier, Tier::HOST);
    EXPECT_EQ(desc.group_set_id, 0);
    ASSERT_EQ(desc.device_blocks.size(), 1u);
    EXPECT_EQ(desc.device_blocks[0], device_block);

    delete node;
}

TEST_F(FullGroupSetTest, HostLeafDetection) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    // A: evicted from device, has host data
    setHostBlock(a, 0, 15);
    // B: evicted from device, has host data
    setHostBlock(b, 0, 25);

    // B is HostLeaf (no child with host value)
    EXPECT_TRUE(group_->isLeafAtTier(b, Tier::HOST));
    // A is NOT HostLeaf (child B has host value)
    EXPECT_FALSE(group_->isLeafAtTier(a, Tier::HOST));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, HostCandidateEligibility) {
    auto* a = makeNode(100);
    // Evicted from device, has host data
    setHostBlock(a, 0, 15);

    // A host-leaf holding host data is candidate-eligible.
    EXPECT_TRUE(group_->isSlotEvictable(*a, Tier::HOST));

    delete a;
}

TEST_F(FullGroupSetTest, HostCandidateNotEligibleWhenNonLeaf) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;
    setHostBlock(a, 0, 15);
    setHostBlock(b, 0, 25);

    // A has a child still holding host data, so it is not a host-leaf.
    EXPECT_FALSE(group_->isSlotEvictable(*a, Tier::HOST));
    EXPECT_TRUE(group_->isSlotEvictable(*b, Tier::HOST));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, NoDataNotEligible) {
    auto* a = makeNode(100);
    // No data at any tier -> not a leaf at that tier -> not eligible.
    EXPECT_FALSE(group_->isSlotEvictable(*a, Tier::DEVICE));
    delete a;
}

TEST_F(FullGroupSetTest, CompleteDeviceValueRequiresExactPoolCardinalityAndNoNullBlocks) {
    auto second_pool = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "full_group_set_test_second");
    ASSERT_NE(second_pool, nullptr);
    auto two_pool_group                = std::make_shared<FullGroupSet>();
    TestGroupSpec first;
    first.tag                   = "tag_0";
    first.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    first.kv_block_stride_bytes = 1;
    TestGroupSpec second        = first;
    second.tag                  = "tag_1";
    two_pool_group->initialize(0,
                               makeTestTopology({std::move(first), std::move(second)}),
                               {0, 1},
                               {pool_, second_pool});

    GroupSetResource slot;
    EXPECT_FALSE(two_pool_group->hasCompleteDeviceValue(slot));

    slot.device_blocks = {10};
    EXPECT_FALSE(two_pool_group->hasCompleteDeviceValue(slot));

    slot.device_blocks = {10, NULL_BLOCK_IDX};
    EXPECT_FALSE(two_pool_group->hasCompleteDeviceValue(slot));

    slot.device_blocks = {10, 20};
    EXPECT_TRUE(two_pool_group->hasCompleteDeviceValue(slot));

    slot.device_blocks = {10, 20, 30};
    EXPECT_FALSE(two_pool_group->hasCompleteDeviceValue(slot));
}

}  // namespace
}  // namespace rtp_llm
