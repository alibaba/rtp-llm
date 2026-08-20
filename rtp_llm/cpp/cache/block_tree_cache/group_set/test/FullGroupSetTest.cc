#include <gtest/gtest.h>

#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestTopology;

class FullGroupSetTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "full_group_set_test");
        ASSERT_NE(pool_, nullptr);
        host_pool_ = block_tree_cache_test::makeHostPool(1, 128);
        ASSERT_NE(host_pool_, nullptr);
        group_     = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_}, host_pool_, nullptr);
        auto group = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 1);
        group_->initialize(0, makeTestTopology({std::move(group)}), {0});
        tree_ = std::make_unique<BlockTree>(std::vector<GroupSetPtr>{group_});
    }

    void TearDown() override {
        for (const auto block : held_blocks_) {
            pool_->decRef(block);
        }
        for (const auto block : held_host_blocks_) {
            host_pool_->decTreeRef(block, BlockTreeRefType::CACHE);
        }
    }

    TreeNode* makeNode(CacheKeyType key, int group_count = 1) {
        auto* node      = new TreeNode();
        node->cache_key = key;
        node->group_set_resources.resize(static_cast<size_t>(group_count));
        return node;
    }

    BlockIdxType setDeviceBlock(TreeNode* node, int group_set_id) {
        const auto block = pool_->malloc();
        EXPECT_TRUE(block.has_value());
        if (!block.has_value()) {
            return NULL_BLOCK_IDX;
        }
        pool_->incRef(block.value());
        held_blocks_.insert(block.value());
        node->group_set_resources[static_cast<size_t>(group_set_id)].device_blocks = {block.value()};
        return block.value();
    }

    void clearDeviceBlock(TreeNode* node, int group_set_id) {
        node->group_set_resources[static_cast<size_t>(group_set_id)].device_blocks = {NULL_BLOCK_IDX};
    }

    void setHostBlock(TreeNode* node, int group_set_id) {
        const auto block = host_pool_->malloc();
        EXPECT_TRUE(block.has_value());
        if (!block.has_value()) {
            return;
        }
        host_pool_->incTreeRef(block.value(), BlockTreeRefType::CACHE);
        held_host_blocks_.insert(block.value());
        node->group_set_resources[static_cast<size_t>(group_set_id)].host_block = block.value();
    }

    void setDiskSlot(TreeNode* node, int group_set_id, BlockIdxType disk_block) {
        node->group_set_resources[static_cast<size_t>(group_set_id)].disk_slot = disk_block;
    }

    DeviceBlockPoolPtr               pool_;
    std::shared_ptr<HostBlockPool>   host_pool_;
    std::unordered_set<BlockIdxType> held_blocks_;
    std::unordered_set<BlockIdxType> held_host_blocks_;
    std::shared_ptr<FullGroupSet>    group_;
    std::unique_ptr<BlockTree>       tree_;
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
    EXPECT_TRUE(tree_->isLeafAtTier(c, 0, Tier::DEVICE));
    // B is NOT DeviceLeaf (child C has device value)
    EXPECT_FALSE(tree_->isLeafAtTier(b, 0, Tier::DEVICE));
    // A is NOT DeviceLeaf (child B has device value)
    EXPECT_FALSE(tree_->isLeafAtTier(a, 0, Tier::DEVICE));

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

    EXPECT_FALSE(tree_->isLeafAtTier(a, 0, Tier::DEVICE));

    // Evict B's device data
    clearDeviceBlock(b, 0);

    // Now A should be DeviceLeaf
    EXPECT_TRUE(tree_->isLeafAtTier(a, 0, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, EvictFromTierDevice) {
    auto* a = makeNode(100);
    setDeviceBlock(a, 0);

    a->group_set_resources[0].evictFromTier(Tier::DEVICE);

    // Device blocks should be cleared
    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::DEVICE));

    delete a;
}

TEST_F(FullGroupSetTest, EvictFromTierHost) {
    auto* a = makeNode(100);
    setHostBlock(a, 0);

    a->group_set_resources[0].evictFromTier(Tier::HOST);

    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::HOST));

    delete a;
}

TEST_F(FullGroupSetTest, EvictFromTierDisk) {
    auto* a = makeNode(100);
    setDiskSlot(a, 0, 8);

    a->group_set_resources[0].evictFromTier(Tier::DISK);

    EXPECT_FALSE(a->group_set_resources[0].hasTier(Tier::DISK));

    delete a;
}

TEST_F(FullGroupSetTest, MatchValidatorFullPathValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setDeviceBlock(node, 0);

    EXPECT_TRUE(validator->validate(node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, MatchValidatorHostDataValid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setHostBlock(node, 0);

    EXPECT_TRUE(validator->validate(node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, MatchValidatorEmptyInvalid) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    // No data in any tier

    EXPECT_FALSE(validator->validate(node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, MatchValidatorBusyResourceBreaksPrefixLikeHole) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto validator = group_->createMatchValidator();

        auto* busy_node = makeNode(100);
        setDeviceBlock(busy_node, 0);
        busy_node->group_set_resources[0].transfer_state = state;

        auto* usable_node = makeNode(200);
        setDeviceBlock(usable_node, 0);

        // Busy resource is unusable, and the FULL prefix latch keeps later usable
        // nodes invalid so device reuse stays contiguous from the root.
        EXPECT_FALSE(validator->validate(busy_node->group_set_resources[0]));
        EXPECT_FALSE(validator->validate(usable_node->group_set_resources[0]));

        delete busy_node;
        delete usable_node;
    }
}

TEST_F(FullGroupSetTest, MatchValidatorAllowsLoadingResource) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    setHostBlock(node, 0);
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOADING;

    EXPECT_TRUE(validator->validate(node->group_set_resources[0]));

    delete node;
}

TEST_F(FullGroupSetTest, HostLeafDetection) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    // A: evicted from device, has host data
    setHostBlock(a, 0);
    // B: evicted from device, has host data
    setHostBlock(b, 0);

    // B is HostLeaf (no child with host value)
    EXPECT_TRUE(tree_->isLeafAtTier(b, 0, Tier::HOST));
    // A is NOT HostLeaf (child B has host value)
    EXPECT_FALSE(tree_->isLeafAtTier(a, 0, Tier::HOST));

    delete a;
    delete b;
}

TEST_F(FullGroupSetTest, CompleteDeviceValueRequiresDeviceTierAndNoNullBlocks) {
    auto second_pool = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "full_group_set_test_second");
    ASSERT_NE(second_pool, nullptr);
    auto two_pool_group =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_, second_pool}, nullptr, nullptr);
    auto      first          = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 1);
    GroupBase second         = first;
    two_pool_group->initialize(
        0, makeTestTopology({std::move(first), std::move(second)}), {0, 1});

    GroupSetResource resource;
    EXPECT_FALSE(resource.hasCompleteDeviceValue());

    resource.device_blocks = {10, NULL_BLOCK_IDX};
    EXPECT_FALSE(resource.hasCompleteDeviceValue());

    resource.device_blocks = {10, 20};
    EXPECT_TRUE(resource.hasCompleteDeviceValue());
}

}  // namespace
}  // namespace rtp_llm
