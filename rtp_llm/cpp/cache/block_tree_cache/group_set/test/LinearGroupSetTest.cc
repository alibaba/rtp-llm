#include <gtest/gtest.h>

#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::TestGroupSpec;

class LinearGroupSetTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "linear_group_set_test");
        ASSERT_NE(pool_, nullptr);
        group_ = std::make_shared<LinearGroupSet>();
        TestGroupSpec spec;
        spec.tag                   = "tag_0";
        spec.policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
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

    DeviceBlockPoolPtr               pool_;
    std::unordered_set<BlockIdxType> held_blocks_;
    std::shared_ptr<LinearGroupSet>  group_;
};

TEST_F(LinearGroupSetTest, AnyNodeWithDataIsSlotEvictable) {
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

TEST_F(LinearGroupSetTest, EvictFromTierDevice) {
    auto* node = makeNode(100);
    setDeviceBlock(node, 0);

    group_->evictFromTier(node, node->group_set_resources[0], Tier::DEVICE);

    // Device blocks are cleared; heap ownership no longer lives on the group.
    EXPECT_FALSE(node->group_set_resources[0].hasTier(Tier::DEVICE));

    delete node;
}

TEST_F(LinearGroupSetTest, MatchValidatorHasData) {
    auto validator = group_->createMatchValidator();

    auto* node_with = makeNode(100);
    setDeviceBlock(node_with, 0);
    EXPECT_TRUE(validator->validate(node_with, node_with->group_set_resources[0]));

    auto* node_empty = makeNode(200);
    EXPECT_FALSE(validator->validate(node_empty, node_empty->group_set_resources[0]));

    delete node_with;
    delete node_empty;
}

TEST_F(LinearGroupSetTest, MatchValidatorBusySlotInvalidWithoutLatch) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto validator = group_->createMatchValidator();

        auto* busy_node = makeNode(100);
        setDeviceBlock(busy_node, 0);
        busy_node->group_set_resources[0].transfer_state = state;

        auto* usable_node = makeNode(200);
        setDeviceBlock(usable_node, 0);

        // Point-state semantics: a busy node is unusable on its own but must
        // not poison later nodes.
        EXPECT_FALSE(validator->validate(busy_node, busy_node->group_set_resources[0]));
        EXPECT_TRUE(validator->validate(usable_node, usable_node->group_set_resources[0]));

        delete busy_node;
        delete usable_node;
    }
}

TEST_F(LinearGroupSetTest, MatchValidatorAllowsLoadingSlot) {
    auto validator = group_->createMatchValidator();

    auto* node = makeNode(100);
    node->group_set_resources[0].host_block    = 15;
    node->group_set_resources[0].transfer_state = GroupSetTransferState::LOADING;

    EXPECT_TRUE(validator->validate(node, node->group_set_resources[0]));

    delete node;
}

TEST_F(LinearGroupSetTest, EmptySlotIsNotEvictable) {
    auto* node = makeNode(100);
    EXPECT_FALSE(group_->isSlotEvictable(*node, Tier::DEVICE));
    EXPECT_FALSE(group_->isSlotEvictable(*node, Tier::HOST));
    delete node;
}

}  // namespace
}  // namespace rtp_llm
