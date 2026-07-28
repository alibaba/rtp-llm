#include <gtest/gtest.h>

#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::TestGroupSpec;

class SWAGroupSetTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = block_tree_cache_test::makeDevicePool({{1, 0}}, 128, "swa_group_set_test");
        ASSERT_NE(pool_, nullptr);
        group_ = std::make_shared<SWAGroupSet>(
            /*sliding_window_size=*/128,
            /*seq_size_per_block=*/64);
        TestGroupSpec spec;
        spec.tag                        = "tag_0";
        spec.policy                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
        spec.policy.enable_prefix_reuse = true;
        spec.policy.sliding_window_size = 128;
        spec.kv_block_stride_bytes      = 1;
        spec.seq_size_per_block         = 64;
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

    DeviceBlockPoolPtr               pool_;
    std::unordered_set<BlockIdxType> held_blocks_;
    std::shared_ptr<SWAGroupSet>     group_;
};

TEST_F(SWAGroupSetTest, AnyNodeWithDataIsSlotEvictable) {
    // SWA allows any node holding data at the tier to be a candidate (not just leaves).
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;

    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);

    // Both A and B are candidate-eligible even though A has a child holding data.
    EXPECT_TRUE(group_->isSlotEvictable(*a, Tier::DEVICE));
    EXPECT_TRUE(group_->isSlotEvictable(*b, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(SWAGroupSetTest, WindowValidatorConnectedPath) {
    auto  validator     = group_->createMatchValidator();
    auto* swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());

    // Path: root -> A (has data) -> B (has data) -> C (has data)
    auto* a = makeNode(100);
    auto* b = makeNode(200);
    auto* c = makeNode(300);
    setDeviceBlock(a, 0);
    setDeviceBlock(b, 0);
    setDeviceBlock(c, 0);

    EXPECT_TRUE(validator->validate(a, a->group_set_resources[0]));
    EXPECT_TRUE(validator->validate(b, b->group_set_resources[0]));
    EXPECT_TRUE(validator->validate(c, c->group_set_resources[0]));

    EXPECT_TRUE(swa_validator->connectedToRoot());
    // 3 blocks * 64 tokens = 192
    EXPECT_EQ(swa_validator->accumulatedLength(), 192u);

    delete a;
    delete b;
    delete c;
}

TEST_F(SWAGroupSetTest, WindowValidatorGapRequiresEnoughWindowAfterReset) {
    std::unique_ptr<MatchValidator> validator     = group_->createMatchValidator();
    SWAMatchValidator*              swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());

    TreeNode* a = makeNode(100);
    TreeNode* b = makeNode(200);
    TreeNode* c = makeNode(300);
    TreeNode* d = makeNode(400);
    setDeviceBlock(a, 0);
    setDeviceBlock(c, 0);
    setDeviceBlock(d, 0);

    EXPECT_TRUE(validator->validate(a, a->group_set_resources[0]));
    EXPECT_TRUE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 64u);

    EXPECT_FALSE(validator->validate(b, b->group_set_resources[0]));
    EXPECT_FALSE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 0u);

    EXPECT_FALSE(validator->validate(c, c->group_set_resources[0]));
    EXPECT_EQ(swa_validator->accumulatedLength(), 64u);

    EXPECT_TRUE(validator->validate(d, d->group_set_resources[0]));
    EXPECT_EQ(swa_validator->accumulatedLength(), 128u);

    delete a;
    delete b;
    delete c;
    delete d;
}

TEST_F(SWAGroupSetTest, WindowValidatorMultitierNoReset) {
    auto validator = group_->createMatchValidator();

    // B has no device data but has host data -> should not reset
    auto* a = makeNode(100);
    auto* b = makeNode(200);
    auto* c = makeNode(300);
    setDeviceBlock(a, 0);
    // B: host data only
    clearDeviceBlock(b, 0);
    b->group_set_resources[0].host_block = 15;
    setDeviceBlock(c, 0);

    EXPECT_TRUE(validator->validate(a, a->group_set_resources[0]));
    // B has host data -> !is_empty() is true -> no reset
    EXPECT_TRUE(validator->validate(b, b->group_set_resources[0]));
    EXPECT_TRUE(validator->validate(c, c->group_set_resources[0]));

    auto* swa_validator = dynamic_cast<SWAMatchValidator*>(validator.get());
    EXPECT_TRUE(swa_validator->connectedToRoot());
    EXPECT_EQ(swa_validator->accumulatedLength(), 192u);

    delete a;
    delete b;
    delete c;
}

TEST_F(SWAGroupSetTest, ComputeReferenceCountCountsHostAndDiskBlocks) {
    TreeNode* a = makeNode(100);
    TreeNode* b = makeNode(200);
    TreeNode* c = makeNode(300);
    TreeNode* d = makeNode(400);

    setHostBlock(a, 0, 10);
    a->group_set_resources[0].disk_slot = 11;
    setHostBlock(b, 0, 20);
    b->group_set_resources[0].disk_slot = 21;
    setHostBlock(c, 0, 30);
    c->group_set_resources[0].disk_slot = 31;
    setHostBlock(d, 0, 40);
    d->group_set_resources[0].disk_slot = 41;

    std::vector<TreeNode*> path = {a, b, c, d};
    EXPECT_EQ(group_->computeReuseBlockCount(path.size(), path), 2u);

    delete a;
    delete b;
    delete c;
    delete d;
}

TEST_F(SWAGroupSetTest, IndependentEvictionDoesNotAffectFull) {
    // SWA eviction only affects SWA group data, not Full group
    auto*              node       = makeNode(100, 2);         // 2 groups: 0=Full, 1=SWA
    const BlockIdxType full_block = setDeviceBlock(node, 0);  // Full data
    setDeviceBlock(node, 1);                                  // SWA data

    group_->evictFromTier(node, node->group_set_resources[1], Tier::DEVICE);

    // SWA data cleared
    EXPECT_FALSE(node->group_set_resources[1].hasTier(Tier::DEVICE));
    // Full data intact
    EXPECT_TRUE(node->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_EQ(node->group_set_resources[0].device_blocks[0], full_block);

    delete node;
}

TEST_F(SWAGroupSetTest, SlotEvictabilityRequiresTierDataButNotLeafTopology) {
    auto* a          = makeNode(100);
    auto* b          = makeNode(200);
    a->children[200] = b;
    b->parent        = a;
    setDeviceBlock(a, 0);

    EXPECT_TRUE(group_->isSlotEvictable(*a, Tier::DEVICE));
    EXPECT_FALSE(group_->isSlotEvictable(*a, Tier::HOST));
    EXPECT_FALSE(group_->isSlotEvictable(*b, Tier::DEVICE));

    delete a;
    delete b;
}

TEST_F(SWAGroupSetTest, SlidingWindowConfig) {
    EXPECT_EQ(group_->slidingWindowSize(), 128);
    EXPECT_EQ(group_->seqSizePerBlock(), 64);
}

}  // namespace
}  // namespace rtp_llm
