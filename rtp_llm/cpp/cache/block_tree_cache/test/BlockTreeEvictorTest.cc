#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

namespace rtp_llm {
namespace {

std::shared_ptr<FullComponentGroup> makeFullGroup(int gid) {
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = gid;
    return group;
}

std::shared_ptr<HostBlockPool> makePageableHostPool(size_t usable_blocks) {
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = "block_tree_evictor_test_host";
    config->physical_block_count = usable_blocks + 1;
    config->payload_bytes        = 64;
    config->stride_bytes         = 4096;
    config->enable_pinned        = false;
    config->alignment            = 4096;

    auto pool = std::make_shared<HostBlockPool>(config);
    if (!pool->init()) {
        return nullptr;
    }
    return pool;
}

GroupSlot makeSlot(Tier tier, BlockIdxType block) {
    GroupSlot slot;
    switch (tier) {
        case Tier::DEVICE:
            slot.device_blocks = {block};
            break;
        case Tier::HOST:
            slot.host_block = block;
            break;
        case Tier::DISK:
            slot.disk_slot = block;
            break;
        default:
            break;
    }
    return slot;
}

bool initEvictor(BlockTreeEvictor& evictor) {
    return evictor.init({}, EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO);
}

class BlockTreeEvictorTest: public ::testing::Test {
protected:
    void SetUp() override {
        group_ = makeFullGroup(0);
        groups_.push_back(group_);
        tree_    = std::make_unique<BlockTree>(1);
        evictor_ = std::make_unique<BlockTreeEvictor>(groups_, BlockTreeEvictor::ExecuteTransferFn{}, false);
        ASSERT_TRUE(initEvictor(*evictor_));
    }

    BlockTreeInsertResult insert(const CacheKeysType& keys, const std::vector<std::vector<GroupSlot>>& slots) {
        auto result = tree_->insertNode(nullptr, keys, slots);
        evictor_->onInsertCommitted(result);
        return result;
    }

    std::shared_ptr<FullComponentGroup> group_;
    std::vector<ComponentGroupPtr>      groups_;
    std::unique_ptr<BlockTree>          tree_;
    std::unique_ptr<BlockTreeEvictor>   evictor_;
};

TEST(BlockTreeEvictorInitTest, RejectsNullGroupWithoutThrowing) {
    std::vector<ComponentGroupPtr> groups = {nullptr};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);

    EXPECT_NO_THROW(EXPECT_FALSE(initEvictor(evictor)));
}

TEST(BlockTreeEvictorInitTest, RejectsNegativeGroupIdWithoutThrowing) {
    std::vector<ComponentGroupPtr> groups = {makeFullGroup(-1)};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);

    EXPECT_NO_THROW(EXPECT_FALSE(initEvictor(evictor)));
}

TEST(BlockTreeEvictorInitTest, RejectsOutOfRangeGroupIdWithoutThrowing) {
    std::vector<ComponentGroupPtr> groups = {makeFullGroup(1)};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);

    EXPECT_NO_THROW(EXPECT_FALSE(initEvictor(evictor)));
}

TEST(BlockTreeEvictorInitTest, RejectsGroupIdDifferentFromVectorIndexWithoutThrowing) {
    std::vector<ComponentGroupPtr> groups = {makeFullGroup(0), makeFullGroup(0)};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);

    EXPECT_NO_THROW(EXPECT_FALSE(initEvictor(evictor)));
}

TEST_F(BlockTreeEvictorTest, MatchUpdatesIntermediateHistoryWithoutAdmittingIt) {
    std::vector<std::vector<GroupSlot>> slots = {{makeSlot(Tier::DEVICE, 10)},
                                                  {makeSlot(Tier::DEVICE, 11)}};
    auto                                result = insert({100, 200}, slots);
    ASSERT_EQ(result.inserted_nodes.size(), 2u);

    TreeNode* parent = result.inserted_nodes[0].node;
    TreeNode* leaf   = result.inserted_nodes[1].node;
    ASSERT_NE(parent, nullptr);
    ASSERT_NE(leaf, nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    evictor_->onMatched({parent, leaf});

    const auto parent_meta = parent->group_slots[0].candidate_meta;
    const auto leaf_meta   = leaf->group_slots[0].candidate_meta;
    EXPECT_EQ(parent_meta.last_access_seq, leaf_meta.last_access_seq);
    EXPECT_EQ(parent_meta.hit_count, 1u);
    EXPECT_EQ(leaf_meta.hit_count, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    evictor_->onNodeAboutToRemove(leaf);
    tree_->removeNode(leaf);
    evictor_->onTopologyChanged(parent);

    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);
    auto victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, parent);
    EXPECT_EQ(parent->group_slots[0].candidate_meta.last_access_seq, parent_meta.last_access_seq);
    EXPECT_EQ(parent->group_slots[0].candidate_meta.hit_count, parent_meta.hit_count);
}

TEST_F(BlockTreeEvictorTest, LastReferenceReleaseReadmitsLazyDroppedCandidate) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    group_->setHostPool(host_pool);

    const BlockIdxType block = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(block));
    ASSERT_EQ(host_pool->refCount(block), 1u);

    auto result = insert({100}, {{makeSlot(Tier::HOST, block)}});
    ASSERT_NE(result.leaf, nullptr);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    GroupBlockSet match_set{0, Tier::HOST, {{block}}, {result.leaf}};
    group_->referenceBlocks(match_set);
    group_->referenceBlocks(match_set);
    ASSERT_EQ(host_pool->refCount(block), 3u);

    EXPECT_FALSE(evictor_->chooseVictim(Tier::HOST).has_value());
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    group_->unreferenceBlocks(match_set);
    evictor_->refreshCandidatesAfterRelease(match_set);
    EXPECT_EQ(host_pool->refCount(block), 2u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    group_->unreferenceBlocks(match_set);
    evictor_->refreshCandidatesAfterRelease(match_set);
    EXPECT_EQ(host_pool->refCount(block), 1u);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    auto victim = evictor_->chooseVictim(Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, result.leaf);

    result.leaf->group_slots[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, block);
}

TEST_F(BlockTreeEvictorTest, LoadBackFailureRestoresSourceAndRejectsDuplicateBegin) {
    auto result = insert({100}, {{makeSlot(Tier::HOST, 10)}});
    ASSERT_NE(result.leaf, nullptr);
    auto& slot = result.leaf->group_slots[0];
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    EXPECT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    EXPECT_EQ(slot.transfer_state, SlotTransferState::LOADING_BACK);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    EXPECT_FALSE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    EXPECT_EQ(slot.transfer_state, SlotTransferState::LOADING_BACK);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, false);
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
}

TEST_F(BlockTreeEvictorTest, LoadBackSuccessAdmitsOnlyStableDeviceSlot) {
    auto result = insert({100}, {{makeSlot(Tier::HOST, 10)}});
    ASSERT_NE(result.leaf, nullptr);
    auto& slot = result.leaf->group_slots[0];

    ASSERT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    slot.host_block    = NULL_BLOCK_IDX;
    slot.device_blocks = {20};
    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, true);

    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, result.leaf);
}

TEST_F(BlockTreeEvictorTest, DemotionExcludesSourceAndRollbackOrSuccessRestoresOneTier) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    group_->setHostPool(host_pool);

    auto result = insert({100}, {{makeSlot(Tier::DEVICE, 10)}});
    ASSERT_NE(result.leaf, nullptr);
    auto& slot = result.leaf->group_slots[0];
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto plan = evictor_->buildPlan(*victim);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(plan->primary.target_blocks.size(), 1u);
    EXPECT_EQ(slot.transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 0u);

    evictor_->rollbackPreparedPlan(*plan);
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{10}));
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);

    victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    plan = evictor_->buildPlan(*victim);
    ASSERT_TRUE(plan.has_value());
    const BlockIdxType target_block = plan->primary.target_blocks[0];
    evictor_->complete(*tree_, *plan, BlockTreeEvictor::CopyResultSet{true, {}});

    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::DEVICE));
    EXPECT_EQ(slot.host_block, target_block);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(host_pool->refCount(target_block), 1u);

    slot.host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, target_block);
}

TEST(BlockTreeEvictorStatsTest, AggregatesCandidatesAcrossGroupsAndTiers) {
    auto group0 = makeFullGroup(0);
    auto group1 = makeFullGroup(1);
    std::vector<ComponentGroupPtr> groups = {group0, group1};
    BlockTreeEvictor evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(initEvictor(evictor));

    BlockTree tree(2);
    auto      first = tree.insertNode(nullptr,
                                {100},
                                {{makeSlot(Tier::DEVICE, 10), makeSlot(Tier::DISK, 30)}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::HOST, 20), GroupSlot{}}});
    evictor.onInsertCommitted(second);

    const CandidateStats stats = evictor.candidateStats();
    EXPECT_EQ(stats.device_candidates, 1u);
    EXPECT_EQ(stats.host_candidates, 1u);
    EXPECT_EQ(stats.disk_candidates, 1u);

    EXPECT_FALSE(evictor.chooseVictim(0, Tier::DISK).has_value());
    auto disk_victim = evictor.chooseVictim(1, Tier::DISK);
    ASSERT_TRUE(disk_victim.has_value());
    EXPECT_EQ(disk_victim->node, first.leaf);
    EXPECT_EQ(disk_victim->component_group_id, 1);
}

TEST(BlockTreeEvictorPolicyTest, MatchDoesNotChangeFifoAdmissionOrder) {
    auto group = makeFullGroup(0);
    std::vector<ComponentGroupPtr> groups = {group};
    BlockTreeEvictor evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(evictor.init({}, EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO));

    BlockTree tree(1);
    auto first = tree.insertNode(nullptr, {100}, {{makeSlot(Tier::DEVICE, 10)}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::DEVICE, 20)}});
    evictor.onInsertCommitted(second);
    const uint64_t first_admission = first.leaf->group_slots[0].candidate_meta.admission_seq;

    evictor.onMatched({first.leaf});

    EXPECT_EQ(first.leaf->group_slots[0].candidate_meta.admission_seq, first_admission);
    auto victim = evictor.chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, first.leaf);
}

TEST(BlockTreeEvictorPolicyTest, MatchUpdatesLfuHitCountAndOrder) {
    auto group = makeFullGroup(0);
    std::vector<ComponentGroupPtr> groups = {group};
    BlockTreeEvictor evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(evictor.init({}, EvictionPolicy::LFU, EvictionPolicy::LRU, EvictionPolicy::FIFO));

    BlockTree tree(1);
    auto first = tree.insertNode(nullptr, {100}, {{makeSlot(Tier::DEVICE, 10)}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::DEVICE, 20)}});
    evictor.onInsertCommitted(second);

    evictor.onMatched({first.leaf});

    EXPECT_EQ(first.leaf->group_slots[0].candidate_meta.hit_count, 1u);
    EXPECT_EQ(second.leaf->group_slots[0].candidate_meta.hit_count, 0u);
    auto victim = evictor.chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, second.leaf);
}

}  // namespace
}  // namespace rtp_llm
