#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
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

class NoopDiskBlockIO final: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(uint64_t, void*, size_t) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(uint64_t, const void*, size_t) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(const std::vector<DiskRead>&) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(const std::vector<DiskWrite>&) override {
        return DiskBlockIOStatus::OK;
    }

    void close() override {}

    std::string debugString() const override {
        return "NoopDiskBlockIO";
    }
};

std::shared_ptr<BlockTreeDiskBlockPool> makeTestDiskPool(size_t usable_blocks, const std::string& name) {
    auto config                  = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type            = BlockPoolType::DISK;
    config->pool_name            = name;
    config->work_dir             = "/tmp";
    config->payload_bytes        = 64;
    config->stride_bytes         = 64;
    config->disk_size_bytes      = (usable_blocks + 1) * config->stride_bytes;
    config->physical_block_count = usable_blocks + 1;
    config->buffered_io          = true;

    auto pool = std::make_shared<BlockTreeDiskBlockPool>(config, std::make_unique<NoopDiskBlockIO>());
    if (!pool->init()) {
        return nullptr;
    }
    return pool;
}

DeviceBlockPoolPtr makeTestDevicePool(size_t usable_blocks, const std::string& name) {
    const size_t physical_blocks = usable_blocks + 1;
    const size_t block_bytes     = 16;

    MemoryLayoutConfig layout;
    layout.layer_num                  = 1;
    layout.block_num                  = static_cast<uint32_t>(physical_blocks);
    layout.dtype                      = TYPE_INT8;
    layout.kv_cache_offset_bytes      = 0;
    layout.kv_block_stride_bytes      = block_bytes;
    layout.kv_block_pool_size_bytes   = physical_blocks * block_bytes;
    layout.block_stride_bytes         = block_bytes;
    layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
    layout.local_head_num_kv          = 1;
    layout.seq_size_per_block         = 1;
    layout.kernel_blocks_per_kv_block = 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = name;
    config->physical_block_count    = physical_blocks;
    config->total_size_bytes        = layout.total_size_bytes;
    config->memory_layouts          = {layout};
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
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

class BlockTreeEvictorTestPeer {
public:
    static EvictionMove
    makeMove(BlockTreeEvictor& evictor, TreeNode* node, int group_id, Tier source_tier, Tier target_tier) {
        return evictor.makeMove(node, group_id, source_tier, target_tier);
    }

    static bool prepareMove(BlockTreeEvictor& evictor, EvictionMove& move) {
        return evictor.prepareMove(move);
    }

    static std::vector<int>
    selectCascadeGroups(BlockTreeEvictor& evictor, TreeNode* node, int group_id, Tier tier, bool reverse) {
        return evictor.selectCascadeGroups(node, group_id, tier, reverse);
    }

    static void rollbackMove(BlockTreeEvictor& evictor, const EvictionMove& move) {
        BlockTreeEvictor::EvictionPlan plan;
        plan.primary = move;
        evictor.rollbackPreparedPlan(plan);
    }
};

std::vector<int> cascadeGroupIds(const BlockTreeEvictor::EvictionPlan& plan) {
    std::vector<int> result;
    result.reserve(plan.cascade_moves.size());
    for (const EvictionMove& move : plan.cascade_moves) {
        result.push_back(move.component_group_id);
    }
    return result;
}

std::vector<BlockIdxType> exhaustPool(IBlockPool& pool) {
    std::vector<BlockIdxType> blocks;
    while (true) {
        auto block = pool.malloc();
        if (!block.has_value()) {
            break;
        }
        pool.incRef(*block);
        blocks.push_back(*block);
    }
    return blocks;
}

void releaseBlocks(IBlockPool& pool, const std::vector<BlockIdxType>& blocks) {
    for (BlockIdxType block : blocks) {
        pool.decRef(block);
    }
}

class CascadeTestEnvironment {
public:
    bool init(bool enable_reverse_eviction = false) {
        auto full                  = makeFullGroup(0);
        auto swa                   = std::make_shared<SWAComponentGroup>(2, 1);
        swa->component_group_id    = 1;
        auto linear                = std::make_shared<LinearComponentGroup>();
        linear->component_group_id = 2;
        groups_                    = {full, swa, linear};

        for (size_t gid = 0; gid < groups_.size(); ++gid) {
            auto host = makePageableHostPool(2);
            auto disk = makeTestDiskPool(2, "block_tree_evictor_cascade_" + std::to_string(gid));
            if (host == nullptr || disk == nullptr) {
                return false;
            }
            groups_[gid]->setHostPool(host);
            groups_[gid]->setDiskPool(disk);
            host_pools_.push_back(std::move(host));
            disk_pools_.push_back(std::move(disk));
        }

        evictor_ = std::make_unique<BlockTreeEvictor>(
            groups_,
            [this](const TransferDescriptor& descriptor) {
                transfer_group_ids_.push_back(descriptor.component_group_id);
                if (transfer_statuses_.empty()) {
                    return CopyStatus::OK;
                }
                const CopyStatus status = transfer_statuses_.front();
                transfer_statuses_.pop_front();
                return status;
            },
            enable_reverse_eviction);
        if (!initEvictor(*evictor_)) {
            return false;
        }

        std::vector<GroupSlot> slots(groups_.size());
        host_blocks_.resize(groups_.size(), NULL_BLOCK_IDX);
        for (size_t gid = 0; gid < groups_.size(); ++gid) {
            host_blocks_[gid] = groups_[gid]->allocateSingleBlock(Tier::HOST);
            if (isNullBlockIdx(host_blocks_[gid])) {
                return false;
            }
            slots[gid].host_block = host_blocks_[gid];
        }

        tree_       = std::make_unique<BlockTree>(static_cast<int>(groups_.size()));
        auto result = tree_->insertNode(nullptr, {100}, {slots});
        evictor_->onInsertCommitted(result);
        node_ = result.leaf;
        return node_ != nullptr;
    }

    std::optional<BlockTreeEvictor::EvictionPlan> buildPlan(int primary_group_id) {
        auto victim = evictor_->chooseVictim(primary_group_id, Tier::HOST);
        if (!victim.has_value()) {
            return std::nullopt;
        }
        return evictor_->buildPlan(*victim);
    }

    GroupBlockSet hostSet(int group_id) const {
        return GroupBlockSet{group_id, Tier::HOST, {{host_blocks_[static_cast<size_t>(group_id)]}}, {node_}};
    }

    void setTransferStatuses(std::initializer_list<CopyStatus> statuses) {
        transfer_statuses_.assign(statuses.begin(), statuses.end());
        transfer_group_ids_.clear();
    }

    void releaseResidentBlocks() {
        for (size_t gid = 0; gid < groups_.size(); ++gid) {
            auto& slot = node_->group_slots[gid];
            if (slot.has_value(Tier::HOST)) {
                const BlockIdxType block = slot.host_block;
                slot.host_block          = NULL_BLOCK_IDX;
                groups_[gid]->releaseSingleBlock(Tier::HOST, block);
            }
            if (slot.has_value(Tier::DISK)) {
                const BlockIdxType block = slot.disk_slot;
                slot.disk_slot           = NULL_BLOCK_IDX;
                groups_[gid]->releaseSingleBlock(Tier::DISK, block);
            }
        }
    }

    void expectAllPoolsFree() const {
        for (size_t gid = 0; gid < groups_.size(); ++gid) {
            EXPECT_EQ(host_pools_[gid]->freeBlocksNum(), 2u);
            EXPECT_EQ(disk_pools_[gid]->freeBlocksNum(), 2u);
        }
    }

    std::vector<ComponentGroupPtr>                       groups_;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools_;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools_;
    std::vector<BlockIdxType>                            host_blocks_;
    std::unique_ptr<BlockTree>                           tree_;
    std::unique_ptr<BlockTreeEvictor>                    evictor_;
    TreeNode*                                            node_{nullptr};
    std::deque<CopyStatus>                               transfer_statuses_;
    std::vector<int>                                     transfer_group_ids_;
};

class BlockTreeEvictorTest: public ::testing::Test {
protected:
    void SetUp() override {
        group_                = makeFullGroup(0);
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(test_info, nullptr);
        device_pool_ = makeTestDevicePool(128, "block_tree_evictor_fixture_" + std::string(test_info->name()));
        ASSERT_NE(device_pool_, nullptr);
        group_->setDevicePools({device_pool_}, {"tag_0"});
        groups_.push_back(group_);
        tree_    = std::make_unique<BlockTree>(1);
        evictor_ = std::make_unique<BlockTreeEvictor>(
            groups_,
            [this](const TransferDescriptor&) {
                ++transfer_calls_;
                return CopyStatus::OK;
            },
            false);
        ASSERT_TRUE(initEvictor(*evictor_));
    }

    BlockTreeInsertResult insert(const CacheKeysType& keys, const std::vector<std::vector<GroupSlot>>& slots) {
        auto result = tree_->insertNode(nullptr, keys, slots);
        for (const BlockTreeInsertedNode& inserted : result.inserted_nodes) {
            TreeNode* node = inserted.node;
            if (node == nullptr) {
                continue;
            }
            for (const ComponentGroupPtr& group : groups_) {
                if (group == nullptr || group->component_group_id < 0) {
                    continue;
                }
                const size_t gid = static_cast<size_t>(group->component_group_id);
                if (gid >= node->group_slots.size()) {
                    continue;
                }
                GroupSlot& slot = node->group_slots[gid];
                if (!slot.has_value(Tier::DEVICE)) {
                    continue;
                }
                const auto blocks = group->getBlocks(slot, Tier::DEVICE);
                if (!blocks.empty()) {
                    group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}});
                }
            }
        }
        for (const BlockTreeAdoptedSlot& adopted : result.adopted_slots) {
            if (adopted.node == nullptr || adopted.component_group_id < 0) {
                continue;
            }
            const size_t gid = static_cast<size_t>(adopted.component_group_id);
            if (gid >= groups_.size() || groups_[gid] == nullptr || gid >= adopted.node->group_slots.size()) {
                continue;
            }
            const ComponentGroupPtr& group  = groups_[gid];
            const auto               blocks = group->getBlocks(adopted.node->group_slots[gid], Tier::DEVICE);
            if (!blocks.empty()) {
                group->referenceBlocks(GroupBlockSet{adopted.component_group_id, Tier::DEVICE, {blocks}});
            }
        }
        evictor_->onInsertCommitted(result);
        return result;
    }

    std::shared_ptr<FullComponentGroup> group_;
    DeviceBlockPoolPtr                  device_pool_;
    std::vector<ComponentGroupPtr>      groups_;
    std::unique_ptr<BlockTree>          tree_;
    std::unique_ptr<BlockTreeEvictor>   evictor_;
    size_t                              transfer_calls_{0};
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

TEST(BlockTreeEvictorPolicyTest, ForwardCascadeIncludesOnlyChainReceivers) {
    auto full                             = makeFullGroup(0);
    auto swa                              = std::make_shared<SWAComponentGroup>(2, 1);
    swa->component_group_id               = 1;
    auto linear                           = std::make_shared<LinearComponentGroup>();
    linear->component_group_id            = 2;
    full->group_type                      = CacheGroupType::FULL;
    swa->group_type                       = CacheGroupType::SWA;
    linear->group_type                    = CacheGroupType::LINEAR;
    full->evict_policy                    = CacheEvictPolicy::CHAIN;
    swa->evict_policy                     = CacheEvictPolicy::INDEPENDENT;
    linear->evict_policy                  = CacheEvictPolicy::NONE;
    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);

    EXPECT_TRUE(BlockTreeEvictorTestPeer::selectCascadeGroups(
                    evictor, /*node=*/nullptr, /*source_group_id=*/0, Tier::HOST, /*reverse=*/false)
                    .empty());
    EXPECT_TRUE(BlockTreeEvictorTestPeer::selectCascadeGroups(
                    evictor, /*node=*/nullptr, /*source_group_id=*/1, Tier::HOST, /*reverse=*/false)
                    .empty());

    linear->evict_policy = CacheEvictPolicy::CHAIN;
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroups(
                  evictor, /*node=*/nullptr, /*source_group_id=*/0, Tier::HOST, /*reverse=*/false),
              (std::vector<int>{2}));
}

TEST(BlockTreeEvictorPolicyTest, ReverseCascadeNeitherStartsFromNorTargetsNonChainGroups) {
    auto full                             = makeFullGroup(0);
    auto swa                              = std::make_shared<SWAComponentGroup>(2, 1);
    swa->component_group_id               = 1;
    auto linear                           = std::make_shared<LinearComponentGroup>();
    linear->component_group_id            = 2;
    full->evict_policy                    = CacheEvictPolicy::CHAIN;
    swa->evict_policy                     = CacheEvictPolicy::INDEPENDENT;
    linear->evict_policy                  = CacheEvictPolicy::NONE;
    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, true);

    BlockTree tree(3);
    auto      inserted =
        tree.insertNode(nullptr, {100}, {{makeSlot(Tier::HOST, 1), makeSlot(Tier::HOST, 2), makeSlot(Tier::HOST, 3)}});
    ASSERT_NE(inserted.leaf, nullptr);
    EXPECT_TRUE(BlockTreeEvictorTestPeer::selectCascadeGroups(
                    evictor, inserted.leaf, /*source_group_id=*/0, Tier::HOST, /*reverse=*/true)
                    .empty());
    EXPECT_TRUE(BlockTreeEvictorTestPeer::selectCascadeGroups(
                    evictor, inserted.leaf, /*source_group_id=*/1, Tier::HOST, /*reverse=*/true)
                    .empty());

    linear->evict_policy = CacheEvictPolicy::CHAIN;
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroups(
                  evictor, inserted.leaf, /*source_group_id=*/0, Tier::HOST, /*reverse=*/true),
              (std::vector<int>{2}));
}

TEST_F(BlockTreeEvictorTest, MatchUpdatesIntermediateHistoryWithoutAdmittingIt) {
    const auto allocated = device_pool_->malloc(2);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    const BlockIdxType                  parent_block = (*allocated)[0];
    const BlockIdxType                  leaf_block   = (*allocated)[1];
    std::vector<std::vector<GroupSlot>> slots        = {{makeSlot(Tier::DEVICE, parent_block)},
                                                        {makeSlot(Tier::DEVICE, leaf_block)}};
    auto                                result       = insert({100, 200}, slots);
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
    group_->unreferenceBlocks(GroupBlockSet{0, Tier::DEVICE, {{leaf_block}}});
    leaf->group_slots[0].device_blocks.clear();
    tree_->removeNode(leaf);
    evictor_->onTopologyChanged(parent);

    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);
    auto victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, parent);
    EXPECT_EQ(parent->group_slots[0].candidate_meta.last_access_seq, parent_meta.last_access_seq);
    EXPECT_EQ(parent->group_slots[0].candidate_meta.hit_count, parent_meta.hit_count);

    evictor_->onNodeAboutToRemove(parent);
    group_->unreferenceBlocks(GroupBlockSet{0, Tier::DEVICE, {{parent_block}}});
    parent->group_slots[0].device_blocks.clear();
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

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsNewRequestPinWithoutAllocatingTarget) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_pin");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeSlot(Tier::HOST, source)}});
    ASSERT_NE(result.leaf, nullptr);

    EvictionMove  stale = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);
    GroupBlockSet pin{0, Tier::HOST, {{source}}, {result.leaf}};
    group_->referenceBlocks(pin);
    ASSERT_EQ(host_pool->refCount(source), 2u);

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(result.leaf->group_slots[0].host_block, source);
    EXPECT_EQ(host_pool->refCount(source), 2u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(transfer_calls_, 0u);

    group_->unreferenceBlocks(pin);
    evictor_->refreshCandidatesAfterRelease(pin);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    result.leaf->group_slots[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, PrepareMovePreservesLoadBackOwner) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_load_back");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeSlot(Tier::HOST, source)}});
    ASSERT_NE(result.leaf, nullptr);
    EvictionMove stale = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);

    ASSERT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    ASSERT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::LOADING_BACK);
    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::LOADING_BACK);
    EXPECT_EQ(result.leaf->group_slots[0].host_block, source);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, false);
    EXPECT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    result.leaf->group_slots[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source);
}

TEST_F(BlockTreeEvictorTest, PrepareMovePreservesExistingDemotionOwnerAndTarget) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_demotion");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeSlot(Tier::HOST, source)}});
    ASSERT_NE(result.leaf, nullptr);

    EvictionMove stale = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);
    EvictionMove owner = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);
    ASSERT_TRUE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, owner));
    ASSERT_EQ(owner.target_blocks.size(), 1u);
    const BlockIdxType owner_target = owner.target_blocks[0];
    ASSERT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::DEMOTING);

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(result.leaf->group_slots[0].host_block, source);
    EXPECT_TRUE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->refCount(owner_target), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(transfer_calls_, 0u);

    BlockTreeEvictorTestPeer::rollbackMove(*evictor_, owner);
    EXPECT_EQ(result.leaf->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_FALSE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    result.leaf->group_slots[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsSourceTierChangedByLoadBack) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_tier_change");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeSlot(Tier::HOST, source)}});
    ASSERT_NE(result.leaf, nullptr);
    EvictionMove stale = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);

    ASSERT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    auto&         slot       = result.leaf->group_slots[0];
    GroupBlockSet device_set = group_->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(device_set.per_node.size(), 1u);
    ASSERT_EQ(device_set.per_node.front().size(), 1u);
    const BlockIdxType device_block = device_set.per_node.front().front();
    slot.device_blocks              = device_set.per_node.front();
    group_->unreferenceBlocks(GroupBlockSet{0, Tier::HOST, {{source}}});
    group_->evictFromTier(result.leaf, slot, Tier::HOST);
    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, true);
    ASSERT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    ASSERT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    ASSERT_FALSE(host_pool->isAllocated(source));

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    slot.device_blocks = {NULL_BLOCK_IDX};
    group_->unreferenceBlocks(device_set);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsFullNodeThatBecameNonLeaf) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_topology");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);

    const BlockIdxType parent_source = group_->allocateSingleBlock(Tier::HOST);
    const BlockIdxType child_source  = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(parent_source));
    ASSERT_FALSE(isNullBlockIdx(child_source));
    auto parent_result = insert({100}, {{makeSlot(Tier::HOST, parent_source)}});
    ASSERT_NE(parent_result.leaf, nullptr);
    EvictionMove stale = BlockTreeEvictorTestPeer::makeMove(*evictor_, parent_result.leaf, 0, Tier::HOST, Tier::DISK);

    auto child_result = tree_->insertNode(parent_result.leaf, {101}, {{makeSlot(Tier::HOST, child_source)}});
    evictor_->onInsertCommitted(child_result);
    ASSERT_NE(child_result.leaf, nullptr);
    ASSERT_FALSE(group_->isSlotEvictable(*parent_result.leaf, Tier::HOST));

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(parent_result.leaf->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(parent_result.leaf->group_slots[0].host_block, parent_source);
    EXPECT_EQ(host_pool->refCount(parent_source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    parent_result.leaf->group_slots[0].host_block = NULL_BLOCK_IDX;
    child_result.leaf->group_slots[0].host_block  = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, parent_source);
    group_->releaseSingleBlock(Tier::HOST, child_source);
}

TEST_F(BlockTreeEvictorTest, StaleSameTierSnapshotReadmitsAuthoritativeReplacement) {
    ASSERT_TRUE(torch::cuda::is_available()) << "C002-T01B requires CUDA";
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_same_tier_disk");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group_->setHostPool(host_pool);
    group_->setDiskPool(disk_pool);
    const size_t initial_device_free = device_pool_->freeBlocksNum();

    const BlockIdxType host_a = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_FALSE(isNullBlockIdx(host_a));
    auto result = insert({100}, {{makeSlot(Tier::HOST, host_a)}});
    ASSERT_NE(result.leaf, nullptr);
    auto&        slot    = result.leaf->group_slots[0];
    EvictionMove stale_a = BlockTreeEvictorTestPeer::makeMove(*evictor_, result.leaf, 0, Tier::HOST, Tier::DISK);

    ASSERT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    GroupBlockSet device_set = group_->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(device_set.per_node.size(), 1u);
    ASSERT_EQ(device_set.per_node[0].size(), 1u);
    const BlockIdxType device_block = device_set.per_node[0][0];
    group_->setBlocks(slot, Tier::DEVICE, device_set.per_node[0]);
    group_->unreferenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_a}}});
    group_->evictFromTier(result.leaf, slot, Tier::HOST);
    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, true);
    ASSERT_FALSE(host_pool->isAllocated(host_a));
    ASSERT_EQ(device_pool_->refCount(device_block), 1u);

    auto device_victim = evictor_->chooseVictim(0, Tier::DEVICE);
    ASSERT_TRUE(device_victim.has_value());
    auto device_to_host = evictor_->buildPlan(*device_victim);
    ASSERT_TRUE(device_to_host.has_value());
    ASSERT_TRUE(device_to_host->cascade_moves.empty());
    ASSERT_EQ(device_to_host->primary.target_blocks.size(), 1u);
    const BlockIdxType host_b = device_to_host->primary.target_blocks[0];
    ASSERT_NE(host_b, host_a);
    auto results = evictor_->performCopy(*device_to_host);
    ASSERT_TRUE(results.primary_success);
    ASSERT_TRUE(results.cascade_success.empty());
    evictor_->complete(*tree_, *device_to_host, results);
    ASSERT_FALSE(device_pool_->isAllocated(device_block));
    ASSERT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    ASSERT_EQ(slot.host_block, host_b);
    ASSERT_EQ(host_pool->refCount(host_b), 1u);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareMove(*evictor_, stale_a));
    EXPECT_TRUE(stale_a.target_blocks.empty());
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(slot.host_block, host_b);
    EXPECT_EQ(host_pool->refCount(host_b), 1u);
    EXPECT_FALSE(host_pool->isAllocated(host_a));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    auto replacement = evictor_->chooseVictim(0, Tier::HOST);
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(replacement->node, result.leaf);
    EXPECT_EQ(replacement->source_blocks, (std::vector<BlockIdxType>{host_b}));
    EXPECT_EQ(transfer_calls_, 1u);

    slot.host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, host_b);
    EXPECT_EQ(device_pool_->freeBlocksNum(), initial_device_free);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
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
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    group_->setHostPool(host_pool);
    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source, NULL_BLOCK_IDX);
    auto result = insert({100}, {{makeSlot(Tier::HOST, source)}});
    ASSERT_NE(result.leaf, nullptr);
    auto& slot = result.leaf->group_slots[0];

    ASSERT_TRUE(evictor_->beginLoadBack(result.leaf, 0, Tier::HOST));
    GroupBlockSet device_set = group_->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(device_set.per_node.size(), 1u);
    ASSERT_EQ(device_set.per_node.front().size(), 1u);
    group_->unreferenceBlocks(GroupBlockSet{0, Tier::HOST, {{source}}});
    group_->evictFromTier(result.leaf, slot, Tier::HOST);
    slot.device_blocks = device_set.per_node.front();
    evictor_->finishLoadBack(result.leaf, 0, Tier::HOST, true);

    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, result.leaf);

    slot.device_blocks = {NULL_BLOCK_IDX};
    group_->unreferenceBlocks(device_set);
}

TEST_F(BlockTreeEvictorTest, DemotionExcludesSourceAndRollbackOrSuccessRestoresOneTier) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    group_->setHostPool(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeSlot(Tier::DEVICE, source_block)}});
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
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{source_block}));
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

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsPinnedSiblingAndReadmitsAfterRelease) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_EQ(
        BlockTreeEvictorTestPeer::selectCascadeGroups(*environment.evictor_, environment.node_, 0, Tier::HOST, false),
        (std::vector<int>{1, 2}));

    GroupBlockSet pin = environment.hostSet(1);
    environment.groups_[1]->referenceBlocks(pin);
    ASSERT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 2u);

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupIds(*plan), (std::vector<int>{2}));
    EXPECT_EQ(plan->primary.component_group_id, 0);
    EXPECT_EQ(environment.node_->group_slots[0].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_slots[2].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.groups_[1]->unreferenceBlocks(pin);
    environment.evictor_->refreshCandidatesAfterRelease(pin);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupIds(*retry), (std::vector<int>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsLoadingBackSiblingAndReadmitsAfterFinish) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_TRUE(environment.evictor_->beginLoadBack(environment.node_, 1, Tier::HOST));

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupIds(*plan), (std::vector<int>{2}));
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::LOADING_BACK);
    EXPECT_EQ(environment.node_->group_slots[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.evictor_->finishLoadBack(environment.node_, 1, Tier::HOST, false);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::IDLE);

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupIds(*retry), (std::vector<int>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsDemotingSiblingWithoutAdoptingItsTarget) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    EvictionMove sibling_owner =
        BlockTreeEvictorTestPeer::makeMove(*environment.evictor_, environment.node_, 1, Tier::HOST, Tier::DISK);
    ASSERT_TRUE(BlockTreeEvictorTestPeer::prepareMove(*environment.evictor_, sibling_owner));
    ASSERT_EQ(sibling_owner.target_blocks.size(), 1u);
    const BlockIdxType owner_target = sibling_owner.target_blocks[0];

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupIds(*plan), (std::vector<int>{2}));
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_slots[1].host_block, environment.host_blocks_[1]);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    EXPECT_EQ(environment.disk_pools_[1]->refCount(owner_target), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    BlockTreeEvictorTestPeer::rollbackMove(*environment.evictor_, sibling_owner);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(owner_target));

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupIds(*retry), (std::vector<int>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, ReverseBuildPlanSkipsPinnedFullSiblingAndReadmitsIt) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init(/*enable_reverse_eviction=*/true));
    ASSERT_EQ(
        BlockTreeEvictorTestPeer::selectCascadeGroups(*environment.evictor_, environment.node_, 2, Tier::HOST, true),
        (std::vector<int>{0, 1}));

    GroupBlockSet pin = environment.hostSet(0);
    environment.groups_[0]->referenceBlocks(pin);
    auto plan = environment.buildPlan(2);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupIds(*plan), (std::vector<int>{1}));
    EXPECT_EQ(environment.node_->group_slots[2].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.groups_[0]->unreferenceBlocks(pin);
    environment.evictor_->refreshCandidatesAfterRelease(pin);
    auto retry = environment.buildPlan(2);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupIds(*retry), (std::vector<int>{0, 1}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, CascadeTargetExhaustionRestoresOnlyFailedSibling) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    const std::vector<BlockIdxType> exhausted = exhaustPool(*environment.disk_pools_[1]);
    ASSERT_EQ(exhausted.size(), 2u);
    ASSERT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 0u);
    const size_t exhausted_capacity = environment.disk_pools_[1]->freeBlocksNum();

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupIds(*plan), (std::vector<int>{2}));
    EXPECT_EQ(environment.node_->group_slots[0].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_slots[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_slots[2].transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_slots[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), exhausted_capacity);
    EXPECT_EQ(environment.host_pools_[1]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 1u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    for (size_t gid = 0; gid < environment.groups_.size(); ++gid) {
        EXPECT_EQ(environment.node_->group_slots[gid].transfer_state, SlotTransferState::IDLE);
        EXPECT_EQ(environment.host_pools_[gid]->refCount(environment.host_blocks_[gid]), 1u);
    }
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 0u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 2u);

    releaseBlocks(*environment.disk_pools_[1], exhausted);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrimaryCopyFailureSuppressesCascadesAndRollsBackFullPlan) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferStatuses({CopyStatus::DEVICE_IO_ERROR, CopyStatus::OK, CopyStatus::OK});

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(cascadeGroupIds(*plan), (std::vector<int>{1, 2}));
    auto results = environment.evictor_->performCopy(*plan);
    EXPECT_FALSE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false, false}));
    EXPECT_EQ(environment.transfer_group_ids_, (std::vector<int>{0}));

    environment.evictor_->complete(*environment.tree_, *plan, results);
    for (size_t gid = 0; gid < environment.groups_.size(); ++gid) {
        EXPECT_EQ(environment.node_->group_slots[gid].transfer_state, SlotTransferState::IDLE);
        EXPECT_EQ(environment.node_->group_slots[gid].host_block, environment.host_blocks_[gid]);
        EXPECT_EQ(environment.host_pools_[gid]->refCount(environment.host_blocks_[gid]), 1u);
        EXPECT_EQ(environment.disk_pools_[gid]->freeBlocksNum(), 2u);
        EXPECT_EQ(environment.host_pools_[gid]->activeTreeCachedBlocksNum(), 0u);
    }
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 3u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, CascadeCopyResultsPublishAndRollbackIndependently) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferStatuses({CopyStatus::OK, CopyStatus::DEVICE_IO_ERROR, CopyStatus::OK});

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(cascadeGroupIds(*plan), (std::vector<int>{1, 2}));
    const BlockIdxType primary_target = plan->primary.target_blocks[0];
    const BlockIdxType failed_target  = plan->cascade_moves[0].target_blocks[0];
    const BlockIdxType success_target = plan->cascade_moves[1].target_blocks[0];

    auto results = environment.evictor_->performCopy(*plan);
    ASSERT_TRUE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false, true}));
    EXPECT_EQ(environment.transfer_group_ids_, (std::vector<int>{0, 1, 2}));
    environment.evictor_->complete(*environment.tree_, *plan, results);

    const auto& primary_slot = environment.node_->group_slots[0];
    const auto& failed_slot  = environment.node_->group_slots[1];
    const auto& success_slot = environment.node_->group_slots[2];
    EXPECT_EQ(primary_slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(primary_slot.has_value(Tier::HOST));
    EXPECT_EQ(primary_slot.disk_slot, primary_target);
    EXPECT_EQ(environment.disk_pools_[0]->refCount(primary_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));

    EXPECT_EQ(failed_slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(failed_slot.host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(failed_target));
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.host_pools_[1]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);

    EXPECT_EQ(success_slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(success_slot.has_value(Tier::HOST));
    EXPECT_EQ(success_slot.disk_slot, success_target);
    EXPECT_EQ(environment.disk_pools_[2]->refCount(success_target), 1u);
    EXPECT_FALSE(environment.host_pools_[2]->isAllocated(environment.host_blocks_[2]));

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, DirectCompleteMissingCascadeResultRollsBack) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(cascadeGroupIds(*plan), (std::vector<int>{1, 2}));
    const BlockIdxType primary_target = plan->primary.target_blocks[0];
    const BlockIdxType first_target   = plan->cascade_moves[0].target_blocks[0];
    const BlockIdxType missing_target = plan->cascade_moves[1].target_blocks[0];

    BlockTreeEvictor::CopyResultSet synthetic_results;
    synthetic_results.primary_success = true;
    synthetic_results.cascade_success = {true};
    environment.evictor_->complete(*environment.tree_, *plan, synthetic_results);
    EXPECT_TRUE(environment.transfer_group_ids_.empty());

    const auto& primary_slot = environment.node_->group_slots[0];
    const auto& first_slot   = environment.node_->group_slots[1];
    const auto& missing_slot = environment.node_->group_slots[2];
    EXPECT_EQ(primary_slot.disk_slot, primary_target);
    EXPECT_EQ(first_slot.disk_slot, first_target);
    EXPECT_EQ(environment.disk_pools_[0]->refCount(primary_target), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->refCount(first_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));
    EXPECT_FALSE(environment.host_pools_[1]->isAllocated(environment.host_blocks_[1]));

    EXPECT_EQ(missing_slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(missing_slot.host_block, environment.host_blocks_[2]);
    EXPECT_EQ(environment.host_pools_[2]->refCount(environment.host_blocks_[2]), 1u);
    EXPECT_FALSE(environment.disk_pools_[2]->isAllocated(missing_target));
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.host_pools_[2]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorStatsTest, AggregatesCandidatesAcrossGroupsAndTiers) {
    auto group0      = makeFullGroup(0);
    auto group1      = makeFullGroup(1);
    auto device_pool = makeTestDevicePool(1, "block_tree_evictor_stats_device");
    auto host_pool   = makePageableHostPool(1);
    auto disk_pool   = makeTestDiskPool(1, "block_tree_evictor_stats_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    group0->setDevicePools({device_pool}, {"tag_0"});
    group0->setHostPool(host_pool);
    group1->setDiskPool(disk_pool);
    std::vector<ComponentGroupPtr> groups = {group0, group1};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(initEvictor(evictor));

    GroupBlockSet      device_set = group0->allocateBlocks(Tier::DEVICE, 1);
    const BlockIdxType host_block = group0->allocateSingleBlock(Tier::HOST);
    const BlockIdxType disk_block = group1->allocateSingleBlock(Tier::DISK);
    ASSERT_EQ(device_set.per_node.size(), 1u);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    BlockTree tree(2);
    auto      first = tree.insertNode(
        nullptr,
        {100},
        {{makeSlot(Tier::DEVICE, device_set.per_node.front().front()), makeSlot(Tier::DISK, disk_block)}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::HOST, host_block), GroupSlot{}}});
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

    first.leaf->group_slots[0].device_blocks = {NULL_BLOCK_IDX};
    first.leaf->group_slots[1].disk_slot     = NULL_BLOCK_IDX;
    second.leaf->group_slots[0].host_block   = NULL_BLOCK_IDX;
    group0->unreferenceBlocks(device_set);
    group0->releaseSingleBlock(Tier::HOST, host_block);
    group1->releaseSingleBlock(Tier::DISK, disk_block);
}

TEST(BlockTreeEvictorPolicyTest, MatchDoesNotChangeFifoAdmissionOrder) {
    auto group       = makeFullGroup(0);
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_fifo_policy");
    ASSERT_NE(device_pool, nullptr);
    group->setDevicePools({device_pool}, {"tag_0"});
    std::vector<ComponentGroupPtr> groups = {group};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(evictor.init({}, EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO));

    GroupBlockSet device_set = group->allocateBlocks(Tier::DEVICE, 2);
    ASSERT_EQ(device_set.per_node.size(), 2u);
    BlockTree tree(1);
    auto      first = tree.insertNode(nullptr, {100}, {{makeSlot(Tier::DEVICE, device_set.per_node[0][0])}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::DEVICE, device_set.per_node[1][0])}});
    evictor.onInsertCommitted(second);
    const uint64_t first_admission = first.leaf->group_slots[0].candidate_meta.admission_seq;

    evictor.onMatched({first.leaf});

    EXPECT_EQ(first.leaf->group_slots[0].candidate_meta.admission_seq, first_admission);
    auto victim = evictor.chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, first.leaf);

    first.leaf->group_slots[0].device_blocks  = {NULL_BLOCK_IDX};
    second.leaf->group_slots[0].device_blocks = {NULL_BLOCK_IDX};
    group->unreferenceBlocks(device_set);
}

TEST(BlockTreeEvictorPolicyTest, MatchUpdatesLfuHitCountAndOrder) {
    auto group       = makeFullGroup(0);
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_lfu_policy");
    ASSERT_NE(device_pool, nullptr);
    group->setDevicePools({device_pool}, {"tag_0"});
    std::vector<ComponentGroupPtr> groups = {group};
    BlockTreeEvictor               evictor(groups, BlockTreeEvictor::ExecuteTransferFn{}, false);
    ASSERT_TRUE(evictor.init({}, EvictionPolicy::LFU, EvictionPolicy::LRU, EvictionPolicy::FIFO));

    GroupBlockSet device_set = group->allocateBlocks(Tier::DEVICE, 2);
    ASSERT_EQ(device_set.per_node.size(), 2u);
    BlockTree tree(1);
    auto      first = tree.insertNode(nullptr, {100}, {{makeSlot(Tier::DEVICE, device_set.per_node[0][0])}});
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode(nullptr, {200}, {{makeSlot(Tier::DEVICE, device_set.per_node[1][0])}});
    evictor.onInsertCommitted(second);

    evictor.onMatched({first.leaf});

    EXPECT_EQ(first.leaf->group_slots[0].candidate_meta.hit_count, 1u);
    EXPECT_EQ(second.leaf->group_slots[0].candidate_meta.hit_count, 0u);
    auto victim = evictor.chooseVictim(Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, second.leaf);

    first.leaf->group_slots[0].device_blocks  = {NULL_BLOCK_IDX};
    second.leaf->group_slots[0].device_blocks = {NULL_BLOCK_IDX};
    group->unreferenceBlocks(device_set);
}

}  // namespace
}  // namespace rtp_llm
