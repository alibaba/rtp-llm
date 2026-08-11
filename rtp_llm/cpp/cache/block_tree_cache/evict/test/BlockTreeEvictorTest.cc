#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::allocateDeviceBlocksForTest;
using block_tree_cache_test::MultiNodeBlocks;
using block_tree_cache_test::releaseLowerTierSeedRefs;
using block_tree_cache_test::unreferenceDeviceBlocksForTest;

std::shared_ptr<FullGroupSet> makeFullGroup(const DeviceBlockPoolPtr& device_pool) {
    return std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, nullptr, nullptr);
}

TreeNode* insertedNode(const BlockTreeInsertResult& result) {
    return result.inserted_nodes.back();
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

void initializeGroups(const std::vector<GroupSetPtr>&        groups,
                      const std::vector<DeviceBlockPoolPtr>& device_pools,
                      std::vector<GroupBase>                 group_bases) {
    RTP_LLM_CHECK(groups.size() == device_pools.size());
    RTP_LLM_CHECK(groups.size() == group_bases.size());
    auto topology = block_transfer_engine_test::makeTestTopology(std::move(group_bases));
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        groups[group_set_id]->initialize(group_set_id, topology, {group_set_id});
    }
}

void initializeFullGroup(const GroupSetPtr& group, const DeviceBlockPoolPtr& device_pool) {
    initializeGroups(
        {group},
        {device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16)});
}

std::vector<GroupSetPtr> makeCascadeGroups() {
    auto full_device_pool   = makeTestDevicePool(2, "cascade_policy_full");
    auto swa_device_pool    = makeTestDevicePool(2, "cascade_policy_swa");
    auto linear_device_pool = makeTestDevicePool(2, "cascade_policy_linear");
    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, nullptr, nullptr);
    auto swa  = std::make_shared<SWAGroupSet>(2, 1, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, nullptr, nullptr);
    auto linear =
        std::make_shared<LinearGroupSet>(std::vector<DeviceBlockPoolPtr>{linear_device_pool}, nullptr, nullptr);
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    full_policy.enable_prefix_reuse   = true;
    swa_policy.enable_prefix_reuse    = true;
    linear_policy.enable_prefix_reuse = true;
    swa_policy.sliding_window_size    = 2;
    std::vector<GroupSetPtr> groups   = {full, swa, linear};
    initializeGroups(groups,
                     {full_device_pool, swa_device_pool, linear_device_pool},
                     {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16),
                      block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16),
                      block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16)});
    return groups;
}

GroupSetResource makeResource(Tier tier, BlockIdxType block) {
    GroupSetResource resource;
    switch (tier) {
        case Tier::DEVICE:
            resource.device_blocks = {block};
            break;
        case Tier::HOST:
            resource.host_block = block;
            break;
        case Tier::DISK:
            resource.disk_slot = block;
            break;
        default:
            break;
    }
    return resource;
}

void initEvictor(BlockTreeEvictor& evictor) {
    evictor.init(EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO);
}

class TestEvictorRuntime {
public:
    std::unique_ptr<BlockTreeEvictor> make(BlockTree* tree, BlockTreeEvictor::ExecuteTransferFn execute_transfer) {
        return std::make_unique<BlockTreeEvictor>(
            tree,
            std::move(execute_transfer),
            nullptr,
            nullptr,
            metrics_reporter_,
            mutex_,
            0,
            0,
            [](Tier) { return true; },
            [](bool, bool) {},
            [](CacheKeyType, size_t) {});
    }

private:
    BlockTreeCacheMetricsReporter metrics_reporter_;
    std::mutex                    mutex_;
};

class BlockTreeEvictorTestPeer {
public:
    static TransferDescriptor
    makeDesc(BlockTreeEvictor& evictor, TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) {
        return evictor.makeDesc(node, group_set_id, source_tier, target_tier);
    }

    static bool prepareDesc(BlockTreeEvictor& evictor, TransferDescriptor& eviction_desc) {
        return evictor.prepareDesc(eviction_desc);
    }

    static std::vector<size_t>
    selectCascadeGroupSets(BlockTreeEvictor& evictor, TreeNode* node, size_t group_set_id, Tier tier) {
        return evictor.selectCascadeGroupSets(node, group_set_id, tier);
    }

    static void rollbackDesc(BlockTreeEvictor& evictor, const TransferDescriptor& eviction_desc) {
        BlockTreeEvictor::EvictionPlan plan;
        plan.primary_desc = eviction_desc;
        evictor.rollbackPreparedPlan(plan);
    }
};

std::vector<size_t> cascadeGroupSetIds(const BlockTreeEvictor::EvictionPlan& plan) {
    std::vector<size_t> result;
    result.reserve(plan.cascade_descs.size());
    for (const TransferDescriptor& cascade_desc : plan.cascade_descs) {
        result.push_back(cascade_desc.group_set_id);
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
        pool.incRef(*block, BlockRefType::REQUEST);
        blocks.push_back(*block);
    }
    return blocks;
}

void releaseBlocks(IBlockPool& pool, const std::vector<BlockIdxType>& blocks) {
    for (BlockIdxType block : blocks) {
        pool.decRef(block, BlockRefType::REQUEST);
    }
}

class CascadeTestEnvironment {
public:
    bool init() {
        std::vector<DeviceBlockPoolPtr> device_pools = {
            makeTestDevicePool(2, "cascade_environment_full"),
            makeTestDevicePool(2, "cascade_environment_swa"),
            makeTestDevicePool(2, "cascade_environment_linear"),
        };
        for (size_t group_set_id = 0; group_set_id < device_pools.size(); ++group_set_id) {
            auto host = makePageableHostPool(2);
            auto disk = makeTestDiskPool(2, "block_tree_evictor_cascade_" + std::to_string(group_set_id));
            if (host == nullptr || disk == nullptr) {
                return false;
            }
            host_pools_.push_back(std::move(host));
            disk_pools_.push_back(std::move(disk));
        }

        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pools[0]}, host_pools_[0], disk_pools_[0]);
        auto swa = std::make_shared<SWAGroupSet>(
            2, 1, std::vector<DeviceBlockPoolPtr>{device_pools[1]}, host_pools_[1], disk_pools_[1]);
        auto linear = std::make_shared<LinearGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pools[2]}, host_pools_[2], disk_pools_[2]);
        groups_                           = {full, swa, linear};
        auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
        auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
        auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        full_policy.enable_prefix_reuse   = true;
        swa_policy.enable_prefix_reuse    = true;
        linear_policy.enable_prefix_reuse = true;
        swa_policy.sliding_window_size    = 2;
        initializeGroups(groups_,
                         device_pools,
                         {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16),
                          block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16),
                          block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16)});

        tree_ = std::make_unique<BlockTree>(groups_);

        evictor_ = evictor_runtime_.make(
            tree_.get(),
            [this](const TransferDescriptor& descriptor) {
                transfer_group_set_ids_.push_back(descriptor.group_set_id);
                if (transfer_results_.empty()) {
                    return true;
                }
                const bool success = transfer_results_.front();
                transfer_results_.pop_front();
                return success;
            });
        initEvictor(*evictor_);

        std::vector<GroupSetResource> resources(groups_.size());
        host_blocks_.resize(groups_.size(), NULL_BLOCK_IDX);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            host_blocks_[group_set_id] =
                groups_[group_set_id]->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
            if (isNullBlockIdx(host_blocks_[group_set_id])) {
                return false;
            }
            resources[group_set_id].host_block = host_blocks_[group_set_id];
        }

        auto result = tree_->insertNode({100}, {resources}, /*collect_path=*/false);
        releaseLowerTierSeedRefs(groups_, {resources});
        evictor_->onInsertCommitted(result);
        node_ = insertedNode(result);
        return node_ != nullptr;
    }

    std::optional<BlockTreeEvictor::EvictionPlan> buildPlan(size_t primary_group_set_id) {
        auto victim = evictor_->chooseVictim(primary_group_set_id, Tier::HOST);
        if (!victim.has_value()) {
            return std::nullopt;
        }
        return evictor_->buildPlan(*victim);
    }

    MultiNodeResource hostSet(size_t group_set_id) const {
        return MultiNodeResource{
            group_set_id, Tier::HOST, {{node_, {host_blocks_[static_cast<size_t>(group_set_id)]}}}};
    }

    void setTransferResults(std::initializer_list<bool> results) {
        transfer_results_.assign(results.begin(), results.end());
        transfer_group_set_ids_.clear();
    }

    void releaseResidentBlocks() {
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            auto& resource = node_->group_set_resources[group_set_id];
            if (resource.hasTier(Tier::HOST)) {
                const BlockIdxType block = resource.host_block;
                resource.host_block      = NULL_BLOCK_IDX;
                groups_[group_set_id]->releaseSingleBlock(Tier::HOST, block, BlockRefType::BLOCK_CACHE);
            }
            if (resource.hasTier(Tier::DISK)) {
                const BlockIdxType block = resource.disk_slot;
                resource.disk_slot       = NULL_BLOCK_IDX;
                groups_[group_set_id]->releaseSingleBlock(Tier::DISK, block, BlockRefType::BLOCK_CACHE);
            }
        }
    }

    void expectAllPoolsFree() const {
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            EXPECT_EQ(host_pools_[group_set_id]->freeBlocksNum(), 2u);
            EXPECT_EQ(disk_pools_[group_set_id]->freeBlocksNum(), 2u);
        }
    }

    std::vector<GroupSetPtr>                             groups_;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools_;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools_;
    std::vector<BlockIdxType>                            host_blocks_;
    std::unique_ptr<BlockTree>                           tree_;
    TestEvictorRuntime                                   evictor_runtime_;
    std::unique_ptr<BlockTreeEvictor>                    evictor_;
    TreeNode*                                            node_{nullptr};
    std::deque<bool>                                     transfer_results_;
    std::vector<size_t>                                  transfer_group_set_ids_;
};

class BlockTreeEvictorTest: public ::testing::Test {
protected:
    void SetUp() override {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(test_info, nullptr);
        device_pool_ = makeTestDevicePool(128, "block_tree_evictor_fixture_" + std::string(test_info->name()));
        ASSERT_NE(device_pool_, nullptr);
        resetGroup();
    }

    void resetGroup(std::shared_ptr<HostBlockPool> host_pool = nullptr, BlockTreeDiskBlockPoolPtr disk_pool = nullptr) {
        group_ = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pool_}, std::move(host_pool), std::move(disk_pool));
        initializeFullGroup(group_, device_pool_);
        groups_  = {group_};
        tree_    = std::make_unique<BlockTree>(groups_);
        evictor_ = evictor_runtime_.make(
            tree_.get(),
            [this](const TransferDescriptor&) {
                ++transfer_calls_;
                return true;
            });
        initEvictor(*evictor_);
    }

    BlockTreeInsertResult insert(const CacheKeysType&                              keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources) {
        auto result = tree_->insertNode(keys, resources, /*collect_path=*/false);
        releaseLowerTierSeedRefs(groups_, resources);
        evictor_->onInsertCommitted(result);
        return result;
    }

    bool reserveAndBeginLoad(TreeNode* node, size_t group_set_id, Tier source) {
        GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_state != GroupSetTransferState::IDLE || resource.getTopTier() != source) {
            return false;
        }
        resource.transfer_state = GroupSetTransferState::LOAD_PENDING;
        evictor_->refreshCandidate(node, group_set_id);
        resource.transfer_state = GroupSetTransferState::LOADING;
        evictor_->refreshCandidate(node, group_set_id);
        return true;
    }

    void settleLoad(TreeNode* node, size_t group_set_id, bool copy_ok) {
        node->group_set_resources[group_set_id].transfer_state = GroupSetTransferState::IDLE;
        if (copy_ok) {
            evictor_->onTierEntered(node, group_set_id, Tier::DEVICE);
        } else {
            evictor_->refreshCandidate(node, group_set_id);
        }
    }

    std::shared_ptr<FullGroupSet>     group_;
    DeviceBlockPoolPtr                device_pool_;
    std::vector<GroupSetPtr>          groups_;
    std::unique_ptr<BlockTree>        tree_;
    TestEvictorRuntime                evictor_runtime_;
    std::unique_ptr<BlockTreeEvictor> evictor_;
    size_t                            transfer_calls_{0};
};

TEST_F(BlockTreeEvictorTest, PendingReleasesFollowAsyncPlanSourcePools) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(2, "pending_release_plan_disk");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    auto verify_tier = [this](Tier source_tier, Tier target_tier, IBlockPool* pool, BlockIdxType block) {
        BlockTreeEvictor::EvictionPlan plan;
        plan.primary_desc.group_set_id  = 0;
        plan.primary_desc.source_tier   = source_tier;
        plan.primary_desc.target_tier   = target_tier;
        plan.primary_desc.source_blocks = {block};

        evictor_->reservePendingReleases(plan);
        ASSERT_EQ(evictor_->pending_release_counts_.at(pool), 1u);
        evictor_->settlePendingReleases(plan);
        EXPECT_EQ(evictor_->pending_release_counts_.at(pool), 0u);
    };

    verify_tier(Tier::DEVICE, Tier::HOST, device_pool_.get(), 7);
    verify_tier(Tier::HOST, Tier::DISK, host_pool.get(), 8);
}

TEST_F(BlockTreeEvictorTest, PendingReleasesCountEveryDeviceMemberBlock) {
    auto policy   = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(policy, {0}, 16),
         block_transfer_engine_test::makeTestGroupBase(policy, {1}, 16)});
    group_ =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool_, device_pool_}, nullptr, nullptr);
    group_->initialize(0, std::move(topology), {0, 1});
    groups_  = {group_};
    tree_    = std::make_unique<BlockTree>(groups_);
    evictor_ = evictor_runtime_.make(tree_.get(), BlockTreeEvictor::ExecuteTransferFn{});
    initEvictor(*evictor_);

    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc.group_set_id  = 0;
    plan.primary_desc.source_tier   = Tier::DEVICE;
    plan.primary_desc.target_tier   = Tier::HOST;
    plan.primary_desc.source_blocks = {7, 8};

    evictor_->reservePendingReleases(plan);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 2u);
    evictor_->settlePendingReleases(plan);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
}

TEST_F(BlockTreeEvictorTest, SettlePendingReleasesReportsPoolAndBlock) {
    constexpr BlockIdxType block = 17;
    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc.group_set_id  = 0;
    plan.primary_desc.source_tier   = Tier::DEVICE;
    plan.primary_desc.target_tier   = Tier::HOST;
    plan.primary_desc.source_blocks = {block};

    try {
        evictor_->settlePendingReleases(plan);
        FAIL() << "settling an unreserved pending release should fail";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("pool=" + device_pool_->poolName()), std::string::npos);
        EXPECT_NE(message.find("block=17"), std::string::npos);
        EXPECT_NE(message.find("pending=0"), std::string::npos);
    }
}

TEST_F(BlockTreeEvictorTest, PoolWatermarkExcessRejectsPendingReleasesAboveUsedBlocks) {
    ASSERT_EQ(device_pool_->usedBlocksNum(), 0u);
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_[device_pool_.get()] = 1;
    }

    std::string error_message;
    try {
        (void)evictor_->poolWatermarkExcess(device_pool_.get(), 0.5);
    } catch (const std::runtime_error& error) {
        error_message = error.what();
    }
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_.clear();
    }

    ASSERT_FALSE(error_message.empty()) << "pending releases above used blocks should fail";
    EXPECT_NE(error_message.find("pool=" + device_pool_->poolName()), std::string::npos);
    EXPECT_NE(error_message.find("pending=1"), std::string::npos);
    EXPECT_NE(error_message.find("used=0"), std::string::npos);
}

TEST_F(BlockTreeEvictorTest, ComputeGroupSetExcessRejectsNonPositiveRatio) {
    for (double ratio : {0.0, -0.1}) {
        try {
            (void)evictor_->computeGroupSetExcess(*group_, Tier::DEVICE, ratio);
            FAIL() << "non-positive watermark ratio should fail: " << ratio;
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("group_set=0"), std::string::npos);
            EXPECT_NE(message.find("tier=DEVICE"), std::string::npos);
            EXPECT_NE(message.find("ratio="), std::string::npos);
        }
    }
}

TEST(BlockTreeEvictorCascadeTest, NonLeafCascadeFollowsGroupPriority) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&  evictor        = *evictor_holder;

    const std::vector<GroupSetResource> resources = {
        makeResource(Tier::HOST, 1), makeResource(Tier::HOST, 2), makeResource(Tier::HOST, 3)};
    auto path = tree.insertNode({100, 200}, {resources, resources}, /*collect_path=*/false);
    ASSERT_EQ(path.inserted_nodes.size(), 2u);
    TreeNode* non_leaf = path.inserted_nodes.front();
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                  evictor, non_leaf, /*source_group_set_id=*/0, Tier::HOST),
              (std::vector<size_t>{1, 2}));
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                  evictor, non_leaf, /*source_group_set_id=*/1, Tier::HOST),
              (std::vector<size_t>{2}));
    EXPECT_TRUE(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                    evictor, non_leaf, /*source_group_set_id=*/2, Tier::HOST)
                    .empty());
}

TEST(BlockTreeEvictorCascadeTest, ReverseCascadeIsAlwaysEnabledAtLeaf) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&  evictor        = *evictor_holder;

    auto inserted =
        tree.insertNode({100},
                        {{makeResource(Tier::HOST, 1), makeResource(Tier::HOST, 2), makeResource(Tier::HOST, 3)}},
                        /*collect_path=*/false);
    ASSERT_NE(insertedNode(inserted), nullptr);
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                  evictor, insertedNode(inserted), /*source_group_set_id=*/0, Tier::HOST),
              (std::vector<size_t>{1, 2}));
    EXPECT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                  evictor, insertedNode(inserted), /*source_group_set_id=*/1, Tier::HOST),
              (std::vector<size_t>{0, 2}));
}

TEST_F(BlockTreeEvictorTest, TierEntryRefreshesLastAccessTime) {
    const std::optional<BlockIdList> allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    BlockTreeInsertResult result = insert({100}, {{makeResource(Tier::DEVICE, allocated->front())}});
    TreeNode*             node   = insertedNode(result);
    ASSERT_NE(node, nullptr);

    CandidateMeta& candidate_meta = node->group_set_resources[0].candidate_meta;
    candidate_meta.tier_enter_time_us  = 0;
    candidate_meta.last_access_time_us = 0;
    evictor_->onTierEntered(node, 0, Tier::DEVICE);

    EXPECT_GT(candidate_meta.tier_enter_time_us, 0);
    EXPECT_EQ(candidate_meta.last_access_time_us, candidate_meta.tier_enter_time_us);
}

TEST_F(BlockTreeEvictorTest, MatchUpdatesIntermediateHistoryWithoutAdmittingIt) {
    const auto allocated = device_pool_->malloc(3);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 3u);
    const BlockIdxType                         parent_block = (*allocated)[0];
    const BlockIdxType                         leaf_block   = (*allocated)[1];
    const BlockIdxType                         rival_block  = (*allocated)[2];
    std::vector<std::vector<GroupSetResource>> resources    = {{makeResource(Tier::DEVICE, parent_block)},
                                                               {makeResource(Tier::DEVICE, leaf_block)}};
    auto                                       result       = insert({100, 200}, resources);
    ASSERT_EQ(result.inserted_nodes.size(), 2u);
    auto rival = insert({300}, {{makeResource(Tier::DEVICE, rival_block)}});
    ASSERT_NE(insertedNode(rival), nullptr);

    TreeNode* parent = result.inserted_nodes[0];
    TreeNode* leaf   = result.inserted_nodes[1];
    ASSERT_NE(parent, nullptr);
    ASSERT_NE(leaf, nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 2u);

    const int64_t parent_insert_time_us = parent->group_set_resources[0].candidate_meta.insert_time_us;
    evictor_->onMatched({parent, leaf});

    const auto parent_meta = parent->group_set_resources[0].candidate_meta;
    const auto leaf_meta   = leaf->group_set_resources[0].candidate_meta;
    EXPECT_EQ(parent_meta.last_access_seq, leaf_meta.last_access_seq);
    EXPECT_EQ(parent_meta.insert_time_us, parent_insert_time_us);
    EXPECT_EQ(parent_meta.last_access_time_us, leaf_meta.last_access_time_us);
    EXPECT_GE(parent_meta.last_access_time_us, parent_insert_time_us);
    EXPECT_EQ(parent_meta.hit_count, 1u);
    EXPECT_EQ(leaf_meta.hit_count, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 2u);

    const MultiNodeResource leaf_resource{0, Tier::DEVICE, {{leaf, {leaf_block}}}};
    group_->unmapDeviceBlocksFromTreeNode(leaf_resource);
    leaf->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(leaf_resource, BlockRefType::BLOCK_CACHE);
    evictor_->refreshCandidate(leaf, 0);
    tree_->removeNodeAndEmptyAncestors(leaf);
    evictor_->onTopologyChanged(parent);

    ASSERT_EQ(evictor_->candidateStats().device_candidates, 2u);
    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(rival));
    EXPECT_EQ(parent->group_set_resources[0].candidate_meta.last_access_seq, parent_meta.last_access_seq);
    EXPECT_EQ(parent->group_set_resources[0].candidate_meta.hit_count, parent_meta.hit_count);

    const MultiNodeResource parent_resource{0, Tier::DEVICE, {{parent, {parent_block}}}};
    group_->unmapDeviceBlocksFromTreeNode(parent_resource);
    parent->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(parent_resource, BlockRefType::BLOCK_CACHE);
    evictor_->refreshCandidate(parent, 0);
    const MultiNodeResource rival_resource{0, Tier::DEVICE, {{insertedNode(rival), {rival_block}}}};
    group_->unmapDeviceBlocksFromTreeNode(rival_resource);
    insertedNode(rival)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(rival_resource, BlockRefType::BLOCK_CACHE);
    evictor_->refreshCandidate(insertedNode(rival), 0);
}

TEST_F(BlockTreeEvictorTest, ExistingGroupFillAdmitsChildAndRemovesFullParentCandidate) {
    const auto allocated = device_pool_->malloc(2);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    const BlockIdxType parent_block = (*allocated)[0];
    const BlockIdxType child_block  = (*allocated)[1];

    BlockTreeInsertResult parent_result = insert({100}, {{makeResource(Tier::DEVICE, parent_block)}});
    ASSERT_NE(insertedNode(parent_result), nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    GroupSetResource empty_resource;
    empty_resource.device_blocks      = {NULL_BLOCK_IDX};
    BlockTreeInsertResult empty_child = tree_->insertNode(
        {100, 200}, {{makeResource(Tier::DEVICE, parent_block)}, {empty_resource}}, /*collect_path=*/false);
    evictor_->onInsertCommitted(empty_child);
    ASSERT_NE(insertedNode(empty_child), nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    const BlockTreeInsertResult fill_result =
        insert({100, 200}, {{makeResource(Tier::DEVICE, parent_block)}, {makeResource(Tier::DEVICE, child_block)}});
    ASSERT_TRUE(fill_result.inserted_nodes.empty());
    ASSERT_EQ(fill_result.adopted_nodes.size(), 1u);
    EXPECT_EQ(fill_result.adopted_nodes.front().first, insertedNode(empty_child));

    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    const std::optional<TransferDescriptor> victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(empty_child));
    EXPECT_EQ(victim->source_blocks, (std::vector<BlockIdxType>{child_block}));
}

TEST_F(BlockTreeEvictorTest, LastReferenceReleaseReadmitsLazyDroppedCandidate) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const BlockIdxType block = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(block));
    ASSERT_EQ(host_pool->refCount(block), 1u);

    auto result = insert({100}, {{makeResource(Tier::HOST, block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    MultiNodeResource match_set{0, Tier::HOST, {{insertedNode(result), {block}}}};
    group_->referenceBlocks(match_set, BlockRefType::REQUEST);
    group_->referenceBlocks(match_set, BlockRefType::REQUEST);
    ASSERT_EQ(host_pool->refCount(block), 3u);

    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST).has_value());
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    group_->unreferenceBlocks(match_set, BlockRefType::REQUEST);
    evictor_->refreshCandidatesAfterRelease(match_set);
    EXPECT_EQ(host_pool->refCount(block), 2u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    group_->unreferenceBlocks(match_set, BlockRefType::REQUEST);
    evictor_->refreshCandidatesAfterRelease(match_set);
    EXPECT_EQ(host_pool->refCount(block), 1u);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, block, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsNewRequestPinWithoutAllocatingTarget) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_pin");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);
    MultiNodeResource pin{0, Tier::HOST, {{insertedNode(result), {source}}}};
    group_->referenceBlocks(pin, BlockRefType::REQUEST);
    ASSERT_EQ(host_pool->refCount(source), 2u);

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_EQ(host_pool->refCount(source), 2u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(transfer_calls_, 0u);

    group_->unreferenceBlocks(pin, BlockRefType::REQUEST);
    evictor_->refreshCandidatesAfterRelease(pin);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockRefType::BLOCK_CACHE);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsSourceBlockIdentityMismatch) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_prepare_identity");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType other  = group_->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_FALSE(isNullBlockIdx(source));
    ASSERT_FALSE(isNullBlockIdx(other));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);
    stale.source_blocks = {other};
    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockRefType::BLOCK_CACHE);
    group_->releaseSingleBlock(Tier::HOST, other, BlockRefType::REQUEST);
}

TEST_F(BlockTreeEvictorTest, PrepareMovePreservesLoadOwner) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_load");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);

    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    ASSERT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    settleLoad(insertedNode(result), 0, false);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, PrepareMovePreservesExistingDemotionOwnerAndTarget) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_demotion");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);
    TransferDescriptor owner =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);
    ASSERT_TRUE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, owner));
    ASSERT_EQ(owner.target_blocks.size(), 1u);
    const BlockIdxType owner_target = owner.target_blocks[0];
    ASSERT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_TRUE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->refCount(owner_target), 1u);
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockRefType::EVICTION), 1u);
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 0u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(transfer_calls_, 0u);

    BlockTreeEvictorTestPeer::rollbackDesc(*evictor_, owner);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(host_pool->refCount(source), 1u);
    EXPECT_FALSE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockRefType::EVICTION), 0u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsSourceTierChangedByLoad) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_tier_change");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(result), 0, Tier::HOST, Tier::DISK);

    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    auto&           resource   = insertedNode(result)->group_set_resources[0];
    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group_, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_EQ(device_set.front().size(), 1u);
    const BlockIdxType device_block = device_set.front().front();
    resource.setBlocks(Tier::DEVICE, device_set.front());
    const MultiNodeResource device_resource{0, Tier::DEVICE, {{insertedNode(result), device_set.front()}}};
    group_->mapDeviceBlocksToTreeNode(device_resource);
    group_->unreferenceBlocks(MultiNodeResource{0, Tier::HOST, {{insertedNode(result), {source}}}},
                              BlockRefType::BLOCK_CACHE);
    resource.evictFromTier(Tier::HOST);
    settleLoad(insertedNode(result), 0, true);
    ASSERT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    ASSERT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    ASSERT_FALSE(host_pool->isAllocated(source));

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    group_->unmapDeviceBlocksFromTreeNode(device_resource);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, device_set, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, PrepareMoveRejectsFullNodeThatBecameNonLeaf) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_topology");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType parent_source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType child_source  = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_source));
    ASSERT_FALSE(isNullBlockIdx(child_source));
    auto parent_result = insert({100}, {{makeResource(Tier::HOST, parent_source)}});
    ASSERT_NE(insertedNode(parent_result), nullptr);
    TransferDescriptor stale =
        BlockTreeEvictorTestPeer::makeDesc(*evictor_, insertedNode(parent_result), 0, Tier::HOST, Tier::DISK);

    const std::vector<std::vector<GroupSetResource>> child_resources = {{makeResource(Tier::HOST, parent_source)},
                                                                        {makeResource(Tier::HOST, child_source)}};
    // The duplicate entry reuses a tree-owned block, so it needs its own seed hold
    // for the release below to stay balanced.
    group_->referenceBlocks(MultiNodeResource{0, Tier::HOST, {{nullptr, {parent_source}}}}, BlockRefType::BLOCK_CACHE);
    auto child_result = tree_->insertNode({100, 101}, child_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups_, child_resources);
    evictor_->onInsertCommitted(child_result);
    ASSERT_NE(insertedNode(child_result), nullptr);
    ASSERT_FALSE(tree_->isLeafAtTier(insertedNode(parent_result), 0, Tier::HOST));

    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(*evictor_, stale));
    EXPECT_TRUE(stale.target_blocks.empty());
    EXPECT_EQ(insertedNode(parent_result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(parent_result)->group_set_resources[0].host_block, parent_source);
    EXPECT_EQ(host_pool->refCount(parent_source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(transfer_calls_, 0u);

    insertedNode(parent_result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    insertedNode(child_result)->group_set_resources[0].host_block  = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, parent_source, BlockRefType::BLOCK_CACHE);
    group_->releaseSingleBlock(Tier::HOST, child_source, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, LoadingStateExcludesAndIdleStateReadmitsSource) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    EXPECT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    settleLoad(insertedNode(result), 0, false);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
}

TEST_F(BlockTreeEvictorTest, LoadSuccessAdmitsOnlyStableDeviceResource) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(source, NULL_BLOCK_IDX);
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];

    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group_, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_EQ(device_set.front().size(), 1u);
    group_->unreferenceBlocks(MultiNodeResource{0, Tier::HOST, {{insertedNode(result), {source}}}},
                              BlockRefType::BLOCK_CACHE);
    resource.evictFromTier(Tier::HOST);
    resource.setBlocks(Tier::DEVICE, device_set.front());
    const MultiNodeResource device_resource{0, Tier::DEVICE, {{insertedNode(result), device_set.front()}}};
    group_->mapDeviceBlocksToTreeNode(device_resource);
    settleLoad(insertedNode(result), 0, true);

    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));

    group_->unmapDeviceBlocksFromTreeNode(device_resource);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, device_set, BlockRefType::BLOCK_CACHE);
}

TEST_F(BlockTreeEvictorTest, DemotionExcludesSourceAndRollbackOrSuccessRestoresOneTier) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    const CandidateMeta candidate_meta = resource.candidate_meta;
    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto plan = evictor_->buildPlan(*victim);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(plan->primary_desc.target_blocks.size(), 1u);
    EXPECT_EQ(plan->primary_timing.tier_enter_time_us, candidate_meta.tier_enter_time_us);
    EXPECT_EQ(plan->primary_timing.insert_time_us, candidate_meta.insert_time_us);
    EXPECT_EQ(plan->primary_timing.last_access_time_us, candidate_meta.last_access_time_us);
    EXPECT_GE(plan->primary_timing.selected_time_us, candidate_meta.last_access_time_us);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::EVICTION), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 0u);

    evictor_->rollbackPreparedPlan(*plan);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source_block}));
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::EVICTION), 0u);

    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    plan = evictor_->buildPlan(*victim);
    ASSERT_TRUE(plan.has_value());
    const BlockIdxType target_block = plan->primary_desc.target_blocks[0];
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::EVICTION), 1u);
    evictor_->complete(*plan, BlockTreeEvictor::CopyResultSet{true, {}});

    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_EQ(resource.host_block, target_block);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(host_pool->refCount(target_block), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::EVICTION), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 1u);

    resource.host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, target_block, BlockRefType::BLOCK_CACHE);
}

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsPinnedSiblingAndReadmitsAfterRelease) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_EQ(BlockTreeEvictorTestPeer::selectCascadeGroupSets(
                  *environment.evictor_, environment.node_, 0, Tier::HOST),
              (std::vector<size_t>{1, 2}));

    MultiNodeResource pin = environment.hostSet(1);
    environment.groups_[1]->referenceBlocks(pin, BlockRefType::REQUEST);
    ASSERT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 2u);

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{2}));
    EXPECT_EQ(plan->primary_desc.group_set_id, 0);
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.groups_[1]->unreferenceBlocks(pin, BlockRefType::REQUEST);
    environment.evictor_->refreshCandidatesAfterRelease(pin);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsLoadingSiblingAndReadmitsAfterFinish) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.node_->group_set_resources[1].transfer_state = GroupSetTransferState::LOADING;
    environment.evictor_->refreshCandidate(environment.node_, 1);

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.node_->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    environment.evictor_->refreshCandidate(environment.node_, 1);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, BuildPlanSkipsDemotingSiblingWithoutAdoptingItsTarget) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    TransferDescriptor sibling_owner =
        BlockTreeEvictorTestPeer::makeDesc(*environment.evictor_, environment.node_, 1, Tier::HOST, Tier::DISK);
    ASSERT_TRUE(BlockTreeEvictorTestPeer::prepareDesc(*environment.evictor_, sibling_owner));
    ASSERT_EQ(sibling_owner.target_blocks.size(), 1u);
    const BlockIdxType owner_target = sibling_owner.target_blocks[0];

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    EXPECT_EQ(environment.disk_pools_[1]->refCount(owner_target), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    BlockTreeEvictorTestPeer::rollbackDesc(*environment.evictor_, sibling_owner);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(owner_target));

    auto retry = environment.buildPlan(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->rollbackPreparedPlan(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, LeafBuildPlanSkipsPinnedFullSiblingAndReadmitsIt) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_EQ(
        BlockTreeEvictorTestPeer::selectCascadeGroupSets(*environment.evictor_, environment.node_, 2, Tier::HOST),
        (std::vector<size_t>{0, 1}));

    MultiNodeResource pin = environment.hostSet(0);
    environment.groups_[0]->referenceBlocks(pin, BlockRefType::REQUEST);
    auto plan = environment.buildPlan(2);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{1}));
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    environment.groups_[0]->unreferenceBlocks(pin, BlockRefType::REQUEST);
    environment.evictor_->refreshCandidatesAfterRelease(pin);
    auto retry = environment.buildPlan(2);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{0, 1}));
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
    EXPECT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), exhausted_capacity);
    EXPECT_EQ(environment.host_pools_[1]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 1u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    environment.evictor_->rollbackPreparedPlan(*plan);
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(environment.host_pools_[group_set_id]->refCount(environment.host_blocks_[group_set_id]), 1u);
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
    environment.setTransferResults({false, true, true});

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{1, 2}));
    auto results = environment.evictor_->taskRunner().performCopy(*plan);
    EXPECT_FALSE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false, false}));
    EXPECT_EQ(environment.transfer_group_set_ids_, (std::vector<size_t>{0}));

    environment.evictor_->complete(*plan, results);
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].host_block,
                  environment.host_blocks_[group_set_id]);
        EXPECT_EQ(environment.host_pools_[group_set_id]->refCount(environment.host_blocks_[group_set_id]), 1u);
        EXPECT_EQ(environment.disk_pools_[group_set_id]->freeBlocksNum(), 2u);
        EXPECT_EQ(environment.host_pools_[group_set_id]->activeTreeCachedBlocksNum(), 0u);
    }
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 3u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, CascadeCopyResultsPublishAndRollbackIndependently) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferResults({true, false, true});

    auto plan = environment.buildPlan(0);
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{1, 2}));
    const BlockIdxType primary_target = plan->primary_desc.target_blocks[0];
    const BlockIdxType failed_target  = plan->cascade_descs[0].target_blocks[0];
    const BlockIdxType success_target = plan->cascade_descs[1].target_blocks[0];

    auto results = environment.evictor_->taskRunner().performCopy(*plan);
    ASSERT_TRUE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false, true}));
    EXPECT_EQ(environment.transfer_group_set_ids_, (std::vector<size_t>{0, 1, 2}));
    environment.evictor_->complete(*plan, results);

    const auto& primary_resource = environment.node_->group_set_resources[0];
    const auto& failed_resource  = environment.node_->group_set_resources[1];
    const auto& success_resource = environment.node_->group_set_resources[2];
    EXPECT_EQ(primary_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(primary_resource.hasTier(Tier::HOST));
    EXPECT_EQ(primary_resource.disk_slot, primary_target);
    EXPECT_EQ(environment.disk_pools_[0]->refCount(primary_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));

    EXPECT_EQ(failed_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(failed_resource.host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->refCount(environment.host_blocks_[1]), 1u);
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(failed_target));
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.host_pools_[1]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);

    EXPECT_EQ(success_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(success_resource.hasTier(Tier::HOST));
    EXPECT_EQ(success_resource.disk_slot, success_target);
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
    ASSERT_EQ(cascadeGroupSetIds(*plan), (std::vector<size_t>{1, 2}));
    const BlockIdxType primary_target = plan->primary_desc.target_blocks[0];
    const BlockIdxType first_target   = plan->cascade_descs[0].target_blocks[0];
    const BlockIdxType missing_target = plan->cascade_descs[1].target_blocks[0];

    BlockTreeEvictor::CopyResultSet synthetic_results;
    synthetic_results.primary_success = true;
    synthetic_results.cascade_success = {true};
    environment.evictor_->complete(*plan, synthetic_results);
    EXPECT_TRUE(environment.transfer_group_set_ids_.empty());

    const auto& primary_resource = environment.node_->group_set_resources[0];
    const auto& first_resource   = environment.node_->group_set_resources[1];
    const auto& missing_resource = environment.node_->group_set_resources[2];
    EXPECT_EQ(primary_resource.disk_slot, primary_target);
    EXPECT_EQ(first_resource.disk_slot, first_target);
    EXPECT_EQ(environment.disk_pools_[0]->refCount(primary_target), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->refCount(first_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));
    EXPECT_FALSE(environment.host_pools_[1]->isAllocated(environment.host_blocks_[1]));

    EXPECT_EQ(missing_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(missing_resource.host_block, environment.host_blocks_[2]);
    EXPECT_EQ(environment.host_pools_[2]->refCount(environment.host_blocks_[2]), 1u);
    EXPECT_FALSE(environment.disk_pools_[2]->isAllocated(missing_target));
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.host_pools_[2]->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, RejectsResourcesReservedByAnotherPlan) {
    auto full_device_pool   = makeTestDevicePool(2, "reserved_plan_full_device");
    auto swa_device_pool    = makeTestDevicePool(2, "reserved_plan_swa_device");
    auto linear_device_pool = makeTestDevicePool(2, "reserved_plan_linear_device");
    ASSERT_NE(full_device_pool, nullptr);
    ASSERT_NE(swa_device_pool, nullptr);
    ASSERT_NE(linear_device_pool, nullptr);
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size    = 128;
    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    full_policy.enable_prefix_reuse   = true;
    swa_policy.enable_prefix_reuse    = true;
    linear_policy.enable_prefix_reuse = true;

    auto full_host_pool   = makePageableHostPool(2);
    auto swa_host_pool    = makePageableHostPool(2);
    auto linear_host_pool = makePageableHostPool(2);
    ASSERT_NE(full_host_pool, nullptr);
    ASSERT_NE(swa_host_pool, nullptr);
    ASSERT_NE(linear_host_pool, nullptr);
    auto full =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, full_host_pool, nullptr);
    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, swa_host_pool, nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{linear_device_pool}, linear_host_pool, nullptr);
    initializeGroups({full, swa, linear},
                     {full_device_pool, swa_device_pool, linear_device_pool},
                     {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16, 0, 128, 64),
                      block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16, 0, 128, 64),
                      block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16, 0, 128, 64)});

    std::vector<GroupSetPtr> groups = {full, swa, linear};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&        evictor        = *evictor_holder;
    initEvictor(evictor);

    MultiNodeBlocks full_blocks   = allocateDeviceBlocksForTest(*full, 1, BlockRefType::BLOCK_CACHE);
    MultiNodeBlocks swa_blocks    = allocateDeviceBlocksForTest(*swa, 1, BlockRefType::BLOCK_CACHE);
    MultiNodeBlocks linear_blocks = allocateDeviceBlocksForTest(*linear, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(full_blocks.size(), 1u);
    ASSERT_EQ(swa_blocks.size(), 1u);
    ASSERT_EQ(linear_blocks.size(), 1u);

    auto insert_result = tree.insertNode({100},
                                         {{makeResource(Tier::DEVICE, full_blocks[0][0]),
                                           makeResource(Tier::DEVICE, swa_blocks[0][0]),
                                           makeResource(Tier::DEVICE, linear_blocks[0][0])}},
                                         /*collect_path=*/false);
    ASSERT_NE(insertedNode(insert_result), nullptr);
    unreferenceDeviceBlocksForTest(*full, full_blocks, BlockRefType::BLOCK_CACHE);
    unreferenceDeviceBlocksForTest(*swa, swa_blocks, BlockRefType::BLOCK_CACHE);
    unreferenceDeviceBlocksForTest(*linear, linear_blocks, BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(insert_result);

    auto swa_victim = evictor.chooseVictim(1, Tier::DEVICE);
    ASSERT_TRUE(swa_victim.has_value());
    auto first_plan = evictor.buildPlan(*swa_victim);
    ASSERT_TRUE(first_plan.has_value());
    ASSERT_EQ(first_plan->cascade_descs.size(), 2u);
    EXPECT_EQ(first_plan->cascade_timings.size(), first_plan->cascade_descs.size());
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 1u);

    TransferDescriptor competing = BlockTreeEvictorTestPeer::makeDesc(
        evictor, insertedNode(insert_result), 0, Tier::DEVICE, Tier::HOST);
    EXPECT_FALSE(BlockTreeEvictorTestPeer::prepareDesc(evictor, competing));
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 1u);

    evictor.rollbackPreparedPlan(*first_plan);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[2].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 2u);
}

TEST(BlockTreeEvictorStatsTest, AggregatesCandidatesAcrossGroupsAndTiers) {
    auto device_pool = makeTestDevicePool(1, "block_tree_evictor_stats_device");
    auto host_pool   = makePageableHostPool(1);
    auto disk_pool   = makeTestDiskPool(1, "block_tree_evictor_stats_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group1_device_pool = makeTestDevicePool(1, "block_tree_evictor_stats_unused_device");
    ASSERT_NE(group1_device_pool, nullptr);
    auto group0 = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    auto group1 =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{group1_device_pool}, nullptr, disk_pool);
    initializeGroups(
        {group0, group1},
        {device_pool, group1_device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16),
         block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16)});
    std::vector<GroupSetPtr> groups = {group0, group1};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&        evictor        = *evictor_holder;
    initEvictor(evictor);

    MultiNodeBlocks    device_set = allocateDeviceBlocksForTest(*group0, 1, BlockRefType::BLOCK_CACHE);
    const BlockIdxType host_block = group0->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block = group1->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    const std::vector<std::vector<GroupSetResource>> first_resources = {
        {makeResource(Tier::DEVICE, device_set.front().front()), makeResource(Tier::DISK, disk_block)}};
    auto first = tree.insertNode({100}, first_resources, /*collect_path=*/false);
    unreferenceDeviceBlocksForTest(*group0, device_set, BlockRefType::BLOCK_CACHE);
    releaseLowerTierSeedRefs(groups, first_resources);
    evictor.onInsertCommitted(first);
    const std::vector<std::vector<GroupSetResource>> second_resources = {
        {makeResource(Tier::HOST, host_block), GroupSetResource{}}};
    auto second = tree.insertNode({200}, second_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, second_resources);
    evictor.onInsertCommitted(second);

    const CandidateStats stats = evictor.candidateStats();
    EXPECT_EQ(stats.device_candidates, 1u);
    EXPECT_EQ(stats.host_candidates, 1u);
    EXPECT_EQ(stats.disk_candidates, 1u);

    EXPECT_FALSE(evictor.chooseVictim(0, Tier::DISK).has_value());
    auto disk_victim = evictor.chooseVictim(1, Tier::DISK);
    ASSERT_TRUE(disk_victim.has_value());
    EXPECT_EQ(disk_victim->node, insertedNode(first));
    EXPECT_EQ(disk_victim->group_set_id, 1);

    group0->unmapDeviceBlocksFromTreeNode(
        MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set.front()}}});
    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    insertedNode(first)->group_set_resources[1].disk_slot   = NULL_BLOCK_IDX;
    insertedNode(second)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    unreferenceDeviceBlocksForTest(*group0, device_set, BlockRefType::BLOCK_CACHE);
    group0->releaseSingleBlock(Tier::HOST, host_block, BlockRefType::BLOCK_CACHE);
    group1->releaseSingleBlock(Tier::DISK, disk_block, BlockRefType::BLOCK_CACHE);
}

TEST(BlockTreeEvictorPolicyTest, MatchDoesNotChangeFifoAdmissionOrder) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_fifo_policy");
    ASSERT_NE(device_pool, nullptr);
    auto group = makeFullGroup(device_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&        evictor        = *evictor_holder;
    evictor.init(EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO);

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto first = tree.insertNode({100}, {{makeResource(Tier::DEVICE, device_set[0][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}},
                             BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode({200}, {{makeResource(Tier::DEVICE, device_set[1][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}},
                             BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(second);
    const uint64_t first_admission = insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq;

    evictor.onMatched({insertedNode(first)});

    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq, first_admission);
    auto first_victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(first_victim.has_value());
    EXPECT_EQ(first_victim->node, insertedNode(first));

    // No Host pool is configured, so preparation fails after reserving the
    // source and rolls it back. FIFO admission and relative victim order must
    // survive that rollback unchanged.
    EXPECT_FALSE(evictor.buildPlan(*first_victim).has_value());
    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq, first_admission);
    auto retried_victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(retried_victim.has_value());
    EXPECT_EQ(retried_victim->node, insertedNode(first));

    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}});
    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}});
    insertedNode(second)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockRefType::BLOCK_CACHE);
}

TEST(BlockTreeEvictorPolicyTest, ExistingGroupFillPrecedesNewSuffixAdmission) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_existing_fill_fifo");
    ASSERT_NE(device_pool, nullptr);
    auto group = std::make_shared<LinearGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, nullptr, nullptr);
    initializeGroups(
        {group},
        {device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::LINEAR), {0}, 16)});

    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&        evictor        = *evictor_holder;
    evictor.init(EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO);

    GroupSetResource empty_resource;
    empty_resource.device_blocks = {NULL_BLOCK_IDX};
    auto existing                = tree.insertNode({100}, {{empty_resource}}, /*collect_path=*/false);
    evictor.onInsertCommitted(existing);

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto mixed = tree.insertNode(
        {100, 200},
        {{makeResource(Tier::DEVICE, device_set[0][0])}, {makeResource(Tier::DEVICE, device_set[1][0])}},
        /*collect_path=*/false);
    ASSERT_EQ(mixed.adopted_nodes.size(), 1u);
    ASSERT_EQ(mixed.inserted_nodes.size(), 1u);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(mixed);

    TreeNode* filled_node = mixed.adopted_nodes.front().first;
    TreeNode* new_node    = mixed.inserted_nodes.front();
    ASSERT_NE(filled_node, nullptr);
    ASSERT_NE(new_node, nullptr);
    EXPECT_LT(filled_node->group_set_resources[0].candidate_meta.admission_seq,
              new_node->group_set_resources[0].candidate_meta.admission_seq);

    auto victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, filled_node);

    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{filled_node, device_set[0]}}});
    filled_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{new_node, device_set[1]}}});
    new_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockRefType::BLOCK_CACHE);
}

TEST(BlockTreeEvictorPolicyTest, MatchUpdatesLfuHitCountAndOrder) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_lfu_policy");
    ASSERT_NE(device_pool, nullptr);
    auto group = makeFullGroup(device_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree, BlockTreeEvictor::ExecuteTransferFn{});
    BlockTreeEvictor&        evictor        = *evictor_holder;
    evictor.init(EvictionPolicy::LFU, EvictionPolicy::LRU, EvictionPolicy::FIFO);

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto first = tree.insertNode({100}, {{makeResource(Tier::DEVICE, device_set[0][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}},
                             BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(first);
    auto second = tree.insertNode({200}, {{makeResource(Tier::DEVICE, device_set[1][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}},
                             BlockRefType::BLOCK_CACHE);
    evictor.onInsertCommitted(second);

    evictor.onMatched({insertedNode(first)});

    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.hit_count, 1u);
    EXPECT_EQ(insertedNode(second)->group_set_resources[0].candidate_meta.hit_count, 0u);
    auto victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(second));

    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}});
    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group->unmapDeviceBlocksFromTreeNode(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}});
    insertedNode(second)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockRefType::BLOCK_CACHE);
}

}  // namespace
}  // namespace rtp_llm
