#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/test/TestLayoutSpec.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a tiny multi-pool config with two groups: gid=0 LINEAR(layers 0,1)
// and gid=1 FULL(layers 2,3). Each group has its own per-group block budget,
// so CoordinatorCacheManager creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t       linear_block_num = 6,
                                                 uint32_t       full_block_num   = 8,
                                                 CacheGroupType second_type      = CacheGroupType::FULL) {
    CacheConfig config;
    config.dtype              = rtp_llm::DataType::TYPE_FP16;
    config.layer_num          = 4;

    config.block_num          = std::max(linear_block_num, full_block_num);
    config.seq_size_per_block = 4;
    config.linear_step        = 2;

    auto linear_spec = makeResolvedLinearSpec(config.dtype,
                                              1,
                                              1,
                                              1,
                                              1,
                                              2,
                                              static_cast<uint32_t>(config.seq_size_per_block),
                                              config.dtype,
                                              config.dtype,
                                              "linear");
    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, static_cast<uint32_t>(config.seq_size_per_block), "full");

    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, second_type},
                            {"linear", second_type == CacheGroupType::SWA ? "swa" : "full"});

    const auto linear_stride = linear_spec->block_size_bytes();
    const auto full_stride   = full_spec->block_size_bytes();
    rtp_llm::test::setGroupBlockLayout(
        config, {linear_block_num, full_block_num}, {linear_stride, full_stride}, {0, 0});
    return config;
}

static CacheConfig makeTinySwaMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t swa_block_num = 8) {
    return makeTinyMultiPoolHybridConfig(linear_block_num, swa_block_num, CacheGroupType::SWA);
}

static ModelConfig makeTinyDSV4ModelConfig() {
    ModelConfig mc;
    mc.num_layers                                                = 5;
    mc.hidden_size                                               = 32;
    mc.attn_config.head_num                                      = 4;
    mc.attn_config.kv_head_num                                   = 1;
    mc.attn_config.size_per_head                                 = 8;
    mc.attn_config.rope_head_dim                                 = 4;
    mc.attn_config.indexer_head_dim                              = 8;
    mc.attn_config.indexer_head_num                              = 2;
    mc.attn_config.indexer_topk                                  = 16;
    mc.attn_config.tokens_per_block                              = 128;
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc, {4, 128, 4, 128, 0});
    return mc;
}

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.tokens_per_block = 128;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    setDsv4KvCacheSpecs(mc, ratios);
    return mc;
}

// Build a DSV4 7-pool CacheConfig.
static CacheConfig makeDSV4HybridPoolConfig(uint32_t block_num = 200) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.finalizeBlockNums(block_num, RuntimeConfig{});
    return config;
}

static void setExplicitBlocksForGroup(CacheConfig& config, size_t group_id, uint32_t block_num) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        policies.push_back(config.policyForGroup(gid));
    }
    policies[group_id].explicit_block_num     = block_num;
    policies[group_id].charge_to_paged_budget = block_num > 0;
    config.setGroupPolicies(policies);
}

// Move every explicitly-sized independent group (DSV4's fixed state pools) onto pinned host
// memory. Residency and budget are independent knobs, so a host-resident pool must also drop
// charge_to_paged_budget -- checkGroupResidencyBudget() enforces exactly that pairing.
static std::vector<size_t> setPinnedHostPlacementForExplicitIndependentGroups(CacheConfig& config) {
    std::vector<CacheGroupPolicy> policies;
    std::vector<size_t>           pinned_gids;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        auto policy = config.policyForGroup(gid);
        if (policy.evict_policy == CacheEvictPolicy::INDEPENDENT && policy.explicit_block_num > 0) {
            policy.memory_placement       = CacheMemoryPlacement::HOST_PINNED;
            policy.charge_to_paged_budget = false;
            pinned_gids.push_back(gid);
        }
        policies.push_back(policy);
    }
    config.setGroupPolicies(policies);
    return pinned_gids;
}

static size_t firstExplicitIndependentGroup(const CacheConfig& config) {
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto policy = config.policyForGroup(gid);
        if (policy.evict_policy == CacheEvictPolicy::INDEPENDENT && policy.explicit_block_num > 0) {
            return gid;
        }
    }
    ADD_FAILURE() << "missing explicit independent cache group";
    return 0;
}

static CompleteTokenIdsPtr makeCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block) {
    auto  cti        = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto  input_ids  = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* token_data = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);
    return cti;
}

static BatchKVCacheResourcePtr makeBatchResource(int batch_size, const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.topologyPtr());
    return res;
}

static std::vector<uint32_t> groupBlockNumsSnapshot(const CacheConfig& config) {
    std::vector<uint32_t> block_nums;
    block_nums.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        block_nums.push_back(config.blockNumForGroup(gid));
    }
    return block_nums;
}

static void setGroupBlockNums(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(gid));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(gid));
    }
    rtp_llm::test::setGroupBlockLayout(config, block_nums, kv_strides, scale_strides);
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

// Create CoordinatorCacheManager with SharedBlockCache injected (required before init()).
static CoordinatorCacheManagerPtr
makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION, int64_t reserve_block_ratio = 0) {
    auto coordinator_manager = std::make_shared<CoordinatorCacheManager>(
        config, AllocationType::DEVICE, nullptr, reserve_block_ratio, role_type);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    coordinator_manager->setSharedBlockCache(shared_cache);
    return coordinator_manager;
}

class RecordingMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool gpu, uint64_t) override {
        reg_gpu_flags.push_back(gpu);
        return true;
    }

    bool deregUserMr(void*, bool gpu) override {
        dereg_gpu_flags.push_back(gpu);
        return true;
    }

    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }

    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }

    bool isRdmaMode() override {
        return true;
    }

    std::vector<bool> reg_gpu_flags;
    std::vector<bool> dereg_gpu_flags;
};

class RecordingCacheStore: public CacheStore {
public:
    explicit RecordingCacheStore(std::shared_ptr<MemoryUtil> memory_util): memory_util_(std::move(memory_util)) {}

    void store(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreStoreDoneCallback callback) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return nullptr;
    }

    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return nullptr;
    }

    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return memory_util_;
    }

    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> memory_util_;
};

// Insert a non-resident cache item into the shared block cache for a specific group.
// Returns the BlockIdx allocated for the item (kept blockCache-referenced + request-released).
static BlockIdxType
seedNonResidentCacheItem(const CoordinatorCacheManagerPtr& coordinator_manager, int gid, CacheKeyType key) {
    auto pool   = coordinator_manager->group_block_pools_[static_cast<size_t>(gid)];
    auto blocks = pool->malloc(1);
    EXPECT_EQ(blocks.size(), 1u);
    auto                      shared_cache = coordinator_manager->sharedBlockCache();
    std::vector<BlockIdxType> group_block_ids(coordinator_manager->group_block_pools_.size(), NULL_BLOCK_IDX);
    group_block_ids[static_cast<size_t>(gid)] = blocks[0];
    shared_cache->put(key, group_block_ids, false);
    // SharedBlockCache::put() internally calls pool->blockCacheReference()
    pool->requestFree(blocks);
    return blocks[0];
}

struct PoolCounters {
    size_t free_blocks;
    size_t available_blocks;
    size_t request_refs;
    size_t block_cache_refs;
    size_t connector_refs;
};

static std::vector<PoolCounters> snapshotPoolCounters(const CoordinatorCacheManagerPtr& coordinator_manager) {
    std::vector<PoolCounters> counters;
    counters.reserve(coordinator_manager->group_block_pools_.size());
    for (const auto& pool : coordinator_manager->group_block_pools_) {
        counters.push_back({pool->freeBlocksNum(),
                            pool->availableBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            pool->connectorRefBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const CoordinatorCacheManagerPtr& coordinator_manager,
                                 const std::vector<PoolCounters>&  expected) {
    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), expected.size());
    for (size_t gid = 0; gid < expected.size(); ++gid) {
        const auto& pool = coordinator_manager->group_block_pools_[gid];
        EXPECT_EQ(pool->freeBlocksNum(), expected[gid].free_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->availableBlocksNum(), expected[gid].available_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->requestRefBlocksNum(), expected[gid].request_refs) << "gid=" << gid;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), expected[gid].block_cache_refs) << "gid=" << gid;
        EXPECT_EQ(pool->connectorRefBlocksNum(), expected[gid].connector_refs) << "gid=" << gid;
    }
}

class CoordinatorCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 2u);
    EXPECT_NE(coordinator_manager->group_block_pools_[0], coordinator_manager->group_block_pools_[1]);

    // Per-pool totalBlocksNum = group_block_nums[gid] - 1 (block 0 reserved).
    EXPECT_EQ(coordinator_manager->group_block_pools_[0]->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(coordinator_manager->group_block_pools_[1]->totalBlocksNum(), 8u - 1u);
}

TEST_F(CoordinatorCacheManagerTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 2u);
    EXPECT_EQ(coordinator_manager->group_block_pools_[0]->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(coordinator_manager->group_block_pools_[1]->where(), MemoryType::MEMORY_GPU);
}

TEST_F(CoordinatorCacheManagerTest, SoleGroupBlockPoolReturnsTheIndependentPool) {
    auto config    = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                           /*block_num=*/8,
                                           /*tokens_per_block=*/4,
                                           DataType::TYPE_FP16,
                                           /*local_head_num_kv=*/1,
                                           /*size_per_head=*/1);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 1u);
    EXPECT_EQ(coordinator_manager->soleGroupBlockPool(), coordinator_manager->group_block_pools_[0]);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, TotalAndFreeBlocksAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const size_t expected_total = (6u - 1u) + (8u - 1u);
    EXPECT_EQ(coordinator_manager->totalBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->availableBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->notInUseBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator_manager->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerTest, TokenAggregatorsUseDifferentCapacityScopes) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    // Token capacity aggregators use FULL groups first: 7 blocks * 4 tokens.
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    EXPECT_EQ(coordinator_manager->maxAvailableTokensNum(), 28u);
    EXPECT_EQ(coordinator_manager->availableTokensNum(), 28u);
    EXPECT_EQ(coordinator_manager->totalTokensNum(), 28u);
}

TEST_F(CoordinatorCacheManagerTest, TokenAggregatorsUseCPVirtualBlockSizeForFullGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    EXPECT_EQ(coordinator_manager->maxAvailableTokensNum(), 7u * 4u);
    EXPECT_EQ(coordinator_manager->availableTokensNum(), 7u * 4u);

    coordinator_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(coordinator_manager->maxAvailableTokensNum(), 7u * 8u);
    EXPECT_EQ(coordinator_manager->availableTokensNum(), 7u * 8u);
}

TEST_F(CoordinatorCacheManagerTest, TokenAggregatorsFallBackToGlobalSeqSize) {
    auto config               = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.seq_size_per_block = 4;
    auto coordinator_manager  = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    EXPECT_EQ(coordinator_manager->maxAvailableTokensNum(), 5u * 4u);
    EXPECT_EQ(coordinator_manager->availableTokensNum(), 5u * 4u);
}

TEST_F(CoordinatorCacheManagerTest, RequestAndConnectorRefAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto pool0 = coordinator_manager->group_block_pools_[0];
    auto pool1 = coordinator_manager->group_block_pools_[1];

    const size_t free_total_before = coordinator_manager->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2);
    auto         g1_blocks         = pool1->malloc(3);
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);

    EXPECT_EQ(coordinator_manager->requestRefBlocksNum(), 5u);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), free_total_before - 5u);
    EXPECT_EQ(coordinator_manager->availableBlocksNum(), free_total_before - 5u);

    // Mark some blocks as connector-referenced (simulating cache transfer).
    pool0->connectorReference(g0_blocks[0]);
    pool1->connectorReference(g1_blocks[0]);
    EXPECT_EQ(coordinator_manager->connectorRefBlocksNum(), 2u);

    pool0->requestFree(g0_blocks);
    pool1->requestFree(g1_blocks);
    EXPECT_EQ(coordinator_manager->requestRefBlocksNum(), 0u);

    // Connector still holds 2 blocks → freeBlocksNum (set of returnable
    // ids) drops by 2; notInUseBlocksNum counts blocks not held by *request*
    // or *block cache* refs, so connector-held blocks still count as "not
    // in use" → equals the full pool total.
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), free_total_before - 2u);
    EXPECT_EQ(coordinator_manager->notInUseBlocksNum(), free_total_before);

    pool0->connectorFree(g0_blocks[0]);
    pool1->connectorFree(g1_blocks[0]);
    EXPECT_EQ(coordinator_manager->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), free_total_before);
    EXPECT_EQ(coordinator_manager->notInUseBlocksNum(), free_total_before);
}

TEST_F(CoordinatorCacheManagerTest, BlockCacheRefAggregatesAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    seedNonResidentCacheItem(coordinator_manager, /*gid=*/0, /*key=*/100);
    seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/200);
    seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/201);

    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 3u);
}

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // Layer in linear group.
    {
        auto addr = coordinator_manager->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = coordinator_manager->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
    // Layer in full group.
    {
        auto addr = coordinator_manager->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = coordinator_manager->convertIndexToBuffer(/*layer_id=*/3, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
}

TEST_F(CoordinatorCacheManagerTest, ConvertIndexToBufferPartitionDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto bufs = coordinator_manager->convertIndexToBuffer(
        /*layer_id=*/3, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(CoordinatorCacheManagerTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto addr_default   = coordinator_manager->convertIndexToAddr(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    auto addr_via_layer = coordinator_manager->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_via_layer.kv_addr);

    auto bufs_default = coordinator_manager->convertIndexToBuffer(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = coordinator_manager->convertIndexToBuffer(
        /*layer_id=*/0, /*group_id=*/0, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(CoordinatorCacheManagerTest, AllLayerCacheBaseExposesPerLayerAndPerGroupTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto layout = coordinator_manager->allLayerCacheBase();
    EXPECT_EQ(layout.topology().layerGroupIdsSnapshot(), config.layerGroupIdsSnapshot());
    EXPECT_EQ(layout.topology().groupTypesSnapshot(), config.groupTypesSnapshot());
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num()); ++i) {
        const auto& layer = layout.topology().layer(static_cast<int>(i));
        ASSERT_FALSE(layer.group_tags.empty());
        for (const auto& tag : layer.group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i)) << "layer " << i << " tag=" << tag;
        }
    }
}

// ---------------------------------------------------------------------------
// regUserMr / getMrCostTimeMs
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, RegUserMrWithoutCacheStoreIsNoOpAndZeroCost) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // No CacheStore is plumbed in: regUserMr should be a benign no-op for every
    // group pool, and the aggregated MR cost remains zero.
    EXPECT_NO_THROW(coordinator_manager->regUserMr(/*model_id=*/0, /*cache_store=*/nullptr));
    EXPECT_EQ(coordinator_manager->getMrCostTimeMs(), 0);
}

// ---------------------------------------------------------------------------
// popBlocksFromCache / blockCacheFree
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, PopBlocksFromCacheReturnsEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // Seed identical key on both groups, plus a unique key on the full group.
    auto g0_block_for_100 = seedNonResidentCacheItem(coordinator_manager, /*gid=*/0, /*key=*/100);
    auto g1_block_for_100 = seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/100);
    auto g1_block_for_200 = seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/200);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 3u);

    auto evicted = coordinator_manager->popBlocksFromCache(/*min_blocks_to_free=*/3);
    ASSERT_NE(evicted, nullptr);
    EXPECT_EQ(evicted->batchSize(), 1);
    EXPECT_EQ(evicted->groupNums(), 2);
    EXPECT_TRUE(evicted->cacheResource(0).cacheKeysAreCpCanonical());
    const auto& keys = evicted->cacheKeys(0);
    EXPECT_EQ(keys.size(), 2u);  // 100 (shared) + 200 (g1 only)

    std::unordered_set<CacheKeyType> key_set(keys.begin(), keys.end());
    EXPECT_TRUE(key_set.count(100));
    EXPECT_TRUE(key_set.count(200));

    // Per-group block ids: each group's block should be set only at the matching position.
    // matching the key it owned, and NULL elsewhere.
    const auto& g0_blocks = evicted->blocks(/*batch_id=*/0, /*gid=*/0);
    const auto& g1_blocks = evicted->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 2u);

    auto idx_of = [&](CacheKeyType k) -> size_t {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == k) {
                return i;
            }
        }
        return keys.size();
    };
    const size_t pos_100 = idx_of(100);
    const size_t pos_200 = idx_of(200);
    ASSERT_LT(pos_100, keys.size());
    ASSERT_LT(pos_200, keys.size());

    EXPECT_EQ(g0_blocks[pos_100], g0_block_for_100);
    EXPECT_TRUE(isNullBlockIdx(g0_blocks[pos_200]));
    EXPECT_EQ(g1_blocks[pos_100], g1_block_for_100);
    EXPECT_EQ(g1_blocks[pos_200], g1_block_for_200);
}

TEST_F(CoordinatorCacheManagerTest, PopBlocksFromCacheZeroFreeReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());
    EXPECT_EQ(coordinator_manager->popBlocksFromCache(0), nullptr);
}

TEST_F(CoordinatorCacheManagerTest, PopBlocksFromCacheEmptyCachesReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());
    EXPECT_EQ(coordinator_manager->popBlocksFromCache(/*min_blocks_to_free=*/4), nullptr);
}

TEST_F(CoordinatorCacheManagerTest, BlockCacheFreeReleasesEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    seedNonResidentCacheItem(coordinator_manager, /*gid=*/0, /*key=*/100);
    seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/200);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 2u);

    const size_t free_before = coordinator_manager->freeBlocksNum();
    auto         evicted     = coordinator_manager->popBlocksFromCache(/*min_blocks_to_free=*/2);
    ASSERT_NE(evicted, nullptr);
    // Eviction releases the LRU entries from BlockCache; the underlying blocks
    // are still referenced by blockCacheRef. Releasing those refs is what
    // blockCacheFree() does.
    coordinator_manager->blockCacheFree(evicted);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), free_before + 2u);
}

TEST_F(CoordinatorCacheManagerTest, BlockCacheFreeNullPtrIsNoOp) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());
    EXPECT_NO_THROW(coordinator_manager->blockCacheFree(nullptr));
}

TEST_F(CoordinatorCacheManagerTest, BlockCacheFreeIgnoresDuplicateAndNullBlockIds) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto seeded = seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/300);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 1u);

    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(config.topologyPtr());
    // Same block listed twice in the same group should only be released once;
    // NULL_BLOCK_IDX entries should be skipped.
    batch->mutableBlockIds(0, /*gid=*/1).assign(BlockIndicesType{seeded, seeded, NULL_BLOCK_IDX});
    EXPECT_NO_THROW(coordinator_manager->blockCacheFree(batch));
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 0u);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Group 0 (linear) gets 6 blocks (5 free), group 1 (full) gets 4 blocks (3 free).
    // total_available = 8. Set reserve = 4.
    // Expected per-group reserve: floor(4 * 5/8) = 2 for gid=0, floor(4 * 3/8) = 1 for gid=1.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    coordinator_manager->setReserveBlocksNum(4);

    // seq_len=4 -> 1 block per group.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = coordinator_manager->malloc(malloc_info);
    EXPECT_TRUE(result.success);
}

TEST_F(CoordinatorCacheManagerTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    // Force a group whose available_blocks < need + group_reserve_blocks.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // A reserve large enough to hide most blocks should reject init malloc.
    coordinator_manager->setReserveBlocksNum(coordinator_manager->availableBlocksNum());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;
    auto result                     = coordinator_manager->malloc(malloc_info);
    EXPECT_FALSE(result.success);
}

TEST_F(CoordinatorCacheManagerTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    constexpr size_t reserve_blocks = 6;
    coordinator_manager->setReserveBlocksNum(reserve_blocks);

    const auto snapshots = coordinator_manager->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ("linear", snapshots[0].pool_name);
    EXPECT_EQ("full", snapshots[1].pool_name);

    const size_t total_reservable_available_blocks = snapshots[0].available_blocks + snapshots[1].available_blocks;
    ASSERT_GT(total_reservable_available_blocks, 0u);
    EXPECT_EQ(reserve_blocks * snapshots[0].available_blocks / total_reservable_available_blocks,
              snapshots[0].reserve_blocks);
    EXPECT_EQ(reserve_blocks * snapshots[1].available_blocks / total_reservable_available_blocks,
              snapshots[1].reserve_blocks);
}

TEST_F(CoordinatorCacheManagerTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/20, /*full_block_num=*/6);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    coordinator_manager->setReserveBlocksNum(1);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
    coordinator_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;

    auto result = coordinator_manager->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, /*gid=*/1)), 4u);

    FreeInfo free_info{batch_res, token_ids};
    coordinator_manager->free(free_info);
}

TEST_F(CoordinatorCacheManagerTest, ReserveCheckIsBypassedWhenMallocInfoLacksContext) {
    // hasAvailableBlocksForReserve returns true when info has no resource/tokens.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    MallocInfo info{};
    EXPECT_TRUE(coordinator_manager->hasAvailableBlocksForReserve(info, /*reserve_blocks=*/9999));
}

TEST_F(CoordinatorCacheManagerTest, InitMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    // gid=0 has enough room for the LINEAR tail block; gid=1 cannot satisfy
    // the 3 FULL blocks needed for seq_len=9. Whichever stage rejects -- the
    // per-group capacity preflight or initMallocForCommonLen's group loop --
    // both pools must end up exactly as they started.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/3, /*full_block_num=*/3);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const auto counters_before = snapshotPoolCounters(coordinator_manager);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;

    auto result = coordinator_manager->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    // A 3-block pool exposes 2 usable blocks (block 0 is the null sentinel), so 3 FULL blocks can
    // never fit: the per-group total-capacity test must report PERMANENT so the scheduler errors
    // the stream out instead of parking it in WAITING forever.
    EXPECT_EQ(result.status, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(coordinator_manager->requestRefBlocksNum(), 0u);
    expectPoolCountersEq(coordinator_manager, counters_before);
}

// The same request that a live holder makes un-satisfiable must come back RETRYABLE (stream stays
// WAITING) rather than PERMANENT, and must actually succeed once the holder releases its blocks.
TEST_F(CoordinatorCacheManagerTest, InitMallocReportsRetryablePerGroupCapacityShortage) {
    // Each pool has enough empty-engine capacity for seq_len=8. A live holder
    // leaves the FULL pool one block short, so only the current admission is
    // retryable; the request is not permanently oversized.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto holder_resource = makeBatchResource(/*batch_size=*/1, config);
    holder_resource->setBatchCacheKeys(0, CacheKeysType{100});
    auto       holder_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo holder_info{holder_resource, holder_tokens};
    holder_info.enable_device_cache = false;
    holder_info.reuse_cache         = false;
    ASSERT_TRUE(coordinator_manager->malloc(holder_info).success);

    auto deferred_resource = makeBatchResource(/*batch_size=*/1, config);
    deferred_resource->setBatchCacheKeys(0, CacheKeysType{200, 201});
    auto       deferred_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo deferred_info{deferred_resource, deferred_tokens};
    deferred_info.enable_device_cache = false;
    deferred_info.reuse_cache         = false;
    deferred_info.verbose             = false;

    auto deferred_result = coordinator_manager->malloc(deferred_info);
    EXPECT_FALSE(deferred_result.success);
    EXPECT_EQ(deferred_result.status, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(deferred_resource->curBlocksNum(), 0u);

    coordinator_manager->free(FreeInfo{holder_resource, holder_tokens});
    auto retry_result = coordinator_manager->malloc(deferred_info);
    EXPECT_TRUE(retry_result.success);
}

TEST_F(CoordinatorCacheManagerTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const auto linear_cached = seedNonResidentCacheItem(coordinator_manager, /*gid=*/0, /*key=*/100);
    const auto full_cached   = seedNonResidentCacheItem(coordinator_manager, /*gid=*/1, /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    ASSERT_EQ(coordinator_manager->requestRefBlocksNum(), 0u);
    ASSERT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 2u);

    const size_t available_before = coordinator_manager->availableBlocksNum();
    const auto   counters_before  = snapshotPoolCounters(coordinator_manager);
    coordinator_manager->setReserveBlocksNum(std::max<size_t>(1, available_before * 8));

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    auto result = coordinator_manager->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(coordinator_manager->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator_manager->blockCacheRefBlocksNum(), 2u);
    expectPoolCountersEq(coordinator_manager, counters_before);
}

TEST_F(CoordinatorCacheManagerTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/2);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(coordinator_manager->malloc(init_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];
    const auto counters_before     = snapshotPoolCounters(coordinator_manager);

    // gid=0 can append one real LINEAR tail block. gid=1 has no remaining
    // free blocks and no cache to evict, so FULL allocation fails.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = coordinator_manager->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);
    expectPoolCountersEq(coordinator_manager, counters_before);
}

TEST_F(CoordinatorCacheManagerTest, IncrMallocRollbackRestoresLinearBackfilledSlots) {
    // Block 0 is reserved by each pool, so FULL needs three configured blocks
    // to provide the two request blocks used by the initial allocation.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(coordinator_manager->malloc(init_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 2u);

    auto& linear_ids       = batch_res->mutableBlockIds(0, /*gid=*/0);
    auto  removed_block_id = linear_ids.blocks()[1];
    ASSERT_FALSE(isNullBlockIdx(removed_block_id));
    coordinator_manager->group_block_pools_[0]->requestFree({removed_block_id});
    linear_ids.setAt(1, NULL_BLOCK_IDX);
    const auto counters_before = snapshotPoolCounters(coordinator_manager);

    // LINEAR first backfills the old sparse tail and appends a new tail block.
    // FULL then fails because its independent pool is exhausted. Rollback must
    // restore both the historical NULL slot and the original logical length.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    EXPECT_FALSE(coordinator_manager->malloc(incr_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 2u);
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, /*gid=*/0)[1]));
    expectPoolCountersEq(coordinator_manager, counters_before);
}

// ---------------------------------------------------------------------------
// Full malloc / free cycle
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, MallocAndFreeCycleAcrossPerGroupPools) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/8, /*full_block_num=*/8);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const size_t free_before = coordinator_manager->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = coordinator_manager->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_LT(coordinator_manager->freeBlocksNum(), free_before);

    FreeInfo free_info{batch_res, token_ids};
    coordinator_manager->free(free_info);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), free_before);
}

// ---------------------------------------------------------------------------
// DSV4 7-group HybridPool: covers per-tag addressing and SWA tail
// ---------------------------------------------------------------------------

TEST_F(CoordinatorCacheManagerTest, DSV4InitAndAggregatedCounters) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    EXPECT_EQ(config.groupNums(), 7);
    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 7u);

    // Sum of per-pool totals must equal aggregated totalBlocksNum.
    size_t expected_total = 0;
    for (const auto& pool : coordinator_manager->group_block_pools_) {
        expected_total += pool->totalBlocksNum();
    }
    EXPECT_EQ(coordinator_manager->totalBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->freeBlocksNum(), expected_total);
    EXPECT_EQ(coordinator_manager->availableBlocksNum(), expected_total);
}

TEST_F(CoordinatorCacheManagerTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 7u);
    for (size_t gid = 0; gid < coordinator_manager->group_block_pools_.size(); ++gid) {
        EXPECT_EQ(coordinator_manager->group_block_pools_[gid]->where(), MemoryType::MEMORY_GPU)
            << "gid=" << gid << " tag=" << config.tagForGroup(gid);
    }
}

// memory_placement=HOST_PINNED must move only the opted-in pools off HBM; every other pool of
// the same DSV4 config stays on the device.
TEST_F(CoordinatorCacheManagerTest, DSV4FixedTagPoolsUsePinnedHostBackingWhenPlacementIsHostPinned) {
    auto       config      = makeDSV4HybridPoolConfig(/*block_num=*/200);
    const auto pinned_gids = setPinnedHostPlacementForExplicitIndependentGroups(config);
    ASSERT_FALSE(pinned_gids.empty());
    ASSERT_LT(pinned_gids.size(), static_cast<size_t>(config.groupNums()));

    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    ASSERT_EQ(coordinator_manager->group_block_pools_.size(), 7u);
    const std::unordered_set<size_t> pinned_set(pinned_gids.begin(), pinned_gids.end());
    for (size_t gid = 0; gid < coordinator_manager->group_block_pools_.size(); ++gid) {
        const bool expect_pinned = pinned_set.count(gid) > 0;
        EXPECT_EQ(coordinator_manager->group_block_pools_[gid]->where(),
                  expect_pinned ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_GPU)
            << "gid=" << gid << " tag=" << config.tagForGroup(gid);
    }
}

TEST_F(CoordinatorCacheManagerTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_gid), "hca_state");
    ASSERT_GT(coordinator_manager->group_block_pools_.size(), static_cast<size_t>(hca_state_gid));

    const size_t hca_free_before = coordinator_manager->group_block_pools_[hca_state_gid]->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto result                     = coordinator_manager->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, hca_state_gid);
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - coordinator_manager->group_block_pools_[hca_state_gid]->freeBlocksNum(), 1u);
}

TEST_F(CoordinatorCacheManagerTest, TokenAggregatorsIgnoreSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_gid), "hca_state");
    auto block_nums           = groupBlockNumsSnapshot(config);
    block_nums[hca_state_gid] = 2;
    setGroupBlockNums(config, block_nums);

    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());
    ASSERT_GT(coordinator_manager->group_block_pools_.size(), static_cast<size_t>(hca_state_gid));

    const auto hca_state_tokens =
        coordinator_manager->group_block_pools_[hca_state_gid]->totalBlocksNum() * config.seq_size_per_block;
    EXPECT_LT(hca_state_tokens, coordinator_manager->totalTokensNum());
    EXPECT_EQ(coordinator_manager->availableTokensNum(), coordinator_manager->maxAvailableTokensNum());
    EXPECT_EQ(coordinator_manager->totalTokensNum(), coordinator_manager->maxAvailableTokensNum());
}

TEST_F(CoordinatorCacheManagerTest, DSV4ConfigUsesGroupOwnedBlockSizes) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(config.groupNums(), 7);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto expected_group_bytes =
            config.layerIdsForGroup(gid).size()
            * (config.kvBlockStrideBytesForGroup(gid) + config.kvScaleStrideBytesForGroup(gid));
        EXPECT_EQ(config.blockSizeBytesForGroup(gid), expected_group_bytes) << "gid=" << gid;
    }
}

TEST_F(CoordinatorCacheManagerTest, ReserveRatioExcludesExplicitIndependentPools) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    ASSERT_LT(firstExplicitIndependentGroup(config), static_cast<size_t>(config.groupNums()));

    constexpr int64_t reserve_ratio = 10;
    auto              coordinator_manager = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(coordinator_manager->init());

    size_t reservable_available = 0;
    size_t all_available        = 0;
    for (size_t gid = 0; gid < coordinator_manager->group_block_pools_.size(); ++gid) {
        const size_t available = coordinator_manager->group_block_pools_[gid]->availableBlocksNum();
        all_available += available;
        if (!config.usesExplicitIndependentBlocks(gid)) {
            reservable_available += available;
        }
    }
    ASSERT_GT(reservable_available, 0u);
    ASSERT_GT(all_available, reservable_available);
    EXPECT_EQ(coordinator_manager->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * reservable_available / static_cast<size_t>(100));
    EXPECT_NE(coordinator_manager->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * all_available / static_cast<size_t>(100));
}

TEST_F(CoordinatorCacheManagerTest, DSV4FinalizeBlockNumsUsesHcaStatePoolBlocks) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
}

TEST_F(CoordinatorCacheManagerTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);
    setExplicitBlocksForGroup(config, firstExplicitIndependentGroup(config), 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 123u);
    }
}

TEST_F(CoordinatorCacheManagerTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
}

// Mirror image of DSV4GpuHcaStatePoolIncludesFixedReserve: a pinned-host pool keeps its explicit
// block count but must NOT be deducted from the device paged budget, otherwise the KV cache
// silently shrinks by bytes that never live in HBM.
TEST_F(CoordinatorCacheManagerTest, DSV4PinnedHcaStatePoolExcludesFixedReserve) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);
    const auto pinned_gids = setPinnedHostPlacementForExplicitIndependentGroups(config);
    ASSERT_EQ(pinned_gids.size(), 1u);
    ASSERT_EQ(pinned_gids.front(), explicit_gid);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    // Block counts are unaffected by residency: the explicit pool still gets its 50 blocks.
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    EXPECT_GT(config.blockSizeBytesForGroup(explicit_gid), 0u);
}

TEST_F(CoordinatorCacheManagerTest, DSV4StateSwaPoolsWithoutExplicitBlocksScaleWithLinearStep) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.linear_step = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.typeForGroup(gid) == CacheGroupType::SWA ? 32u : 128u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
}

TEST_F(CoordinatorCacheManagerTest, FinalizeNonExplicitSwaBlocksUsesCeilDivision) {
    auto config        = makeTinySwaMultiPoolHybridConfig();
    config.linear_step = 4;
    RuntimeConfig rt;

    config.finalizeBlockNums(/*global_block_num=*/1, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear gid=*/0), 1u);
    EXPECT_EQ(config.blockNumForGroup(/*swa gid=*/1), 1u);

    config.finalizeBlockNums(/*global_block_num=*/8, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear gid=*/0), 8u);
    EXPECT_EQ(config.blockNumForGroup(/*swa gid=*/1), 2u);

    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear gid=*/0), 9u);
    EXPECT_EQ(config.blockNumForGroup(/*swa gid=*/1), 3u);

    config.linear_step = 1;
    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear gid=*/0), 9u);
    EXPECT_EQ(config.blockNumForGroup(/*swa gid=*/1), 9u);
}

TEST_F(CoordinatorCacheManagerTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num(); ++l) {
        const auto& tags = config.topology().layer(static_cast<int>(l)).group_tags;
        if (std::find(tags.begin(), tags.end(), "csa_kv") != tags.end()) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // csa_kv tag routes to gid=0; it must produce a non-null kv address that
    // matches the CSA group's pool.
    auto addr_csa = coordinator_manager->convertIndexToAddrByTag(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);
    const auto csa_gid = config.groupIdForTag("csa_kv");
    EXPECT_EQ(addr_csa.kv_addr, coordinator_manager->convertIndexToAddr(csa_layer, csa_gid, 1).kv_addr);

    auto addr_swa = coordinator_manager->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);
    EXPECT_THROW((void)coordinator_manager->convertIndexToAddrByTag(csa_layer, "missing", 1), std::exception);
    EXPECT_THROW((void)coordinator_manager->convertIndexToAddr(csa_layer, config.groupNums(), 1), std::exception);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_THROW((void)coordinator_manager->convertIndexToAddr(csa_layer, /*block_id=*/1), std::exception);
}

TEST_F(CoordinatorCacheManagerTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num(); ++l) {
        const auto& tags = config.topology().layer(static_cast<int>(l)).group_tags;
        if (std::find(tags.begin(), tags.end(), "csa_kv") != tags.end()) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    auto buf = coordinator_manager->convertIndexToBufferByTag(csa_layer, "csa_kv", /*block_id=*/1);
    ASSERT_FALSE(buf.empty());
    EXPECT_NE(buf[0].addr, nullptr);

    auto buf_part = coordinator_manager->convertIndexToBufferByTag(
        csa_layer, "csa_kv", /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(buf_part.empty());
    EXPECT_NE(buf_part[0].addr, nullptr);
}

TEST_F(CoordinatorCacheManagerTest, DSV4AllLayerCacheBaseHasPerGroupTensors) {
    auto config    = makeDSV4HybridPoolConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    auto layout = coordinator_manager->allLayerCacheBase();
    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num()); ++l) {
        EXPECT_TRUE(layout.group("swa_kv").hasLayer(l)) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.groups().size(), 7u);
    EXPECT_EQ(layout.topology().groups().size(), 7u);
}

TEST_F(CoordinatorCacheManagerTest, DSV4SharedBlockCacheIsUnifiedAcrossGroups) {
    auto config    = makeDSV4HybridPoolConfig();
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    // All groups share a single SharedBlockCache owned by the allocator.
    auto shared_cache = coordinator_manager->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Inserting a cache item for one group is visible via the shared cache.
    auto pool0  = coordinator_manager->group_block_pools_[0];
    auto blocks = pool0->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    std::vector<BlockIdxType> group_block_ids(coordinator_manager->group_block_pools_.size(), NULL_BLOCK_IDX);
    group_block_ids[0] = blocks[0];
    shared_cache->put(/*cache_key=*/42, group_block_ids, /*is_resident=*/false);
    EXPECT_TRUE(shared_cache->contains(42));

    // The same cache is returned by the allocator accessor.
    EXPECT_EQ(coordinator_manager->sharedBlockCache(), shared_cache);

    // Clean up.
    pool0->requestFree(blocks);
}

TEST_F(CoordinatorCacheManagerTest, DSV4CPShardedInsertThenReuseSamePrefix) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }
    CacheKeysType request_keys = full_keys;
    request_keys.push_back(2000);  // partial tail key present on the incoming request.

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    coordinator_manager->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    coordinator_manager->setCPSlotMapper(cp_mapper);
    ASSERT_TRUE(coordinator_manager->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    coordinator_manager->setCPSlotMapper(cp_mapper);
    coordinator_manager->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    coordinator_manager->free(seed_free);

    auto hit_res = makeBatchResource(/*batch_size=*/1, config);
    hit_res->setBatchCacheKeys(0, request_keys);
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.reuse_cache         = true;
    hit_malloc.enable_device_cache = true;
    coordinator_manager->setCPSlotMapper(cp_mapper);
    auto result = coordinator_manager->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    // 10 full keys under cp_size=2 subsample to 5 canonical match keys;
    // initMallocForCommonLen always drops the last canonical match key (it may
    // be a partial tail, and fully reusing the input would leave no prefill
    // tokens to compute), so only 4 canonical blocks are reusable.
    EXPECT_EQ(result.reuse_len, 4 * spb * 2);

    FreeInfo hit_free{hit_res, hit_tokens};
    coordinator_manager->free(hit_free);
}

TEST_F(CoordinatorCacheManagerTest, DSV4CPShardedEvictionMarksCanonicalResource) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto coordinator_manager = makeAllocator(config);
    ASSERT_TRUE(coordinator_manager->init());

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    coordinator_manager->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(coordinator_manager->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    coordinator_manager->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    coordinator_manager->free(seed_free);

    auto evicted = coordinator_manager->popBlocksFromCache(/*min_blocks_to_free=*/4);
    ASSERT_NE(evicted, nullptr);
    ASSERT_TRUE(evicted->hasCacheKeys());
    EXPECT_TRUE(evicted->cacheResource(0).cacheKeysAreCpCanonical());

    KVCacheResource canonical_source;
    canonical_source.setCacheKeys(full_keys);
    const auto expected_canonical = canonical_source.localCacheKeys(cp_mapper->cpSize() - 1, cp_mapper->cpSize());
    EXPECT_EQ(evicted->cacheKeys(0), expected_canonical);
    const auto& dependencies = evicted->cacheResource(0).blockDependencies();
    ASSERT_EQ(dependencies.size(), expected_canonical.size());
    for (size_t i = 0; i < dependencies.size(); ++i) {
        EXPECT_EQ(dependencies[i].ordinal, static_cast<uint32_t>(i));
        if (i == 0) {
            EXPECT_FALSE(dependencies[i].has_parent);
        } else {
            EXPECT_TRUE(dependencies[i].has_parent);
            EXPECT_EQ(dependencies[i].parent_key, expected_canonical[i - 1]);
        }
    }
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
