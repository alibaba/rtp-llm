#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolCoordinatorKVCacheManager.h"
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

constexpr char kLinearTag[] = "linear";
constexpr char kFullTag[]   = "full";
constexpr char kSwaTag[]    = "swa";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a tiny multi-pool config with two groups: gid=0 LINEAR(layers 0,1)
// and gid=1 FULL(layers 2,3). Each group has its own per-group block budget,
// so HybridPoolCoordinatorKVCacheManager creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t       linear_block_num = 6,
                                                 uint32_t       full_block_num   = 8,
                                                 CacheGroupType second_type      = CacheGroupType::FULL) {
    CacheConfig config;
    config.layer_num          = 4;
    config.seq_size_per_block = 4;
    config.linear_step        = 1;
    constexpr auto dtype      = rtp_llm::DataType::TYPE_FP16;

    auto linear_spec = makeResolvedLinearSpec(
        dtype, 1, 1, 1, 1, 2, static_cast<uint32_t>(config.seq_size_per_block), dtype, dtype, "linear");
    auto full_spec = makeResolvedMhaSpec(dtype, 1, 1, static_cast<uint32_t>(config.seq_size_per_block), "full");

    setTestTopology(config,
                    {makeTestGroupForConfig(config, linear_spec, {0, 1}, CacheGroupType::LINEAR, "linear"),
                     makeTestGroupForConfig(
                         config, full_spec, {2, 3}, second_type, second_type == CacheGroupType::SWA ? "swa" : "full")});

    const auto                 linear_stride   = linear_spec->block_size_bytes();
    const auto                 full_stride     = full_spec->block_size_bytes();
    const auto                 topology_groups = config.topology().groups();
    std::vector<GroupTopology> groups(topology_groups.begin(), topology_groups.end());
    for (auto& group : groups) {
        if (group.tag == "linear") {
            group.policy.explicit_block_num = linear_block_num;
            group.kv_block_stride_bytes     = linear_stride;
        } else {
            group.policy.explicit_block_num = full_block_num;
            group.kv_block_stride_bytes     = full_stride;
        }
        group.kv_scale_stride_bytes = 0;
    }
    config.setTopology(std::move(groups), config.topology().layers());
    return config;
}

static CacheConfig makeTinySwaMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t swa_block_num = 8) {
    return makeTinyMultiPoolHybridConfig(linear_block_num, swa_block_num, CacheGroupType::SWA);
}

static CacheConfig makeTinyDynamicMultiPoolHybridConfig(uint32_t       block_num   = 8,
                                                        CacheGroupType second_type = CacheGroupType::FULL) {
    auto                       config          = makeTinyMultiPoolHybridConfig(block_num, block_num, second_type);
    const auto                 topology_groups = config.topology().groups();
    std::vector<GroupTopology> groups(topology_groups.begin(), topology_groups.end());
    for (auto& group : groups) {
        group.policy.explicit_block_num = 0;
    }
    config.setTopology(std::move(groups), config.topology().layers());
    config.finalizeBlockNums(block_num, RuntimeConfig{});
    return config;
}

static ModelConfig makeTinyDSV4ModelConfig() {
    ModelConfig mc;
    mc.num_layers                                      = 5;
    mc.hidden_size                                     = 32;
    mc.attn_config.head_num                            = 4;
    mc.attn_config.kv_head_num                         = 1;
    mc.attn_config.size_per_head                       = 8;
    mc.attn_config.rope_head_dim                       = 4;
    mc.attn_config.indexer_head_dim                    = 8;
    mc.attn_config.indexer_head_num                    = 2;
    mc.attn_config.indexer_topk                        = 16;
    mc.attn_config.tokens_per_block                    = 128;
    mc.hybrid_attention_config.enable_hybrid_attention = true;
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
    auto mc                                            = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.finalizeBlockNums(block_num, RuntimeConfig{});
    return config;
}

static void setExplicitBlocksForGroup(CacheConfig& config, std::string_view tag, uint32_t block_num) {
    std::unordered_map<std::string, CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.emplace(group.tag, group.policy);
    }
    ASSERT_TRUE(policies.count(std::string(tag)));
    policies.at(std::string(tag)).explicit_block_num = block_num;
    config.setGroupPolicies(policies);
}

static std::string firstExplicitIndependentGroup(const CacheConfig& config) {
    for (const auto& group : config.topology().groups()) {
        if (group.policy.evict_policy == CacheEvictPolicy::INDEPENDENT && group.policy.explicit_block_num > 0) {
            return group.tag;
        }
    }
    ADD_FAILURE() << "missing explicit independent cache group";
    return {};
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

static void setGroupBlockNum(CacheConfig& config, std::string_view tag, uint32_t block_num) {
    const auto                 topology_groups = config.topology().groups();
    std::vector<GroupTopology> groups(topology_groups.begin(), topology_groups.end());
    bool                       found = false;
    for (auto& group : groups) {
        if (group.tag == tag) {
            group.policy.explicit_block_num = block_num;
            found                           = true;
        }
    }
    ASSERT_TRUE(found);
    config.setTopology(std::move(groups), config.topology().layers());
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

// Create HybridPoolCoordinatorKVCacheManager with SharedBlockCache injected (required before init()).
static HybridPoolCoordinatorKVCacheManagerPtr
makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION, int64_t reserve_block_ratio = 0) {
    auto allocator = std::make_shared<HybridPoolCoordinatorKVCacheManager>(
        config, AllocationType::DEVICE, nullptr, reserve_block_ratio, role_type);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    return allocator;
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
static BlockIdxType seedNonResidentCacheItem(const HybridPoolCoordinatorKVCacheManagerPtr& allocator,
                                             std::string_view                              tag,
                                             CacheKeyType                                  key) {
    auto pool   = allocator->groupBlockPools().at(std::string(tag));
    auto blocks = pool->malloc(1);
    EXPECT_EQ(blocks.size(), 1u);
    auto shared_cache = allocator->sharedBlockCache();
    shared_cache->put(key, {{std::string(tag), blocks[0]}}, false);
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

static std::unordered_map<std::string, PoolCounters>
snapshotPoolCounters(const HybridPoolCoordinatorKVCacheManagerPtr& allocator) {
    std::unordered_map<std::string, PoolCounters> counters;
    for (const auto& [tag, pool] : allocator->groupBlockPools()) {
        counters.emplace(tag,
                         PoolCounters{pool->freeBlocksNum(),
                                      pool->availableBlocksNum(),
                                      pool->requestRefBlocksNum(),
                                      pool->blockCacheRefBlocksNum(),
                                      pool->connectorRefBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const HybridPoolCoordinatorKVCacheManagerPtr&        allocator,
                                 const std::unordered_map<std::string, PoolCounters>& expected) {
    ASSERT_EQ(allocator->groupBlockPools().size(), expected.size());
    for (const auto& [tag, pool] : allocator->groupBlockPools()) {
        const auto& counters = expected.at(tag);
        EXPECT_EQ(pool->freeBlocksNum(), counters.free_blocks) << "tag=" << tag;
        EXPECT_EQ(pool->availableBlocksNum(), counters.available_blocks) << "tag=" << tag;
        EXPECT_EQ(pool->requestRefBlocksNum(), counters.request_refs) << "tag=" << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), counters.block_cache_refs) << "tag=" << tag;
        EXPECT_EQ(pool->connectorRefBlocksNum(), counters.connector_refs) << "tag=" << tag;
    }
}

class HybridPoolCoordinatorKVCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_NE(allocator->groupBlockPools().at(kLinearTag), allocator->groupBlockPools().at(kFullTag));

    // Per-pool totalBlocksNum = group_block_nums[gid] - 1 (block 0 reserved).
    EXPECT_EQ(allocator->groupBlockPools().at(kLinearTag)->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(allocator->groupBlockPools().at(kFullTag)->totalBlocksNum(), 8u - 1u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_EQ(allocator->groupBlockPools().at(kLinearTag)->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->groupBlockPools().at(kSwaTag)->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, GetBlockPoolRejectsUnknownTag) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getBlockPool("missing"), nullptr);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, TotalAndFreeBlocksAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const size_t expected_total = (6u - 1u) + (8u - 1u);
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
    EXPECT_EQ(allocator->availableBlocksNum(), expected_total);
    EXPECT_EQ(allocator->notInUseBlocksNum(), expected_total);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, TokenCountsUseSmallestPool) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->totalTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->maxSequenceLength(), 7u * 4u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, MaxSequenceLengthUsesCPVirtualBlockSizeForFullGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->totalTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->maxSequenceLength(), 7u * 4u);

    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(allocator->totalTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->maxSequenceLength(), 7u * 8u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, TokenCountsFallBackToGlobalSeqSize) {
    auto config               = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.seq_size_per_block = 4;
    auto allocator            = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->totalTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, RequestAndConnectorRefAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto pool0 = allocator->groupBlockPools().at(kLinearTag);
    auto pool1 = allocator->groupBlockPools().at(kFullTag);

    const size_t free_total_before = allocator->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2);
    auto         g1_blocks         = pool1->malloc(3);
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);

    EXPECT_EQ(allocator->requestRefBlocksNum(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 5u);
    EXPECT_EQ(allocator->availableBlocksNum(), free_total_before - 5u);

    // Mark some blocks as connector-referenced (simulating cache transfer).
    pool0->connectorReference(g0_blocks[0]);
    pool1->connectorReference(g1_blocks[0]);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 2u);

    pool0->requestFree(g0_blocks);
    pool1->requestFree(g1_blocks);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);

    // Connector still holds 2 blocks → freeBlocksNum (set of returnable
    // ids) drops by 2; notInUseBlocksNum counts blocks not held by *request*
    // or *block cache* refs, so connector-held blocks still count as "not
    // in use" → equals the full pool total.
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 2u);
    EXPECT_EQ(allocator->notInUseBlocksNum(), free_total_before);

    pool0->connectorFree(g0_blocks[0]);
    pool1->connectorFree(g1_blocks[0]);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before);
    EXPECT_EQ(allocator->notInUseBlocksNum(), free_total_before);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, BlockCacheRefAggregatesAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    seedNonResidentCacheItem(allocator, kLinearTag, /*key=*/100);
    seedNonResidentCacheItem(allocator, kFullTag, /*key=*/200);
    seedNonResidentCacheItem(allocator, kFullTag, /*key=*/201);

    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 3u);
}

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // Layer in linear group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/0, kLinearTag, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/0, kLinearTag, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
    // Layer in full group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/3, kFullTag, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/3, kFullTag, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ConvertIndexToBufferPartitionDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto bufs = allocator->convertIndexToBuffer(
        /*layer_id=*/3, kFullTag, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, HybridGroupsKeepWholeBlockTransferForAsymmetricTp) {
    auto config = makeTinyMultiPoolHybridConfig();

    const auto linear_pool_config = BlockPoolConfigHelper::createConfigForGroup(config, kLinearTag);
    const auto full_pool_config   = BlockPoolConfigHelper::createConfigForGroup(config, kFullTag);
    ASSERT_EQ(linear_pool_config.memory_layouts.size(), 1u);
    ASSERT_EQ(full_pool_config.memory_layouts.size(), 1u);
    EXPECT_TRUE(linear_pool_config.memory_layouts[0].enable_hybrid_attention);
    EXPECT_TRUE(full_pool_config.memory_layouts[0].enable_hybrid_attention);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto linear_buffers = allocator->convertIndexToBuffer(
        /*layer_id=*/0, kLinearTag, /*block_id=*/1, /*partition_count=*/2, /*partition_id=*/1);
    ASSERT_EQ(linear_buffers.size(), 1u);
    EXPECT_EQ(linear_buffers[0].size_bytes, config.kvBlockStrideBytesForGroup(kLinearTag));

    auto full_buffers = allocator->convertIndexToBuffer(
        /*layer_id=*/3, kFullTag, /*block_id=*/1, /*partition_count=*/2, /*partition_id=*/1);
    ASSERT_EQ(full_buffers.size(), 1u);
    EXPECT_EQ(full_buffers[0].size_bytes, config.kvBlockStrideBytesForGroup(kFullTag));
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto addr = allocator->convertIndexToAddr(/*layer_id=*/0, kLinearTag, /*block_id=*/1);
    EXPECT_NE(addr.kv_addr, nullptr);

    auto bufs_default = allocator->convertIndexToBuffer(/*layer_id=*/0, kLinearTag, /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = allocator->convertIndexToBuffer(
        /*layer_id=*/0, kLinearTag, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, AllLayerCacheBaseExposesPerLayerAndPerGroupTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    for (const auto& group : config.topology().groups()) {
        EXPECT_EQ(layout.topology().group(group.tag).policy.group_type, group.policy.group_type);
    }
    for (size_t i = 0; i < static_cast<size_t>(config.totalLayerNum()); ++i) {
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

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, RegUserMrWithoutCacheStoreIsNoOpAndZeroCost) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // No CacheStore is plumbed in: regUserMr should be a benign no-op for every
    // group pool, and the aggregated MR cost remains zero.
    EXPECT_NO_THROW(allocator->regUserMr(/*model_id=*/0, /*cache_store=*/nullptr));
    EXPECT_EQ(allocator->getMrCostTimeMs(), 0);
}

// ---------------------------------------------------------------------------
// popBlocksFromCache / blockCacheFree
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, PopBlocksFromCacheReturnsEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // Seed identical key on both groups, plus a unique key on the full group.
    auto g0_block_for_100 = seedNonResidentCacheItem(allocator, kLinearTag, /*key=*/100);
    auto g1_block_for_100 = seedNonResidentCacheItem(allocator, kFullTag, /*key=*/100);
    auto g1_block_for_200 = seedNonResidentCacheItem(allocator, kFullTag, /*key=*/200);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 3u);

    auto evicted = allocator->popBlocksFromCache(/*min_blocks_to_free=*/3);
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
    const auto& g0_blocks = evicted->blocks(0, kLinearTag);
    const auto& g1_blocks = evicted->blocks(0, kFullTag);
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

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, PopBlocksFromCacheZeroFreeReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->popBlocksFromCache(0), nullptr);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, PopBlocksFromCacheEmptyCachesReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->popBlocksFromCache(/*min_blocks_to_free=*/4), nullptr);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, BlockCacheFreeReleasesEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    seedNonResidentCacheItem(allocator, kLinearTag, /*key=*/100);
    seedNonResidentCacheItem(allocator, kFullTag, /*key=*/200);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 2u);

    const size_t free_before = allocator->freeBlocksNum();
    auto         evicted     = allocator->popBlocksFromCache(/*min_blocks_to_free=*/2);
    ASSERT_NE(evicted, nullptr);
    // Eviction releases the LRU entries from SharedBlockCache; the underlying blocks
    // are still referenced by blockCacheRef. Releasing those refs is what
    // blockCacheFree() does.
    allocator->blockCacheFree(evicted);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 2u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, BlockCacheFreeNullPtrIsNoOp) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_NO_THROW(allocator->blockCacheFree(nullptr));
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, BlockCacheFreeIgnoresDuplicateAndNullBlockIds) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto seeded = seedNonResidentCacheItem(allocator, kFullTag, /*key=*/300);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 1u);

    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(config.topologyPtr());
    // Same block listed twice in the same group should only be released once;
    // NULL_BLOCK_IDX entries should be skipped.
    batch->mutableBlockIds(0, kFullTag).assign(BlockIndicesType{seeded, seeded, NULL_BLOCK_IDX});
    EXPECT_NO_THROW(allocator->blockCacheFree(batch));
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Both dynamic groups get global N=6 (5 allocatable blocks each). A reserve
    // of 4 is split evenly, leaving room for one requested block per group.
    auto config    = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    allocator->setReserveBlocksNum(4);

    // seq_len=4 -> 1 block per group.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_TRUE(result.success);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    // Force a group whose available_blocks < need + group_reserve_blocks.
    auto config    = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // A reserve large enough to hide most blocks should reject init malloc.
    allocator->setReserveBlocksNum(allocator->availableBlocksNum());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    constexpr size_t reserve_blocks = 6;
    allocator->setReserveBlocksNum(reserve_blocks);

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    std::unordered_map<std::string, KVCachePoolMetricsSnapshot> snapshots_by_name;
    for (const auto& snapshot : snapshots) {
        snapshots_by_name.emplace(snapshot.pool_name, snapshot);
    }
    const auto& linear = snapshots_by_name.at("linear");
    const auto& full   = snapshots_by_name.at("full");

    const size_t total_reservable_available_blocks = linear.available_blocks + full.available_blocks;
    ASSERT_GT(total_reservable_available_blocks, 0u);
    EXPECT_EQ(reserve_blocks * linear.available_blocks / total_reservable_available_blocks, linear.reserve_blocks);
    EXPECT_EQ(reserve_blocks * full.available_blocks / total_reservable_available_blocks, full.reserve_blocks);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/20);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    allocator->setReserveBlocksNum(2);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;

    auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, kFullTag)), 4u);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ReserveCheckIsBypassedWhenMallocInfoLacksContext) {
    // hasAvailableBlocksForReserve returns true when info has no resource/tokens.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    MallocInfo info{};
    EXPECT_TRUE(allocator->hasAvailableBlocksForReserve(info, /*reserve_blocks=*/9999));
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, InitMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    // gid=0 has enough room for the LINEAR tail block; gid=1 cannot satisfy
    // the 3 FULL blocks needed for seq_len=9. initMallocForCommonLen should
    // roll gid=0 back after gid=1 fails.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/3, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto counters_before = snapshotPoolCounters(allocator);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, kLinearTag), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, kFullTag), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto linear_cached = seedNonResidentCacheItem(allocator, kLinearTag, /*key=*/100);
    const auto full_cached   = seedNonResidentCacheItem(allocator, kFullTag, /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    ASSERT_EQ(allocator->requestRefBlocksNum(), 0u);
    ASSERT_EQ(allocator->blockCacheRefBlocksNum(), 2u);

    const size_t available_before = allocator->availableBlocksNum();
    const auto   counters_before  = snapshotPoolCounters(allocator);
    allocator->setReserveBlocksNum(std::max<size_t>(1, available_before * 8));

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, kLinearTag), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, kFullTag), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 2u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/2);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 1u);
    const auto linear_block_before = batch_res->blocks(0, kLinearTag)[0];
    const auto full_block_before   = batch_res->blocks(0, kFullTag)[0];
    const auto counters_before     = snapshotPoolCounters(allocator);

    // gid=0 can append one real LINEAR tail block. gid=1 has no remaining
    // free blocks and no cache to evict, so FULL allocation fails.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 1u);
    EXPECT_EQ(batch_res->blocks(0, kLinearTag)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, kFullTag)[0], full_block_before);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, IncrMallocRollbackRestoresLinearBackfilledSlots) {
    // Block 0 is reserved by each pool, so FULL needs three configured blocks
    // to provide the two request blocks used by the initial allocation.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 2u);

    auto& linear_ids       = batch_res->mutableBlockIds(0, kLinearTag);
    auto  removed_block_id = linear_ids.blocks()[1];
    ASSERT_FALSE(isNullBlockIdx(removed_block_id));
    allocator->groupBlockPools().at(kLinearTag)->requestFree({removed_block_id});
    linear_ids.setAt(1, NULL_BLOCK_IDX);
    const auto counters_before = snapshotPoolCounters(allocator);

    // LINEAR first backfills the old sparse tail and appends a new tail block.
    // FULL then fails because its independent pool is exhausted. Rollback must
    // restore both the historical NULL slot and the original logical length.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    EXPECT_FALSE(allocator->malloc(incr_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 2u);
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, kLinearTag)[1]));
    expectPoolCountersEq(allocator, counters_before);
}

// ---------------------------------------------------------------------------
// Full malloc / free cycle
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, MallocAndFreeCycleAcrossPerGroupPools) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/8, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const size_t free_before = allocator->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_LT(allocator->freeBlocksNum(), free_before);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ---------------------------------------------------------------------------
// DSV4 7-group HybridPool: covers per-tag addressing and SWA tail
// ---------------------------------------------------------------------------

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4InitAndAggregatedCounters) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);

    // Sum of per-pool totals must equal aggregated totalBlocksNum.
    size_t expected_total = 0;
    for (const auto& [tag, pool] : allocator->groupBlockPools()) {
        (void)tag;
        expected_total += pool->totalBlocksNum();
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
    EXPECT_EQ(allocator->availableBlocksNum(), expected_total);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    for (const auto& [tag, pool] : allocator->groupBlockPools()) {
        EXPECT_EQ(pool->where(), MemoryType::MEMORY_GPU) << "tag=" << tag;
    }
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator     = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(config.group("hca_state").tag, "hca_state");
    const auto   hca_state_pool  = allocator->groupBlockPools().at("hca_state");
    const size_t hca_free_before = hca_state_pool->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, "hca_state");
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - hca_state_pool->freeBlocksNum(), 1u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, TokenCountsIncludeSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    ASSERT_EQ(config.group("hca_state").tag, "hca_state");
    setGroupBlockNum(config, "hca_state", 2);
    config.finalizeBlockNums(/*global_block_num=*/50, RuntimeConfig{});

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto hca_state_tokens =
        allocator->groupBlockPools().at("hca_state")->totalBlocksNum() * config.seq_size_per_block;
    EXPECT_EQ(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->totalTokensNum());
    EXPECT_GT(allocator->maxSequenceLength(), allocator->totalTokensNum());
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4ConfigUsesGroupOwnedBytesForPagedBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_paged_bytes = 0;
    size_t expected_paged_bytes     = 0;
    for (const auto& group : config.topology().groups()) {
        const auto& tag  = group.tag;
        const auto  type = config.typeForGroup(tag);
        const auto  expected_group_bytes =
            config.layerIdsForGroup(tag).size()
            * (config.kvBlockStrideBytesForGroup(tag) + config.kvScaleStrideBytesForGroup(tag));
        EXPECT_EQ(config.blockSizeBytesForGroup(tag), expected_group_bytes) << "tag=" << tag;
        if (!config.usesExplicitIndependentBlocks(tag)
            && (type == CacheGroupType::FULL || type == CacheGroupType::LINEAR)) {
            expected_paged_bytes += expected_group_bytes;
        } else {
            expected_non_paged_bytes += expected_group_bytes;
        }
    }

    EXPECT_GT(expected_non_paged_bytes, 0u);
    EXPECT_GT(expected_paged_bytes, 0u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, ReserveRatioExcludesExplicitIndependentPools) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    ASSERT_FALSE(firstExplicitIndependentGroup(config).empty());

    constexpr int64_t reserve_ratio = 10;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());

    size_t reservable_available = 0;
    size_t all_available        = 0;
    for (const auto& [tag, pool] : allocator->groupBlockPools()) {
        const size_t available = pool->availableBlocksNum();
        all_available += available;
        if (!config.usesExplicitIndependentBlocks(tag)) {
            reservable_available += available;
        }
    }
    ASSERT_GT(reservable_available, 0u);
    ASSERT_GT(all_available, reservable_available);
    EXPECT_EQ(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * reservable_available / static_cast<size_t>(100));
    EXPECT_NE(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * all_available / static_cast<size_t>(100));
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4FinalizeBlockNumsUsesHcaStatePoolBlocks) {
    auto       config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const auto explicit_tag = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_tag, 50);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (const auto& group : config.topology().groups()) {
        const uint32_t expected = config.policyForGroup(group.tag).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(group.tag), expected) << "tag=" << group.tag;
    }

    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_tag);
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), expected_reserve);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4ExplicitReserveBytesAreReportedPerGroup) {
    auto       config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const auto explicit_tag = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_tag, 50);

    size_t sum = 0;
    for (const auto& group : config.topology().groups()) {
        const size_t expected =
            group.tag == explicit_tag ? 50u * config.blockSizeBytesForGroup(group.tag) : static_cast<size_t>(0);
        EXPECT_EQ(config.explicitReserveBytesForGroup(group.tag), expected) << "tag=" << group.tag;
        sum += config.explicitReserveBytesForGroup(group.tag);
    }
    EXPECT_EQ(sum, config.explicitlySizedPoolReserveBytes());
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);
    setExplicitBlocksForGroup(config, firstExplicitIndependentGroup(config), 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (const auto& group : config.topology().groups()) {
        EXPECT_EQ(config.blockNumForGroup(group.tag), 123u);
    }
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), 0u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto       config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const auto explicit_tag = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_tag, 50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (const auto& group : config.topology().groups()) {
        const uint32_t expected = config.policyForGroup(group.tag).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(group.tag), expected) << "tag=" << group.tag;
    }
    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_tag);
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), expected_reserve);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4StateSwaPoolsWithoutExplicitBlocksUseGlobalBlockNum) {
    auto mc                                            = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    ParallelismConfig pc;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.linear_step = 1;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (const auto& group : config.topology().groups()) {
        EXPECT_EQ(config.blockNumForGroup(group.tag), 128u) << "tag=" << group.tag;
    }
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), 0u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, FinalizeRejectsNonUnitLinearStep) {
    auto config        = makeTinyDynamicMultiPoolHybridConfig(/*block_num=*/8, CacheGroupType::SWA);
    config.linear_step = 2;
    RuntimeConfig rt;

    EXPECT_THROW(config.finalizeBlockNums(/*global_block_num=*/9, rt), std::exception);

    config.linear_step = 1;
    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.blockNumForGroup(kLinearTag), 9u);
    EXPECT_EQ(config.blockNumForGroup(kSwaTag), 9u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4ConvertIndexToAddrRoutesToCorrectGroupPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.totalLayerNum(); ++l) {
        for (const auto& group : config.groupsForLayer(static_cast<int>(l))) {
            if (group.get().tag == "csa_kv") {
                csa_layer = static_cast<int>(l);
                break;
            }
        }
        if (csa_layer >= 0) {
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // csa_kv tag routes to the CSA group's pool.
    auto addr_csa = allocator->convertIndexToAddr(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);

    auto addr_swa = allocator->convertIndexToAddr(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, "missing", 1), std::exception);
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, "hca_kv", 1), std::exception);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4ConvertIndexToBufferAcrossGroupsAndPartitions) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    int csa_layer = -1;
    for (size_t l = 0; l < config.totalLayerNum(); ++l) {
        for (const auto& group : config.groupsForLayer(static_cast<int>(l))) {
            if (group.get().tag == "csa_kv") {
                csa_layer = static_cast<int>(l);
                break;
            }
        }
        if (csa_layer >= 0) {
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    auto buf = allocator->convertIndexToBuffer(csa_layer, "csa_kv", /*block_id=*/1);
    ASSERT_FALSE(buf.empty());
    EXPECT_NE(buf[0].addr, nullptr);

    auto buf_part =
        allocator->convertIndexToBuffer(csa_layer, "csa_kv", /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(buf_part.empty());
    EXPECT_NE(buf_part[0].addr, nullptr);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4AllLayerCacheBaseHasPerGroupTensors) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    for (size_t l = 0; l < static_cast<size_t>(config.totalLayerNum()); ++l) {
        EXPECT_TRUE(layout.group("swa_kv").hasLayer(l)) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.groups().size(), 7u);
    EXPECT_EQ(layout.topology().groups().size(), 7u);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4SharedBlockCacheIsUnifiedAcrossGroups) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // All groups share a single SharedBlockCache owned by the allocator.
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Inserting a cache item for one group is visible via the shared cache.
    auto pool0  = allocator->groupBlockPools().at("csa_kv");
    auto blocks = pool0->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    shared_cache->put(
        /*cache_key=*/42,
        {{"csa_kv", blocks[0]}},
        /*is_resident=*/false);
    EXPECT_TRUE(shared_cache->contains(42));

    // The same cache is returned by the allocator accessor.
    EXPECT_EQ(allocator->sharedBlockCache(), shared_cache);

    // Clean up.
    pool0->requestFree(blocks);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4CPShardedInsertThenReuseSamePrefix) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }
    CacheKeysType request_keys = full_keys;
    request_keys.push_back(2000);  // partial tail key present on the incoming request.

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    allocator->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    allocator->setCPSlotMapper(cp_mapper);
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    allocator->setCPSlotMapper(cp_mapper);
    allocator->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    auto hit_res = makeBatchResource(/*batch_size=*/1, config);
    hit_res->setBatchCacheKeys(0, request_keys);
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.reuse_cache         = true;
    hit_malloc.enable_device_cache = true;
    allocator->setCPSlotMapper(cp_mapper);
    auto result = allocator->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 5 * spb * 2);

    FreeInfo hit_free{hit_res, hit_tokens};
    allocator->free(hit_free);
}

TEST_F(HybridPoolCoordinatorKVCacheManagerTest, DSV4CPShardedEvictionMarksCanonicalResource) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    allocator->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    auto evicted = allocator->popBlocksFromCache(/*min_blocks_to_free=*/4);
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
