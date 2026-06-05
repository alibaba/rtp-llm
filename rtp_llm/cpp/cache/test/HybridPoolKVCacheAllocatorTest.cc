#include <gtest/gtest.h>

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
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
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
// so HybridPoolKVCacheAllocator creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t full_block_num = 8) {
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = std::max(linear_block_num, full_block_num);
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 4;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
    linear_spec->layer_num          = 2;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->layer_num          = 2;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.layer_ids                   = {{0, 1}, {2, 3}};
    config.global_layer_ids            = config.layer_ids;
    config.cache_specs                 = {linear_spec, full_spec};
    config.group_types                 = {CacheGroupType::LINEAR, CacheGroupType::FULL};
    config.group_region_names          = {KVCacheRegionName::DEFAULT, KVCacheRegionName::DEFAULT};
    config.linear_group_num            = 1;
    config.full_group_num              = 1;
    config.use_independent_block_pools = true;

    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    config.layer_to_group_ids.assign(static_cast<size_t>(config.layer_num), {});
    config.layer_region_to_group_id.assign(static_cast<size_t>(config.layer_num),
                                           std::vector<int>(static_cast<size_t>(KVCacheRegionName::REGION_COUNT), -1));
    config.layer_group_types.assign(static_cast<size_t>(config.layer_num), CacheGroupType::FULL);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(layer_id)]  = static_cast<int>(gid);
            config.layer_to_group_ids[static_cast<size_t>(layer_id)] = {static_cast<int>(gid)};
            config.layer_region_to_group_id[static_cast<size_t>(layer_id)]
                                           [static_cast<size_t>(KVCacheRegionName::DEFAULT)] = static_cast<int>(gid);
            config.layer_group_types[static_cast<size_t>(layer_id)] =
                (gid == 0) ? CacheGroupType::LINEAR : CacheGroupType::FULL;
        }
    }

    // Per-group block counts: gid=0 -> linear_block_num, gid=1 -> full_block_num.
    config.group_block_nums = {linear_block_num, full_block_num};
    // Same tokens per block for both groups.
    config.group_seq_size_per_block = {config.seq_size_per_block, config.seq_size_per_block};

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(config.kv_block_stride_bytes));
    return config;
}

static CacheConfig makeTinySwaMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t swa_block_num = 8) {
    auto config = makeTinyMultiPoolHybridConfig(linear_block_num, swa_block_num);

    config.group_types      = {CacheGroupType::LINEAR, CacheGroupType::SWA};
    config.linear_group_num = 1;
    config.full_group_num   = 0;
    config.swa_group_num    = 1;
    for (int layer_id : config.layer_ids[1]) {
        config.layer_group_types[static_cast<size_t>(layer_id)] = CacheGroupType::SWA;
    }
    return config;
}

static ModelConfig makeTinyDSV4ModelConfig() {
    ModelConfig mc;
    mc.num_layers                        = 5;
    mc.hidden_size                       = 32;
    mc.attn_config.head_num              = 4;
    mc.attn_config.kv_head_num           = 1;
    mc.attn_config.size_per_head         = 8;
    mc.attn_config.rope_head_dim         = 4;
    mc.attn_config.sliding_window        = 128;
    mc.attn_config.indexer_head_dim      = 8;
    mc.attn_config.indexer_head_num      = 2;
    mc.attn_config.indexer_topk          = 16;
    mc.attn_config.o_groups              = 2;
    mc.attn_config.o_lora_rank           = 16;
    mc.attn_config.layer_compress_ratios = {4, 128, 4, 128, 0};
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
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

// Build a DSV4 7-pool CacheConfig (uses use_independent_block_pools=true).
static CacheConfig makeDSV4HybridPoolConfig(uint32_t block_num = 200) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block     = 128;
    kv_cache_config.dsv4_fixed_pool_blocks = block_num;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config, false, 0);
    config.block_num         = block_num;
    return config;
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
    res->initGroups(config.groupNums(), static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    return res;
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) {
        return !isNullBlockIdx(block);
    }));
}

// Create HybridPoolKVCacheAllocator with SharedBlockCache injected (required before init()).
static HybridPoolKVCacheAllocatorPtr makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION) {
    auto allocator =
        std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE, nullptr, 0, role_type);
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
static BlockIdxType
seedNonResidentCacheItem(const HybridPoolKVCacheAllocatorPtr& allocator, int gid, CacheKeyType key) {
    auto pool   = allocator->groupBlockPools()[static_cast<size_t>(gid)];
    auto blocks = pool->malloc(1);
    EXPECT_EQ(blocks.size(), 1u);
    auto                      shared_cache = allocator->sharedBlockCache();
    std::vector<BlockIdxType> group_slots(allocator->groupBlockPools().size(), NULL_BLOCK_IDX);
    group_slots[static_cast<size_t>(gid)] = blocks[0];
    shared_cache->put(key, group_slots, false);
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

static std::vector<PoolCounters> snapshotPoolCounters(const HybridPoolKVCacheAllocatorPtr& allocator) {
    std::vector<PoolCounters> counters;
    counters.reserve(allocator->groupBlockPools().size());
    for (const auto& pool : allocator->groupBlockPools()) {
        counters.push_back({pool->freeBlocksNum(),
                            pool->availableBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            pool->connectorRefBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const HybridPoolKVCacheAllocatorPtr& allocator,
                                 const std::vector<PoolCounters>&     expected) {
    ASSERT_EQ(allocator->groupBlockPools().size(), expected.size());
    for (size_t gid = 0; gid < expected.size(); ++gid) {
        const auto& pool = allocator->groupBlockPools()[gid];
        EXPECT_EQ(pool->freeBlocksNum(), expected[gid].free_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->availableBlocksNum(), expected[gid].available_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->requestRefBlocksNum(), expected[gid].request_refs) << "gid=" << gid;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), expected[gid].block_cache_refs) << "gid=" << gid;
        EXPECT_EQ(pool->connectorRefBlocksNum(), expected[gid].connector_refs) << "gid=" << gid;
    }
}

class HybridPoolKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_NE(allocator->groupBlockPools()[0], allocator->groupBlockPools()[1]);

    // Per-pool totalBlocksNum = group_block_nums[gid] - 1 (block 0 reserved).
    EXPECT_EQ(allocator->groupBlockPools()[0]->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(allocator->groupBlockPools()[1]->totalBlocksNum(), 8u - 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_EQ(allocator->groupBlockPools()[0]->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->groupBlockPools()[1]->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolKVCacheAllocatorTest, GetBlockPoolReturnsFirstGroupPool) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getBlockPool(), allocator->groupBlockPools()[0]);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, TotalAndFreeBlocksAggregateAcrossGroups) {
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

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseDifferentCapacityScopes) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    // Group 0 (LINEAR): seq_size_per_block=2 -> 5 blocks * 2 = 10
    // Group 1 (FULL):   seq_size_per_block=4 -> 7 blocks * 4 = 28
    config.group_seq_size_per_block = {2, 4};
    auto allocator                  = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 28u);
    EXPECT_EQ(allocator->availableTokensNum(), 10u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsFallBackToGlobalSeqSize) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.group_seq_size_per_block.clear();  // fall back to config.seq_size_per_block
    config.seq_size_per_block = 4;
    auto allocator            = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, RequestAndConnectorRefAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];

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

TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheRefAggregatesAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/201);

    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 3u);
}

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // Layer in linear group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
    // Layer in full group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/3, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToBufferPartitionDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto bufs = allocator->convertIndexToBuffer(
        /*layer_id=*/3, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByRegion) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // DEFAULT region routes through the per-layer default group.
    auto addr_default   = allocator->convertIndexToAddr(/*layer_id=*/0, KVCacheRegionName::DEFAULT, /*block_id=*/1);
    auto addr_via_layer = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_via_layer.kv_addr);

    auto bufs_default = allocator->convertIndexToBuffer(/*layer_id=*/0, KVCacheRegionName::DEFAULT, /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = allocator->convertIndexToBuffer(
        /*layer_id=*/0, KVCacheRegionName::DEFAULT, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, AllLayerCacheBaseExposesPerLayerAndPerRegionTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "layer " << i << " missing kv buffer";
    }
    EXPECT_EQ(layout.layer_to_groups, config.layer_to_group_id);
    EXPECT_EQ(layout.group_types, config.group_types);

    // Per-region matrix dims: layer_all_num x REGION_COUNT.
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs_by_attn.size(); ++i) {
        EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn[i].size(),
                  static_cast<size_t>(KVCacheRegionName::REGION_COUNT));
    }

    // Each layer's DEFAULT region tensor must be defined and identical to its primary kv tensor.
    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
        const auto& by_default =
            layout.layers_to_kv_buffer_ptrs_by_attn[i][static_cast<size_t>(KVCacheRegionName::DEFAULT)];
        EXPECT_TRUE(by_default.defined()) << "layer " << i << " DEFAULT tensor undefined";
        EXPECT_EQ(by_default.data_ptr(), layout.layers_to_kv_buffer_ptrs[i].data_ptr());
    }
}

// ---------------------------------------------------------------------------
// regUserMr / getMrCostTimeMs
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, RegUserMrWithoutCacheStoreIsNoOpAndZeroCost) {
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

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheReturnsEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // Seed identical key on both groups, plus a unique key on the full group.
    auto g0_block_for_100 = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    auto g1_block_for_100 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/100);
    auto g1_block_for_200 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 3u);

    auto evicted = allocator->popBlocksFromCache(/*min_blocks_to_free=*/3);
    ASSERT_NE(evicted, nullptr);
    EXPECT_EQ(evicted->batchSize(), 1);
    EXPECT_EQ(evicted->groupNums(), 2);
    const auto& keys = evicted->cacheKeys(0);
    EXPECT_EQ(keys.size(), 2u);  // 100 (shared) + 200 (g1 only)

    std::unordered_set<CacheKeyType> key_set(keys.begin(), keys.end());
    EXPECT_TRUE(key_set.count(100));
    EXPECT_TRUE(key_set.count(200));

    // Per-group block ids: each group's blocks should be set only at the slot
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

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheZeroFreeReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->popBlocksFromCache(0), nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheEmptyCachesReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->popBlocksFromCache(/*min_blocks_to_free=*/4), nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeReleasesEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 2u);

    const size_t free_before = allocator->freeBlocksNum();
    auto         evicted     = allocator->popBlocksFromCache(/*min_blocks_to_free=*/2);
    ASSERT_NE(evicted, nullptr);
    // Eviction releases the LRU entries from BlockCache; the underlying blocks
    // are still referenced by blockCacheRef. Releasing those refs is what
    // blockCacheFree() does.
    allocator->blockCacheFree(evicted);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 2u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeNullPtrIsNoOp) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_NO_THROW(allocator->blockCacheFree(nullptr));
}

TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeIgnoresDuplicateAndNullBlockIds) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto seeded = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/300);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 1u);

    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(config.groupNums(), static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    // Same block listed twice in the same group should only be released once;
    // NULL_BLOCK_IDX entries should be skipped.
    batch->mutableBlockIds(0, /*gid=*/1).assign(BlockIndicesType{seeded, seeded, NULL_BLOCK_IDX});
    EXPECT_NO_THROW(allocator->blockCacheFree(batch));
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Group 0 (linear) gets 6 blocks (5 free), group 1 (full) gets 4 blocks (3 free).
    // total_available = 8. Set reserve = 4.
    // Expected per-group reserve: floor(4 * 5/8) = 2 for gid=0, floor(4 * 3/8) = 1 for gid=1.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    allocator->setReserveBlockNum(4);

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

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    // Force a group whose available_blocks < need + group_reserve_blocks.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // A reserve large enough to hide most blocks should reject init malloc.
    allocator->setReserveBlockNum(allocator->availableBlocksNum());

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

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveCheckIsBypassedWhenMallocInfoLacksContext) {
    // hasAvailableBlocksForReserve returns true when info has no resource/tokens.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    MallocInfo info{};
    EXPECT_TRUE(allocator->hasAvailableBlocksForReserve(info, /*reserve_blocks=*/9999));
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
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
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto linear_cached = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    const auto full_cached   = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    ASSERT_EQ(allocator->requestRefBlocksNum(), 0u);
    ASSERT_EQ(allocator->blockCacheRefBlocksNum(), 2u);

    const size_t available_before = allocator->availableBlocksNum();
    const auto   counters_before  = snapshotPoolCounters(allocator);
    allocator->setReserveBlockNum(std::max<size_t>(1, available_before * 8));

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
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 2u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
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

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];
    const auto counters_before     = snapshotPoolCounters(allocator);

    // gid=0 can append one real LINEAR tail block. gid=1 has no remaining
    // free blocks and no cache to evict, so FULL allocation fails.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);
    expectPoolCountersEq(allocator, counters_before);
}

// ---------------------------------------------------------------------------
// Full malloc / free cycle
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, MallocAndFreeCycleAcrossPerGroupPools) {
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
// DSV4 7-group HybridPool: covers per-region addressing and SWA tail
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4InitAndAggregatedCounters) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);

    // Sum of per-pool totals must equal aggregated totalBlocksNum.
    size_t expected_total = 0;
    for (const auto& pool : allocator->groupBlockPools()) {
        expected_total += pool->totalBlocksNum();
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
    EXPECT_EQ(allocator->availableBlocksNum(), expected_total);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedRegionPoolsUsePinnedCpuBackingWhenToggled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    // DSV4_FIXED_POOL_USE_MEMORY path → CacheConfigCreator would set this true.
    config.fixed_pool_uses_pinned_cpu = true;
    auto allocator                    = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(config.group_region_names.size(), 7u);
    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);

    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        const auto region_name = config.group_region_names[gid];
        EXPECT_EQ(allocator->groupBlockPools()[gid]->where(),
                  isDsv4FixedRegion(region_name) ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_GPU)
            << "gid=" << gid << " region=" << static_cast<int>(region_name);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedRegionPoolsOnGpuWhenFixedPoolMemoryDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    // Default (env=0): fixed_pool_uses_pinned_cpu == false → all 7 pools on GPU.
    ASSERT_FALSE(config.fixed_pool_uses_pinned_cpu);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        EXPECT_EQ(allocator->groupBlockPools()[gid]->where(), MemoryType::MEMORY_GPU)
            << "gid=" << gid << " region=" << static_cast<int>(config.group_region_names[gid]);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config       = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator    = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    constexpr int hca_state_gid = 5;
    ASSERT_GT(config.group_region_names.size(), static_cast<size_t>(hca_state_gid));
    ASSERT_EQ(config.group_region_names[hca_state_gid], KVCacheRegionName::HCA_STATE);
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_gid));

    const size_t hca_free_before = allocator->groupBlockPools()[hca_state_gid]->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, hca_state_gid);
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - allocator->groupBlockPools()[hca_state_gid]->freeBlocksNum(), 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigSplitsStateBytesOutOfSwaAccumulator) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block     = 128;
    kv_cache_config.dsv4_fixed_pool_blocks = 200;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config, false, 0);

    ASSERT_EQ(config.groupNums(), 7);
    ASSERT_EQ(config.group_region_names.size(), 7u);
    ASSERT_EQ(config.group_block_size_bytes.size(), 7u);

    size_t expected_state_bytes = 0;
    size_t expected_swa_bytes   = 0;
    size_t expected_full_bytes  = 0;
    for (size_t gid = 0; gid < config.group_region_names.size(); ++gid) {
        const auto region = config.group_region_names[gid];
        const auto type   = config.group_types[gid];
        if (isStateRegion(region)) {
            expected_state_bytes += config.group_block_size_bytes[gid];
        } else if (type == CacheGroupType::SWA) {
            expected_swa_bytes += config.group_block_size_bytes[gid];
        } else {
            expected_full_bytes += config.group_block_size_bytes[gid];
        }
    }

    EXPECT_GT(expected_state_bytes, 0u);
    EXPECT_GT(expected_swa_bytes, 0u);
    EXPECT_GT(expected_full_bytes, 0u);

    EXPECT_EQ(config.state_block_size_bytes, expected_state_bytes);
    EXPECT_EQ(config.swa_block_size_bytes, expected_swa_bytes);  // SWA_KV only, no STATE
    EXPECT_EQ(config.block_size_bytes, expected_full_bytes);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesFixedPoolBlocks) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);
    config.fixed_pool_uses_pinned_cpu = true;

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    ASSERT_EQ(config.group_block_nums.size(), config.group_region_names.size());
    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
        const auto type = config.group_types[gid];
        if (type == CacheGroupType::FULL) {
            EXPECT_EQ(config.group_block_nums[gid], 200u) << "gid=" << gid;  // no addition for FULL
        } else {
            EXPECT_EQ(config.group_block_nums[gid], 50u) << "gid=" << gid;
        }
    }

    // CPU-pinned fixed regions must NOT be charged to the HBM fixed-pool reserve.
    size_t expected_reserve = 0;
    for (size_t gid = 0; gid < config.group_block_size_bytes.size(); ++gid) {
        const auto region = config.group_region_names[gid];
        const auto type   = config.group_types[gid];
        if (type != CacheGroupType::FULL && !isDsv4FixedRegion(region)) {
            expected_reserve += static_cast<size_t>(config.dsv4_fixed_pool_blocks) * config.group_block_size_bytes[gid];
        }
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesConfiguredFixedBlocks) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
        const auto type = config.group_types[gid];
        if (type == CacheGroupType::FULL) {
            EXPECT_EQ(config.group_block_nums[gid], 123u);
        } else {
            EXPECT_EQ(config.group_block_nums[gid], 123u);
        }
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4PinnedFixedPoolExcludesFixedReserve) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);
    config.fixed_pool_uses_pinned_cpu = true;  // env>0 simulation

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    size_t expected_reserve = 0;
    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
        const auto region = config.group_region_names[gid];
        const auto type   = config.group_types[gid];
        if (type != CacheGroupType::FULL && !isDsv4FixedRegion(region)) {
            expected_reserve += static_cast<size_t>(config.dsv4_fixed_pool_blocks) * config.group_block_size_bytes[gid];
        }
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4GpuFixedPoolIncludesFixedReserve) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    size_t expected_reserve = 0;
    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
        if (config.group_types[gid] != CacheGroupType::FULL) {
            EXPECT_EQ(config.group_block_nums[gid], 50u) << "gid=" << gid;
            expected_reserve += static_cast<size_t>(config.dsv4_fixed_pool_blocks) * config.group_block_size_bytes[gid];
        }
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedPoolBlocksFallbackFollowsLinearStep) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block = 128;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config, false, 0);
    config.linear_step       = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    ASSERT_EQ(config.group_block_nums.size(), config.group_region_names.size());
    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
        if (config.group_types[gid] == CacheGroupType::FULL) {
            EXPECT_EQ(config.group_block_nums[gid], 128u) << "gid=" << gid;
        } else {
            EXPECT_EQ(config.group_block_nums[gid], 32u) << "gid=" << gid;
        }
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByRegionRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layer_region_to_group_id[l][static_cast<size_t>(KVCacheRegionName::CSA_KV)] >= 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // CSA_KV region routes to gid=0; it must produce a non-null kv address that
    // matches the CSA group's pool.
    auto addr_csa = allocator->convertIndexToAddr(csa_layer, KVCacheRegionName::CSA_KV, /*block_id=*/1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);

    // SWA_KV region (group 6) is mapped for every layer, including csa_layer.
    auto addr_swa = allocator->convertIndexToAddr(csa_layer, KVCacheRegionName::SWA_KV, /*block_id=*/1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two regions live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);

    // DEFAULT region for csa_layer follows layer_to_group_id (=6, SWA pool).
    auto addr_default = allocator->convertIndexToAddr(csa_layer, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_swa.kv_addr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByRegionAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layer_region_to_group_id[l][static_cast<size_t>(KVCacheRegionName::CSA_KV)] >= 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    auto buf = allocator->convertIndexToBuffer(csa_layer, KVCacheRegionName::CSA_KV, /*block_id=*/1);
    ASSERT_FALSE(buf.empty());
    EXPECT_NE(buf[0].addr, nullptr);

    auto buf_part = allocator->convertIndexToBuffer(csa_layer,
                                                    KVCacheRegionName::CSA_KV,
                                                    /*block_id=*/1,
                                                    /*partition_count=*/1,
                                                    /*partition_id=*/0);
    ASSERT_FALSE(buf_part.empty());
    EXPECT_NE(buf_part[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4AllLayerCacheBaseHasPerRegionTensors) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn.size(), static_cast<size_t>(config.layer_all_num));

    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num); ++l) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[l].defined());
        // Every layer must have a SWA_KV region tensor (group 6 covers all layers).
        const auto& swa_t = layout.layers_to_kv_buffer_ptrs_by_attn[l][static_cast<size_t>(KVCacheRegionName::SWA_KV)];
        EXPECT_TRUE(swa_t.defined()) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.group_region_names.size(), 7u);
    EXPECT_EQ(layout.group_types.size(), 7u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4SharedBlockCacheIsUnifiedAcrossGroups) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // All groups share a single SharedBlockCache owned by the allocator.
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Inserting a cache item for one group is visible via the shared cache.
    auto pool0  = allocator->groupBlockPools()[0];
    auto blocks = pool0->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    std::vector<BlockIdxType> group_slots(allocator->groupBlockPools().size(), NULL_BLOCK_IDX);
    group_slots[0] = blocks[0];
    shared_cache->put(/*cache_key=*/42, group_slots, /*is_resident=*/false);
    EXPECT_TRUE(shared_cache->contains(42));

    // The same cache is returned by the allocator accessor.
    EXPECT_EQ(allocator->sharedBlockCache(), shared_cache);

    // Clean up.
    pool0->requestFree(blocks);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedInsertThenReuseSamePrefix) {
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

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    seed_malloc.cp_slot_mapper      = cp_mapper;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    insert_info.cp_slot_mapper = cp_mapper;
    allocator->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    auto hit_res = makeBatchResource(/*batch_size=*/1, config);
    hit_res->setBatchCacheKeys(0, request_keys);
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.reuse_cache         = true;
    hit_malloc.enable_device_cache = true;
    hit_malloc.cp_slot_mapper      = cp_mapper;
    auto result                    = allocator->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 5 * spb * 2);

    FreeInfo hit_free{hit_res, hit_tokens};
    allocator->free(hit_free);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
