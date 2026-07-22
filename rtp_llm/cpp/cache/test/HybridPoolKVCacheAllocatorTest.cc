#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {
using block_tree_cache_test::BlockTreeCacheTestPeer;

using TestHybridPoolKVCacheAllocator = BlockTreeCacheTestAllocator<HybridPoolKVCacheAllocator>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a tiny multi-pool config with two groups: gid=0 LINEAR(layers 0,1)
// and gid=1 FULL(layers 2,3). Each group has its own per-group block budget,
// so TestHybridPoolKVCacheAllocator creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t       linear_block_num = 6,
                                                 uint32_t       full_block_num   = 8,
                                                 CacheGroupType second_type      = CacheGroupType::FULL) {
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = std::max(linear_block_num, full_block_num);
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 4;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

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

    config.use_independent_block_pools = true;
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, second_type},
                            {"linear", second_type == CacheGroupType::SWA ? "swa" : "full"});

    // Same tokens per block for both groups.
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(config.kv_block_stride_bytes));
    const auto linear_stride = linear_spec->block_size_bytes();
    const auto full_stride   = full_spec->block_size_bytes();
    config.setGroupBlockLayout({linear_block_num, full_block_num}, {linear_stride, full_stride}, {0, 0});
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
    mc.attn_config.sliding_window                                = 128;
    mc.attn_config.indexer_head_dim                              = 8;
    mc.attn_config.indexer_head_num                              = 2;
    mc.attn_config.indexer_topk                                  = 16;
    mc.attn_config.o_groups                                      = 2;
    mc.attn_config.o_lora_rank                                   = 16;
    mc.attn_config.tokens_per_block                              = 128;
    mc.attn_config.layer_compress_ratios                         = {4, 128, 4, 128, 0};
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc);
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
    mc.attn_config.tokens_per_block = 128;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

// Build a DSV4 7-pool CacheConfig (uses use_independent_block_pools=true).
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

static void setGroupReservable(CacheConfig& config, size_t group_id, bool reservable) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        policies.push_back(config.policyForGroup(gid));
    }
    policies[group_id].reservable = reservable;
    config.setGroupPolicies(policies);
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
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

static std::shared_ptr<TestHybridPoolKVCacheAllocator>
makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION, int64_t reserve_block_ratio = 0) {
    return std::make_shared<TestHybridPoolKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, reserve_block_ratio, role_type);
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

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int,
              const std::shared_ptr<LoadCopyFence>&) override {
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

struct PoolCounters {
    size_t free_blocks;
    size_t total_blocks;
};

static std::vector<PoolCounters> snapshotPoolCounters(const HybridPoolKVCacheAllocatorPtr& allocator) {
    std::vector<PoolCounters> counters;
    counters.reserve(allocator->groupBlockPools().size());
    for (const auto& pool : allocator->groupBlockPools()) {
        counters.push_back({pool->freeBlocksNum(), pool->totalBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const HybridPoolKVCacheAllocatorPtr& allocator,
                                 const std::vector<PoolCounters>&     expected) {
    ASSERT_EQ(allocator->groupBlockPools().size(), expected.size());
    for (size_t gid = 0; gid < expected.size(); ++gid) {
        const auto& pool = allocator->groupBlockPools()[gid];
        EXPECT_EQ(pool->freeBlocksNum(), expected[gid].free_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->totalBlocksNum(), expected[gid].total_blocks) << "gid=" << gid;
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

TEST_F(HybridPoolKVCacheAllocatorTest, GetDeviceBlockPoolReturnsNullptrInHybridPoolMode) {
    // HybridPoolKVCacheAllocator owns one DeviceBlockPool per group and does not
    // expose a single canonical pool.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getDeviceBlockPool(), nullptr);
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
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseDifferentCapacityScopes) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    // Token capacity aggregators use FULL groups first: 7 blocks * 4 tokens.
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 28u);
    EXPECT_EQ(allocator->availableTokensNum(), 28u);
    EXPECT_EQ(allocator->totalTokensNum(), 28u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseCPVirtualBlockSizeForFullGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 4u);

    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 8u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 8u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsFallBackToGlobalSeqSize) {
    auto config               = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.seq_size_per_block = 4;
    auto allocator            = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IndependentPoolsUseOneBalancedReferenceCount) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];

    const size_t free_total_before = allocator->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2).value();
    auto         g1_blocks         = pool1->malloc(3).value();
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);
    pool0->incRef(g0_blocks);
    pool1->incRef(g1_blocks);

    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 5u);
    for (const auto block : g0_blocks) {
        EXPECT_EQ(pool0->refCount(block), 1u);
    }
    for (const auto block : g1_blocks) {
        EXPECT_EQ(pool1->refCount(block), 1u);
    }

    // A second holder on the same numeric block id remains pool-local.
    pool0->incRef(g0_blocks[0]);
    pool1->incRef(g1_blocks[0]);
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 2u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 2u);

    pool0->decRef(g0_blocks);
    pool1->decRef(g1_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 2u);

    pool0->decRef(g0_blocks[0]);
    pool1->decRef(g1_blocks[0]);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before);
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

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto addr_default   = allocator->convertIndexToAddr(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    auto addr_via_layer = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_via_layer.kv_addr);

    auto bufs_default = allocator->convertIndexToBuffer(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = allocator->convertIndexToBuffer(
        /*layer_id=*/0, /*group_id=*/0, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, AllLayerCacheBaseExposesPerLayerAndPerGroupTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.topology().layerGroupIdsSnapshot(), config.layerGroupIdsSnapshot());
    EXPECT_EQ(layout.topology().groupTypesSnapshot(), config.groupTypesSnapshot());
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
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
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Group 0 (linear) gets 6 blocks (5 free), group 1 (full) gets 4 blocks (3 free).
    // total_available = 8. Set reserve = 4.
    // Expected per-group reserve: floor(4 * 5/8) = 2 for gid=0, floor(4 * 3/8) = 1 for gid=1.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
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

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    // Force a group whose free_blocks < need + group_reserve_blocks.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // A reserve large enough to hide most blocks should reject init malloc.
    allocator->setReserveBlocksNum(allocator->freeBlocksNum());

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

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    constexpr size_t reserve_blocks = 6;
    allocator->setReserveBlocksNum(reserve_blocks);

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ("linear", snapshots[0].pool_name);
    EXPECT_EQ("full", snapshots[1].pool_name);

    const size_t total_reservable_free_blocks = snapshots[0].free_blocks + snapshots[1].free_blocks;
    ASSERT_GT(total_reservable_free_blocks, 0u);
    EXPECT_EQ(reserve_blocks * snapshots[0].free_blocks / total_reservable_free_blocks, snapshots[0].reserve_blocks);
    EXPECT_EQ(reserve_blocks * snapshots[1].free_blocks / total_reservable_free_blocks, snapshots[1].reserve_blocks);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioAndSnapshotsExcludeNonReservablePool) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    setGroupReservable(config, /*linear gid=*/0, false);

    constexpr int64_t reserve_ratio = 50;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);

    const size_t excluded_free      = allocator->groupBlockPools()[0]->freeBlocksNum();
    const size_t participating_free = allocator->groupBlockPools()[1]->freeBlocksNum();
    ASSERT_GT(excluded_free, 0u);
    ASSERT_GT(participating_free, 0u);
    EXPECT_EQ(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * participating_free / static_cast<size_t>(100));
    EXPECT_NE(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * (excluded_free + participating_free) / static_cast<size_t>(100));

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].reserve_blocks, 0u);
    EXPECT_EQ(snapshots[1].reserve_blocks, allocator->reserveBlocksNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, NonReservableOnlyPoolsHaveDivisionSafeZeroReserveShares) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    setGroupReservable(config, /*linear gid=*/0, false);
    setGroupReservable(config, /*full gid=*/1, false);

    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->reserveBlocksNum(), 0u);

    // Exercise snapshot distribution with a non-zero configured reserve and a
    // zero-participant denominator. Every pool must receive a safe zero share.
    allocator->setReserveBlocksNum(7);
    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].reserve_blocks, 0u);
    EXPECT_EQ(snapshots[1].reserve_blocks, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/20, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    allocator->setReserveBlocksNum(1);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;

    auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, /*gid=*/1)), 4u);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
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
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{100});
    ASSERT_TRUE(seeded.success);
    const auto linear_cached = seeded.blocks_by_tag.at(config.tagForGroup(/*gid=*/0)).front();
    const auto full_cached   = seeded.blocks_by_tag.at(config.tagForGroup(/*gid=*/1)).front();
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[0]->refCount(linear_cached), 1u);
    ASSERT_EQ(pools[1]->refCount(full_cached), 1u);

    const size_t available_before = allocator->freeBlocksNum();
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
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(pools[0]->refCount(linear_cached), 1u);
    EXPECT_EQ(pools[1]->refCount(full_cached), 1u);
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

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackRestoresLinearBackfilledSlots) {
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
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 2u);

    auto& linear_ids       = batch_res->mutableBlockIds(0, /*gid=*/0);
    auto  removed_block_id = linear_ids.blocks()[1];
    ASSERT_FALSE(isNullBlockIdx(removed_block_id));
    allocator->groupBlockPools()[0]->decRef(removed_block_id);
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

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 2u);
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, /*gid=*/0)[1]));
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
// DSV4 7-group HybridPool: covers per-tag addressing and SWA tail
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
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        EXPECT_EQ(allocator->groupBlockPools()[gid]->where(), MemoryType::MEMORY_GPU)
            << "gid=" << gid << " tag=" << config.tagForGroup(gid);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator     = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_gid), "hca_state");
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

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsIgnoreSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_gid), "hca_state");
    auto block_nums           = groupBlockNumsSnapshot(config);
    block_nums[hca_state_gid] = 2;
    setGroupBlockNums(config, block_nums);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_gid));

    const auto hca_state_tokens =
        allocator->groupBlockPools()[hca_state_gid]->totalBlocksNum() * config.seq_size_per_block;
    EXPECT_LT(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->maxAvailableTokensNum());
    EXPECT_EQ(allocator->totalTokensNum(), allocator->maxAvailableTokensNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigUsesGroupOwnedBytesForPagedBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_paged_bytes = 0;
    size_t expected_paged_bytes     = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto type = config.typeForGroup(gid);
        const auto expected_group_bytes =
            config.layerIdsForGroup(gid).size()
            * (config.kvBlockStrideBytesForGroup(gid) + config.kvScaleStrideBytesForGroup(gid));
        EXPECT_EQ(config.blockSizeBytesForGroup(gid), expected_group_bytes) << "gid=" << gid;
        if (!config.usesExplicitIndependentBlocks(gid)
            && (type == CacheGroupType::FULL || type == CacheGroupType::LINEAR)) {
            expected_paged_bytes += expected_group_bytes;
        } else {
            expected_non_paged_bytes += expected_group_bytes;
        }
    }

    EXPECT_GT(expected_non_paged_bytes, 0u);
    EXPECT_GT(expected_paged_bytes, 0u);

    EXPECT_EQ(config.block_size_bytes, expected_paged_bytes);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioExcludesExplicitIndependentPools) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    ASSERT_LT(firstExplicitIndependentGroup(config), static_cast<size_t>(config.groupNums()));

    constexpr int64_t reserve_ratio = 10;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());

    size_t reservable_available = 0;
    size_t all_available        = 0;
    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        const size_t available = allocator->groupBlockPools()[gid]->freeBlocksNum();
        all_available += available;
        if (!config.usesExplicitIndependentBlocks(gid)) {
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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesHcaStatePoolBlocks) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }

    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);
    setExplicitBlocksForGroup(config, firstExplicitIndependentGroup(config), 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 123u);
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4StateSwaPoolsWithoutExplicitBlocksScaleWithLinearStep) {
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
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, FinalizeNonExplicitSwaBlocksUsesCeilDivision) {
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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layerTagToGroupIdSnapshot()[l].count("csa_kv") > 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // csa_kv tag routes to gid=0; it must produce a non-null kv address that
    // matches the CSA group's pool.
    auto addr_csa = allocator->convertIndexToAddrByTag(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);
    const auto csa_gid = config.groupIdForTag("csa_kv");
    EXPECT_EQ(addr_csa.kv_addr, allocator->convertIndexToAddr(csa_layer, csa_gid, 1).kv_addr);

    auto addr_swa = allocator->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);
    EXPECT_THROW((void)allocator->convertIndexToAddrByTag(csa_layer, "missing", 1), std::exception);
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, config.groupNums(), 1), std::exception);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, /*block_id=*/1), std::exception);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layerTagToGroupIdSnapshot()[l].count("csa_kv") > 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    auto buf = allocator->convertIndexToBufferByTag(csa_layer, "csa_kv", /*block_id=*/1);
    ASSERT_FALSE(buf.empty());
    EXPECT_NE(buf[0].addr, nullptr);

    auto buf_part = allocator->convertIndexToBufferByTag(
        csa_layer, "csa_kv", /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(buf_part.empty());
    EXPECT_NE(buf_part[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4AllLayerCacheBaseHasPerGroupTensors) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num); ++l) {
        EXPECT_TRUE(layout.group("swa_kv").hasLayer(l)) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.groups().size(), 7u);
    EXPECT_EQ(layout.topology().groups().size(), 7u);
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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedEvictionMarksCanonicalResource) {
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

    KVCacheResource canonical_source;
    canonical_source.setCacheKeys(full_keys);
    const auto expected_canonical = canonical_source.localCacheKeys(cp_mapper->cpSize() - 1, cp_mapper->cpSize());
    const auto before             = allocator->blockTreeCacheOwner()->getKeySnapshot(expected_canonical.size() + 1);
    EXPECT_EQ(before.keys, expected_canonical);

    // The allocator publishes to BlockTreeCache, so eviction is observed through
    // the tree rather than the legacy allocator pop-resource facade.
    EXPECT_GT(BlockTreeCacheTestPeer::reclaimBlocksForTest(*allocator->blockTreeCacheOwner(), /*num_blocks=*/4096), 0);
    const auto after = allocator->blockTreeCacheOwner()->getKeySnapshot(expected_canonical.size() + 1);
    EXPECT_GT(after.version, before.version);
    EXPECT_TRUE(after.keys.empty());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
