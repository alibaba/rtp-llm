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
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/allocator/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/spec/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
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
// so HybridPoolKVCacheAllocator creates two independent BlockPools.
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

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
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
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.use_independent_block_pools = true;
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, second_type},
                            {"linear", second_type == CacheGroupType::SWA ? "swa" : "full"});

    // Same tokens per block for both groups.
    config.group_seq_size_per_block = {config.seq_size_per_block, config.seq_size_per_block};

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
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config      = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);
    config.block_num = block_num;
    return config;
}

static void setExplicitBlocksForGroup(CacheConfig& config, size_t group_id, uint32_t block_num) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        policies.push_back(config.policyForGroup(gid));
    }
    policies[group_id].explicit_block_num = block_num;
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
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.layerGroupIdsSnapshot(),
                    config.kernelBlocksPerKvBlock(),
                    config.groupTypesSnapshot());
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

// Create HybridPoolKVCacheAllocator (SharedBlockCache removed, wire replacement here).
static HybridPoolKVCacheAllocatorPtr makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION) {
    auto allocator =
        std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE, nullptr, 0, role_type);
    // TODO(block_tree_cache refactor): SharedBlockCache removed, wire BlockTreeCache replacement here
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
// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
static BlockIdxType
seedNonResidentCacheItem(const HybridPoolKVCacheAllocatorPtr& allocator, int gid, CacheKeyType key) {
    auto pool   = allocator->groupBlockPools()[static_cast<size_t>(gid)];
    auto blocks = pool->malloc(1).value();
    EXPECT_EQ(blocks.size(), 1u);
    // Replicate the legacy malloc request ref (single-count malloc leaves refCount 0);
    // the subsequent decRef must have a prior holder or it aborts.
    pool->incRef(blocks);
    auto                      shared_cache = allocator->sharedBlockCache();
    std::vector<BlockIdxType> group_slots(allocator->groupBlockPools().size(), NULL_BLOCK_IDX);
    group_slots[static_cast<size_t>(gid)] = blocks[0];
    shared_cache->put(key, group_slots, false);
    // SharedBlockCache::put() internally calls pool->incRef() (the cache holder), so after
    // releasing the request holder the block stays allocated with refCount 1 (cache-held).
    pool->decRef(blocks);
    return blocks[0];
}

// Single-count DeviceBlockPool exposes only free/total block counts (the legacy per-pool
// request/connector/blockCache ref-count columns and availableBlocksNum() are gone; the
// three-way holder split is not recoverable at the pool). Each independent pool still
#endif
// reports its own free/total, which is what these rollback checks rely on.
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

    // In the refactored design the DeviceKVCacheGroups live inside BlockTreeCache, not the
    // allocator. makeAllocator() returns a pre-init allocator; after init() builds the per-group
    // device pools, tests must inject a BlockTreeCache (mirrors KVCacheManager wiring) before any
    // group access (convertIndexToAddr / malloc / reserve / reuse). The fixture owns the
    // BlockTreeCache so it outlives the allocator's raw pointer.
    bool injectBlockTreeCache(const HybridPoolKVCacheAllocatorPtr& allocator, const CacheConfig& config) {
        KVCacheConfig kv_cache_config;  // device-only: memory/disk tiers disabled
        auto          btc = createBlockTreeCache(config, kv_cache_config, allocator);
        if (!btc) {
            return false;
        }
        allocator->setBlockTreeCache(btc.get());
        block_tree_caches_.push_back(std::move(btc));
        return true;
    }

    std::vector<BlockTreeCachePtr> block_tree_caches_;
};

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_EQ(allocator->groupBlockPools()[0]->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->groupBlockPools()[1]->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolKVCacheAllocatorTest, GetBlockPoolReturnsNullptrInHybridPoolMode) {
    // HybridPoolKVCacheAllocator owns one DeviceBlockPool per group and does not
    // expose a single canonical block_pool_; getDeviceBlockPool() must return nullptr.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    EXPECT_EQ(allocator->getDeviceBlockPool(), nullptr);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, TotalAndFreeBlocksAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    EXPECT_EQ(allocator->availableTokensNum(), 28u);
    EXPECT_EQ(allocator->totalTokensNum(), 28u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseCPVirtualBlockSizeForFullGroups) {
    auto config                     = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    config.group_seq_size_per_block = {100, 4};
    auto allocator                  = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 4u);

    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 8u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 8u);
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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];

    const size_t free_total_before = allocator->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2).value();
    auto         g1_blocks         = pool1->malloc(3).value();
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);
    // Single-count malloc reserves capacity only (refCount 0); take one holder per block to
    // occupy them (replicates the legacy auto request ref).
    pool0->incRef(g0_blocks);
    pool1->incRef(g1_blocks);

    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 5u);
    EXPECT_EQ(allocator->availableBlocksNum(), free_total_before - 5u);
    for (auto b : g0_blocks) {
        EXPECT_EQ(pool0->refCount(b), 1u);
    }
    for (auto b : g1_blocks) {
        EXPECT_EQ(pool1->refCount(b), 1u);
    }

    // Take a SECOND holder on one block per group (what a connector transfer would do in the
    // single-count model: another incRef on the same block).
    pool0->incRef(g0_blocks[0]);
    pool1->incRef(g1_blocks[0]);
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 2u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 2u);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 0u);

    // Release the first (request) holder on every block.
    pool0->decRef(g0_blocks);
    pool1->decRef(g1_blocks);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);

    // The two doubly-held blocks stay allocated (refCount drops to 1); the rest are freed.
    EXPECT_TRUE(pool0->isAllocated(g0_blocks[0]));
    EXPECT_TRUE(pool1->isAllocated(g1_blocks[0]));
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 1u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 1u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 2u);
    // notInUseBlocksNum == Σ freeBlocksNum in the single-count model, so held blocks are excluded.
    EXPECT_EQ(allocator->notInUseBlocksNum(), free_total_before - 2u);

    // Release the second holder → blocks are fully freed and availability is restored.
    pool0->decRef(g0_blocks[0]);
    pool1->decRef(g1_blocks[0]);
    EXPECT_FALSE(pool0->isAllocated(g0_blocks[0]));
    EXPECT_FALSE(pool1->isAllocated(g1_blocks[0]));
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before);
    EXPECT_EQ(allocator->availableBlocksNum(), free_total_before);
    EXPECT_EQ(allocator->notInUseBlocksNum(), free_total_before);
}

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheRefAggregatesAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const size_t available_before = allocator->availableBlocksNum();
    const size_t free_before      = allocator->freeBlocksNum();

    const auto b0 = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    const auto b1 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    const auto b2 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/201);

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];
    // 3 blocks are cache-held across the two groups: each is held only by the cache (refCount 1).
    EXPECT_EQ(pool0->refCount(b0), 1u);
    EXPECT_EQ(pool1->refCount(b1), 1u);
    EXPECT_EQ(pool1->refCount(b2), 1u);

    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);

    // Cache-only (refCount==1) blocks stay evictable, so they remain available even though
    // they are no longer free; 3 blocks moved from free to cache-evictable.
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3u);
    EXPECT_EQ(allocator->availableBlocksNum(), available_before);
}
#endif

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto bufs = allocator->convertIndexToBuffer(
        /*layer_id=*/3, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "layer " << i << " missing kv buffer";
    }
    EXPECT_EQ(layout.layer_to_group_ids, config.layerGroupIdsSnapshot());
    EXPECT_EQ(layout.group_types, config.groupTypesSnapshot());

    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_group.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs_by_group.size(); ++i) {
        EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_group[i].size(), static_cast<size_t>(config.groupNums()));
    }

    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
        ASSERT_FALSE(layout.layer_to_group_ids[i].empty());
        const auto  gid        = static_cast<size_t>(layout.layer_to_group_ids[i].front());
        const auto& by_default = layout.layers_to_kv_buffer_ptrs_by_group[i][gid];
        EXPECT_TRUE(by_default.defined()) << "layer " << i << " primary group tensor undefined";
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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // No CacheStore is plumbed in: regUserMr should be a benign no-op for every
    // group pool, and the aggregated MR cost remains zero.
    EXPECT_NO_THROW(allocator->regUserMr(/*model_id=*/0, /*cache_store=*/nullptr));
    EXPECT_EQ(allocator->getMrCostTimeMs(), 0);
}

// ---------------------------------------------------------------------------
// popBlocksFromCache / blockCacheFree
// ---------------------------------------------------------------------------

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheReturnsEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // Seed identical key on both groups, plus a unique key on the full group.
    auto g0_block_for_100 = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    auto g1_block_for_100 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/100);
    auto g1_block_for_200 = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    // The 3 seeded blocks are cache-held (refCount 1) across the two group pools.
    EXPECT_EQ(allocator->groupBlockPools()[0]->refCount(g0_block_for_100), 1u);
    EXPECT_EQ(allocator->groupBlockPools()[1]->refCount(g1_block_for_100), 1u);
    EXPECT_EQ(allocator->groupBlockPools()[1]->refCount(g1_block_for_200), 1u);

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
#endif

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheZeroFreeReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    EXPECT_EQ(allocator->popBlocksFromCache(0), nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheEmptyCachesReturnsNull) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    EXPECT_EQ(allocator->popBlocksFromCache(/*min_blocks_to_free=*/4), nullptr);
}

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeReleasesEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto b0    = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    const auto b1    = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/200);
    auto       pool0 = allocator->groupBlockPools()[0];
    auto       pool1 = allocator->groupBlockPools()[1];
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(pool0->refCount(b0), 1u);
    EXPECT_EQ(pool1->refCount(b1), 1u);

    const size_t free_before = allocator->freeBlocksNum();
    auto         evicted     = allocator->popBlocksFromCache(/*min_blocks_to_free=*/2);
    ASSERT_NE(evicted, nullptr);
    // Eviction removes the LRU entries from BlockCache; the underlying blocks are still held
    // (refCount 1). Releasing those holders is what blockCacheFree() does.
    EXPECT_TRUE(pool0->isAllocated(b0));
    EXPECT_TRUE(pool1->isAllocated(b1));
    allocator->blockCacheFree(evicted);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    EXPECT_FALSE(pool0->isAllocated(b0));
    EXPECT_FALSE(pool1->isAllocated(b1));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 2u);
}
#endif

TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeNullPtrIsNoOp) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    EXPECT_NO_THROW(allocator->blockCacheFree(nullptr));
}

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, BlockCacheFreeIgnoresDuplicateAndNullBlockIds) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto         seeded     = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/300);
    auto         pool1      = allocator->groupBlockPools()[1];
    const size_t free_before = allocator->freeBlocksNum();
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(pool1->refCount(seeded), 1u);

    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(config.groupNums(), static_cast<int>(config.layer_all_num), config.layerGroupIdsSnapshot());
    // Same block listed twice in the same group should only be released once (decRef
    // aborts at refCount 0); NULL_BLOCK_IDX entries should be skipped.
    batch->mutableBlockIds(0, /*gid=*/1).assign(BlockIndicesType{seeded, seeded, NULL_BLOCK_IDX});
    EXPECT_NO_THROW(allocator->blockCacheFree(batch));
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    // The single cache holder was released exactly once, freeing the block.
    EXPECT_FALSE(pool1->isAllocated(seeded));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 1u);
}
#endif

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    constexpr size_t reserve_blocks = 6;
    allocator->setReserveBlockNum(reserve_blocks);

    const auto snapshots = allocator->poolMetricsSnapshots();
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

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/20, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    allocator->setReserveBlockNum(1);

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto linear_cached = seedNonResidentCacheItem(allocator, /*gid=*/0, /*key=*/100);
    const auto full_cached   = seedNonResidentCacheItem(allocator, /*gid=*/1, /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    ASSERT_EQ(allocator->requestRefBlocksNum(), 0u);
    ASSERT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];
    // The 2 device-reuse (cache) blocks are held by the cache (refCount 1).
    ASSERT_EQ(pool0->refCount(linear_cached), 1u);
    ASSERT_EQ(pool1->refCount(full_cached), 1u);

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
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
    // Reserve-reject rolls back request refs but keeps the device-reuse (cache) refs: the 2
    // cached blocks survive (still refCount 1) and availability is unchanged.
    EXPECT_EQ(pool0->refCount(linear_cached), 1u);
    EXPECT_EQ(pool1->refCount(full_cached), 1u);
    EXPECT_EQ(allocator->availableBlocksNum(), available_before);
    expectPoolCountersEq(allocator, counters_before);
}
#endif

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/2);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
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
    auto block_nums           = groupBlockNumsSnapshot(config);
    block_nums[hca_state_gid] = 2;
    setGroupBlockNums(config, block_nums);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_gid));

    const auto hca_state_tokens =
        allocator->groupBlockPools()[hca_state_gid]->totalBlocksNum() * config.group_seq_size_per_block[hca_state_gid];
    EXPECT_LT(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->maxAvailableTokensNum());
    EXPECT_EQ(allocator->totalTokensNum(), allocator->maxAvailableTokensNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigUsesOnlyPagedGroupsForBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);

    ASSERT_EQ(config.groupNums(), 7);
    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_full_bytes = 0;
    size_t expected_full_bytes     = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto type = config.typeForGroup(gid);
        if (type == CacheGroupType::FULL) {
            expected_full_bytes += config.blockSizeBytesForGroup(gid);
        } else {
            expected_non_full_bytes += config.blockSizeBytesForGroup(gid);
        }
    }

    EXPECT_GT(expected_non_full_bytes, 0u);
    EXPECT_GT(expected_full_bytes, 0u);

    EXPECT_EQ(config.block_size_bytes, expected_full_bytes);
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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4StateSwaPoolsWithoutExplicitBlocksUseGlobalBlocks) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);
    config.linear_step = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 128u) << "gid=" << gid;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

    auto addr_swa = allocator->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_DEATH((void)allocator->convertIndexToAddr(csa_layer, /*block_id=*/1), "");
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_group.size(), static_cast<size_t>(config.layer_all_num));

    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num); ++l) {
        EXPECT_FALSE(layout.layers_to_kv_buffer_ptrs[l].defined())
            << "multi-tag DSV4 layer should not publish a legacy single-group tensor";
        const auto& swa_t = layout.layers_to_kv_buffer_ptrs_by_group[l][config.groupIdForTag("swa_kv")];
        EXPECT_TRUE(swa_t.defined()) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.group_tags.size(), 7u);
    EXPECT_EQ(layout.group_types.size(), 7u);
}

// TODO(block_tree_cache refactor): re-enable after SharedBlockCache is replaced
#if 0
TEST_F(HybridPoolKVCacheAllocatorTest, DSV4SharedBlockCacheIsUnifiedAcrossGroups) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // All groups share a single SharedBlockCache owned by the allocator.
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Inserting a cache item for one group is visible via the shared cache.
    auto pool0  = allocator->groupBlockPools()[0];
    auto blocks = pool0->malloc(1).value();
    ASSERT_EQ(blocks.size(), 1u);
    // Single-count malloc reserves capacity only; take the request holder before put()'s
    // cache holder, so the later decRef has a prior holder.
    pool0->incRef(blocks);
    std::vector<BlockIdxType> group_slots(allocator->groupBlockPools().size(), NULL_BLOCK_IDX);
    group_slots[0] = blocks[0];
    shared_cache->put(/*cache_key=*/42, group_slots, /*is_resident=*/false);
    EXPECT_TRUE(shared_cache->contains(42));

    // The same cache is returned by the allocator accessor.
    EXPECT_EQ(allocator->sharedBlockCache(), shared_cache);

    // Clean up: release the request holder (the cache holder from put() keeps the block).
    pool0->decRef(blocks);
}
#endif

// Single-count co-hold invariant: a block co-held by a request holder and a cache holder is
// not freed when the request holder is released; only releasing the last holder frees it.
TEST_F(HybridPoolKVCacheAllocatorTest, RequestReleaseDoesNotFreeCachedBlock) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    auto pool  = allocator->groupBlockPools()[0];
    auto block = pool->malloc().value();
    pool->incRef(block);  // cache holder
    pool->incRef(block);  // request holder
    EXPECT_EQ(pool->refCount(block), 2u);
    pool->decRef(block);  // release request holder
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);  // release cache holder
    EXPECT_FALSE(pool->isAllocated(block));
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedInsertThenReuseSamePrefix) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedEvictionReclaimsDeviceBlocksInPlace) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

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

    // In the refactored BlockTreeCache design popBlocksFromCache no longer surfaces the
    // evicted resource: it reclaims device blocks in place, returns them to the owning
    // pools, and always yields nullptr. Verify the in-place reclaim side effect through the
    // free-block counter (cache-held blocks become free again). The CP-canonical cache-key
    // semantics this case used to assert on the evicted handle are covered by
    // DSV4CPShardedInsertThenReuseSamePrefix (insert -> reuse path).
    const size_t free_before = allocator->freeBlocksNum();
    auto         evicted     = allocator->popBlocksFromCache(/*min_blocks_to_free=*/4);
    EXPECT_EQ(evicted, nullptr);
    EXPECT_GE(allocator->freeBlocksNum(), free_before + 4);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
