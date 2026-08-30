#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
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
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
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

// Build a tiny multi-pool config with two groups: "linear" LINEAR(layers 0,1)
// and "full" FULL(layers 2,3). Each group has its own per-group block budget,
// so HybridPoolKVCacheAllocator creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t       linear_block_num          = 6,
                                                 uint32_t       full_block_num            = 8,
                                                 CacheGroupType second_type               = CacheGroupType::FULL,
                                                 uint32_t       linear_active_tail_blocks = 0) {
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

    const auto second_tag    = second_type == CacheGroupType::SWA ? "swa" : "full";
    auto       linear_policy = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    if (linear_active_tail_blocks > 0) {
        linear_policy.active_tail_blocks = linear_active_tail_blocks;
    }
    rtp_llm::test::assignCacheConfigFromGroupedSpecs(config,
                                                     config.layer_num,
                                                     {linear_spec, full_spec},
                                                     {{0, 1}, {2, 3}},
                                                     {CacheGroupType::LINEAR, second_type},
                                                     {"linear", second_tag},
                                                     {linear_policy, defaultCacheGroupPolicy(second_type)});

    // Same tokens per block for both groups.
    const auto linear_stride = linear_spec->block_size_bytes();
    const auto full_stride   = full_spec->block_size_bytes();
    setGroupBlockLayout(config, {linear_block_num, full_block_num}, {linear_stride, full_stride}, {0, 0});
    return config;
}

static CacheConfig makeTinySwaMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t swa_block_num = 8) {
    return makeTinyMultiPoolHybridConfig(linear_block_num, swa_block_num, CacheGroupType::SWA);
}

static CacheConfig makeTinySingleMhaConfig(CacheGroupType group_type, const std::string& tag, uint32_t block_num = 8) {
    CacheConfig config;
    config.dtype              = rtp_llm::DataType::TYPE_FP16;
    config.layer_num          = 2;
    config.block_num          = block_num;
    config.seq_size_per_block = 4;
    config.linear_step        = 2;

    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, static_cast<uint32_t>(config.seq_size_per_block), tag);
    rtp_llm::test::assignCacheConfigFromGroupedSpecs(
        config, config.layer_num, {full_spec}, {{0, 1}}, {group_type}, {tag});
    setGroupBlockLayout(config, {block_num}, {full_spec->block_size_bytes()}, {0});
    return config;
}

static CacheConfig makeTinySingleFullConfig(uint32_t block_num = 8) {
    return makeTinySingleMhaConfig(CacheGroupType::FULL, "full", block_num);
}

static CacheConfig makeTinySingleSwaConfig(uint32_t block_num = 8) {
    return makeTinySingleMhaConfig(CacheGroupType::SWA, "swa", block_num);
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

static ModelConfig makeOrdinaryMtpModelConfig(uint32_t        num_layers,
                                              uint32_t        kv_head_num,
                                              uint32_t        size_per_head,
                                              KvCacheDataType kv_cache_dtype) {
    ModelConfig config;
    config.num_layers                   = static_cast<int64_t>(num_layers);
    config.max_seq_len                  = 128;
    config.hidden_size                  = 64;
    config.vocab_size                   = 1024;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 4;
    config.attn_config.kv_head_num      = static_cast<int>(kv_head_num);
    config.attn_config.size_per_head    = static_cast<int>(size_per_head);
    config.attn_config.tokens_per_block = 4;
    config.attn_config.use_mla          = false;
    config.attn_config.kv_cache_dtype   = kv_cache_dtype;
    setDefaultKvCacheSpec(config);
    return config;
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
static CacheConfig makeDSV4HybridPoolConfig(uint32_t                block_num        = 200,
                                            std::optional<uint32_t> hca_state_blocks = std::nullopt) {
    auto mc                                            = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    if (hca_state_blocks.has_value()) {
        setDsv4ExplicitPoolBlocks(mc, "hca_state", *hca_state_blocks);
    }
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, 0);
    config.finalizeBlockNums(block_num, RuntimeConfig{});
    return config;
}

static std::string firstExplicitIndependentGroupTag(const CacheConfig& config) {
    for (const auto& group : config.groups()) {
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
    res->initGroups(config);
    return res;
}

static std::map<std::string, uint32_t> blockNumsByTag(const CacheConfig& config) {
    std::map<std::string, uint32_t> block_nums;
    for (const auto& group : config.groups()) {
        block_nums[group.tag] = group.block_num;
    }
    return block_nums;
}

// Override named group capacities through the test-only topology helper.
static void setGroupBlockNums(CacheConfig& config, const std::map<std::string, uint32_t>& block_nums_by_tag) {
    std::vector<uint32_t> block_nums;
    std::vector<size_t>   kv_strides;
    std::vector<size_t>   scale_strides;
    for (const auto& group : config.groups()) {
        const auto it = block_nums_by_tag.find(group.tag);
        block_nums.push_back(it == block_nums_by_tag.end() ? group.block_num : it->second);
        kv_strides.push_back(group.kv_block_stride_bytes);
        scale_strides.push_back(group.kv_scale_stride_bytes);
    }
    setGroupBlockLayout(config, block_nums, kv_strides, scale_strides);
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

// Create HybridPoolKVCacheAllocator with SharedBlockCache injected (required before init()).
static HybridPoolKVCacheAllocatorPtr
makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION, int64_t reserve_block_ratio = 0) {
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, reserve_block_ratio, role_type);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    return allocator;
}

class IncrementalFailureInjectingAllocator final: public HybridPoolKVCacheAllocator {
public:
    explicit IncrementalFailureInjectingAllocator(const CacheConfig& config):
        HybridPoolKVCacheAllocator(config, AllocationType::DEVICE, nullptr, 0, RoleType::PDFUSION) {}

    std::string                      fail_tag;
    bool                             failure_enabled = false;
    mutable std::vector<std::string> observed_tags;
    mutable BlockIndicesType         linear_blocks_at_failure;

protected:
    bool shouldInjectGroupAllocationFailureForTest(const BatchKVCacheResource& resource,
                                                   int                         batch_id,
                                                   std::string_view            tag,
                                                   bool                        incremental) const override {
        if (!failure_enabled || !incremental) {
            return false;
        }
        observed_tags.emplace_back(tag);
        if (tag != fail_tag) {
            return false;
        }
        linear_blocks_at_failure = resource.blocks(batch_id, "linear");
        return true;
    }
};

static std::shared_ptr<IncrementalFailureInjectingAllocator> makeFailingAllocator(const CacheConfig& config) {
    auto allocator = std::make_shared<IncrementalFailureInjectingAllocator>(config);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    allocator->fail_tag = "full";
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

// Insert a cache item into the shared block cache for a specific group.
// Returns the BlockIdx allocated for the item (kept blockCache-referenced + request-released).
static BlockIdxType seedCacheItem(const HybridPoolKVCacheAllocatorPtr& allocator,
                                  const CacheConfig&                   config,
                                  std::string_view                     tag,
                                  CacheKeyType                         key,
                                  bool                                 is_resident = false) {
    auto pool   = allocator->blockPool(tag);
    auto blocks = pool->malloc(1);
    EXPECT_EQ(blocks.size(), 1u);
    config.group(tag);
    auto shared_cache = allocator->sharedBlockCache();
    shared_cache->put(key, {{std::string(tag), blocks[0]}}, is_resident);
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

static std::vector<PoolCounters> snapshotPoolCounters(const HybridPoolKVCacheAllocatorPtr& allocator,
                                                      const CacheConfig&                   config) {
    std::vector<PoolCounters> counters;
    counters.reserve(config.groups().size());
    for (const auto& group : config.groups()) {
        const auto& pool = allocator->blockPool(group.tag);
        counters.push_back({pool->freeBlocksNum(),
                            pool->availableBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            pool->connectorRefBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const HybridPoolKVCacheAllocatorPtr& allocator,
                                 const CacheConfig&                   config,
                                 const std::vector<PoolCounters>&     expected) {
    ASSERT_EQ(config.groups().size(), expected.size());
    for (size_t group_ordinal = 0; group_ordinal < expected.size(); ++group_ordinal) {
        const auto& tag  = config.groups()[group_ordinal].tag;
        const auto& pool = allocator->blockPool(tag);
        EXPECT_EQ(pool->freeBlocksNum(), expected[group_ordinal].free_blocks) << "tag=" << tag;
        EXPECT_EQ(pool->availableBlocksNum(), expected[group_ordinal].available_blocks) << "tag=" << tag;
        EXPECT_EQ(pool->requestRefBlocksNum(), expected[group_ordinal].request_refs) << "tag=" << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), expected[group_ordinal].block_cache_refs) << "tag=" << tag;
        EXPECT_EQ(pool->connectorRefBlocksNum(), expected[group_ordinal].connector_refs) << "tag=" << tag;
    }
}

class HybridPoolKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

class FailOnceOnFullGroupInitAllocator: public HybridPoolKVCacheAllocator {
public:
    using HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator;

protected:
    bool initGroup(const KVCacheGroupPtr& group) override {
        if (!failed_once_ && group->tag() == "full") {
            failed_once_ = true;
            return false;
        }
        return group->init();
    }

private:
    bool failed_once_ = false;
};

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, FailedGroupInitPublishesNothingAndRetryStartsClean) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = std::make_shared<FailOnceOnFullGroupInitAllocator>(
        config, AllocationType::DEVICE, nullptr, 0, RoleType::PDFUSION);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());

    EXPECT_THROW((void)allocator->init(), std::exception);
    EXPECT_THROW((void)allocator->blockPool("linear"), std::exception);
    EXPECT_THROW((void)allocator->blockPool("full"), std::exception);

    ASSERT_TRUE(allocator->init());
    EXPECT_NE(allocator->blockPool("linear"), nullptr);
    EXPECT_NE(allocator->blockPool("full"), nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_NE(allocator->blockPool("linear"), allocator->blockPool("full"));

    EXPECT_EQ(allocator->blockPool("linear")->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(allocator->blockPool("full")->totalBlocksNum(), 8u - 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, OrdinarySingleMtpUsesExactMainAndProposeMemoryLayouts) {
    auto score_config = makeOrdinaryMtpModelConfig(
        /*num_layers=*/2, /*kv_head_num=*/1, /*size_per_head=*/8, KvCacheDataType::BASE);
    auto propose_config = makeOrdinaryMtpModelConfig(
        /*num_layers=*/1, /*kv_head_num=*/2, /*size_per_head=*/16, KvCacheDataType::FP8);

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 6;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;
    auto config                 = CacheConfigCreator::createSpConfig(score_config,
                                                     propose_config,
                                                     ParallelismConfig{},
                                                     RuntimeConfig{},
                                                     kv_cache_config,
                                                     sp_config,
                                                     /*warm_up_result=*/std::nullopt,
                                                     /*is_mtp=*/true,
                                                     /*is_eagle=*/false);

    ASSERT_EQ(config.groupNums(), 1);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    ASSERT_EQ(config.layer_num, 2u);
    ASSERT_EQ(config.layer_all_num, 4u);

    const auto legacy_contract = BlockPoolConfigHelper::createConfig(config);
    ASSERT_EQ(legacy_contract.memory_layouts.size(), 3u);
    EXPECT_EQ(legacy_contract.memory_layouts[0].layer_num, 2u);
    EXPECT_EQ(legacy_contract.memory_layouts[1].layer_num, 1u);
    EXPECT_EQ(legacy_contract.memory_layouts[2].layer_num, 1u);
    EXPECT_EQ(legacy_contract.memory_layouts[0].kv_scale_stride_bytes, 0u);
    EXPECT_GT(legacy_contract.memory_layouts[1].kv_scale_stride_bytes, 0u);
    EXPECT_EQ(legacy_contract.memory_layouts[1].kv_block_stride_bytes,
              config.mtp_sub_configs[0]->soleGroupForLayer(0).spec->block_size_bytes());
    EXPECT_NE(legacy_contract.memory_layouts[0].kv_block_stride_bytes,
              legacy_contract.memory_layouts[1].kv_block_stride_bytes);

    size_t expected_offset = 0;
    for (const auto& memory_layout : legacy_contract.memory_layouts) {
        EXPECT_EQ(memory_layout.kv_cache_offset_bytes, expected_offset);
        expected_offset += memory_layout.kv_block_pool_size_bytes;
        EXPECT_EQ(memory_layout.kv_scale_offset_bytes, expected_offset);
        expected_offset += memory_layout.kv_scale_pool_size_bytes;
    }
    EXPECT_EQ(legacy_contract.total_size_bytes, expected_offset);

    auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());
    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    const auto pool = allocator->soleGroupBlockPool();
    ASSERT_NE(pool, nullptr);

    const auto& actual_contract = pool->config_;
    ASSERT_EQ(actual_contract.memory_layouts.size(), legacy_contract.memory_layouts.size());
    EXPECT_EQ(actual_contract.total_size_bytes, legacy_contract.total_size_bytes);
    EXPECT_EQ(pool->getTotalSizeBytes(), legacy_contract.total_size_bytes);

    const auto  all_layout = manager->allLayerCacheBase();
    const auto& group      = all_layout.group("default");
    ASSERT_EQ(group.activeLayerCount(), 4u);
    const auto* pool_base = static_cast<const char*>(pool->getBaseAddress());

    for (size_t global_layer = 0; global_layer < 4; ++global_layer) {
        const size_t layout_id   = global_layer < 2 ? 0 : global_layer - 1;
        const size_t local_layer = global_layer < 2 ? global_layer : 0;
        const auto&  expected    = legacy_contract.memory_layouts[layout_id];
        const auto&  actual      = actual_contract.memory_layouts[layout_id];
        EXPECT_EQ(actual.layer_num, expected.layer_num);
        EXPECT_EQ(actual.block_num, expected.block_num);
        EXPECT_EQ(actual.kv_block_stride_bytes, expected.kv_block_stride_bytes);
        EXPECT_EQ(actual.kv_scale_stride_bytes, expected.kv_scale_stride_bytes);
        EXPECT_EQ(actual.kv_cache_offset_bytes, expected.kv_cache_offset_bytes);
        EXPECT_EQ(actual.kv_scale_offset_bytes, expected.kv_scale_offset_bytes);

        ASSERT_TRUE(group.hasLayer(global_layer));
        const auto& layer = group.at(global_layer);
        ASSERT_TRUE(layer.kv_addr.defined());
        EXPECT_EQ(layer.kv_addr.sizes().vec(),
                  (std::vector<int64_t>{
                      static_cast<int64_t>(expected.block_num),
                      static_cast<int64_t>(expected.kv_block_stride_bytes / rtp_llm::getTypeSize(expected.dtype))}));
        EXPECT_EQ(layer.kv_addr.nbytes(), static_cast<size_t>(expected.block_num) * expected.kv_block_stride_bytes);

        const auto block_addr       = allocator->convertIndexToAddr(static_cast<int>(global_layer), /*block_id=*/1);
        const auto expected_kv_addr = pool_base + expected.kv_cache_offset_bytes
                                      + local_layer * expected.block_num * expected.kv_block_stride_bytes
                                      + expected.kv_block_stride_bytes;
        EXPECT_EQ(block_addr.kv_addr, expected_kv_addr);

        if (expected.hasScale()) {
            ASSERT_TRUE(layer.kv_scale_addr.defined());
            EXPECT_EQ(layer.kv_scale_addr.sizes().vec(),
                      (std::vector<int64_t>{static_cast<int64_t>(expected.block_num),
                                            static_cast<int64_t>(expected.kv_scale_stride_bytes / sizeof(float))}));
            EXPECT_EQ(layer.kv_scale_addr.nbytes(),
                      static_cast<size_t>(expected.block_num) * expected.kv_scale_stride_bytes);
            const auto expected_scale_addr = pool_base + expected.kv_scale_offset_bytes
                                             + local_layer * expected.block_num * expected.kv_scale_stride_bytes
                                             + expected.kv_scale_stride_bytes;
            EXPECT_EQ(block_addr.kv_scale_addr, expected_scale_addr);
        } else {
            EXPECT_FALSE(layer.kv_scale_addr.defined());
            EXPECT_EQ(block_addr.kv_scale_addr, nullptr);
        }
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, SingleFullCoordinatorSupportsInitAndIncrementalAllocation) {
    auto config    = makeTinySingleFullConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_NE(allocator->blockPool("full"), nullptr);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 1u);

    const auto first_block = batch_res->blocks(0, "full").front();
    token_ids->setSeqLength(8);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(incr_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2u);
    EXPECT_EQ(batch_res->blocks(0, "full").front(), first_block);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SingleFullCoordinatorReusesCachedPrefix) {
    auto config    = makeTinySingleFullConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto cached_block = seedCacheItem(allocator, config, "full", /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(cached_block));

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;

    const auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 4);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2u);
    EXPECT_EQ(batch_res->blocks(0, "full").front(), cached_block);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SingleSwaWithoutPrefixReuseDoesNotReportPhantomReuse) {
    auto config = makeTinySingleSwaConfig();
    ASSERT_FALSE(config.group("swa").policy.enable_prefix_reuse);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;

    const auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SingleFullCoordinatorFreesEveryRequestReference) {
    auto config    = makeTinySingleFullConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    const auto counters_before = snapshotPoolCounters(allocator, config);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101});
    batch_res->setBatchCacheKeys(1, CacheKeysType{100, 101});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_GT(allocator->requestRefBlocksNum(), 0u);

    allocator->free(FreeInfo{batch_res, token_ids});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    expectPoolCountersEq(allocator, config, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SingleFullCoordinatorConvertsAddressesAndCopiesBlocks) {
    auto config    = makeTinySingleFullConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    const auto src = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    const auto dst = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/2);
    ASSERT_NE(src.kv_addr, nullptr);
    ASSERT_NE(dst.kv_addr, nullptr);
    ASSERT_NE(src.kv_addr, dst.kv_addr);

    const auto bytes = config.group("full").spec->block_size_bytes();
    memset(src.kv_addr, 0x5a, bytes);
    memset(dst.kv_addr, 0, bytes);
    allocator->blockBatchCopyByTag({TaggedBlockIdPair{"full", 1, 2}});
    EXPECT_EQ(memcmp(src.kv_addr, dst.kv_addr, bytes), 0);
}

TEST_F(HybridPoolKVCacheAllocatorTest, MixedCoordinatorRollsBackEarlierBatchAndGroupOnLaterFailure) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/5, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/2, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    batch_res->setBatchCacheKeys(1, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    std::vector<std::map<std::string, BlockIndicesType>> blocks_before(2);
    for (int batch_id = 0; batch_id < 2; ++batch_id) {
        for (const auto& group : config.groups()) {
            blocks_before[static_cast<size_t>(batch_id)][group.tag] = batch_res->blocks(batch_id, group.tag);
        }
    }
    const auto counters_before = snapshotPoolCounters(allocator, config);
    token_ids->setSeqLength(8);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    EXPECT_FALSE(allocator->malloc(incr_info).success);

    for (int batch_id = 0; batch_id < 2; ++batch_id) {
        for (const auto& group : config.groups()) {
            EXPECT_EQ(batch_res->blocks(batch_id, group.tag),
                      blocks_before[static_cast<size_t>(batch_id)].at(group.tag));
        }
    }
    expectPoolCountersEq(allocator, config, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->blockPool("linear")->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->blockPool("swa")->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolKVCacheAllocatorTest, GetBlockPoolReturnsNullptrInHybridPoolMode) {
    // HybridPoolKVCacheAllocator owns one BlockPool per group and does not
    // expose a single canonical block_pool_; getBlockPool() must return nullptr.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getBlockPool(), nullptr);
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

TEST_F(HybridPoolKVCacheAllocatorTest, RequestAndConnectorRefAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto pool0 = allocator->blockPool("linear");
    auto pool1 = allocator->blockPool("full");

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

    seedCacheItem(allocator, config, "linear", /*key=*/100);
    seedCacheItem(allocator, config, "full", /*key=*/200);
    seedCacheItem(allocator, config, "full", /*key=*/201);

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

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByTag) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto addr_default   = allocator->convertIndexToAddrByTag(/*layer_id=*/0, "linear", /*block_id=*/1);
    auto addr_via_layer = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_via_layer.kv_addr);

    auto bufs_default = allocator->convertIndexToBufferByTag(/*layer_id=*/0, "linear", /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = allocator->convertIndexToBufferByTag(
        /*layer_id=*/0, "linear", /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, AllLayerCacheBaseExposesPerLayerAndPerGroupTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.topology().layers().size(), config.layers().size());
    for (size_t layer_id = 0; layer_id < config.layers().size(); ++layer_id) {
        EXPECT_EQ(layout.topology().groupsForLayer(static_cast<int>(layer_id)), config.layers()[layer_id])
            << "layer " << layer_id;
    }
    for (const auto& group : config.groups()) {
        EXPECT_EQ(layout.topology().group(group.tag).policy.group_type, group.policy.group_type) << group.tag;
    }
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
        const auto& layer_tags = layout.topology().groupsForLayer(static_cast<int>(i));
        ASSERT_FALSE(layer_tags.empty());
        for (const auto& tag : layer_tags) {
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
// popBlocksFromCache / blockCacheFree
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, PopBlocksFromCacheReturnsEvictedBatchAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // Seed identical key on both groups, plus a unique key on the full group.
    auto g0_block_for_100 = seedCacheItem(allocator, config, "linear", /*key=*/100);
    auto g1_block_for_100 = seedCacheItem(allocator, config, "full", /*key=*/100);
    auto g1_block_for_200 = seedCacheItem(allocator, config, "full", /*key=*/200);
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
    const auto& g0_blocks = evicted->blocks(/*batch_id=*/0, "linear");
    const auto& g1_blocks = evicted->blocks(/*batch_id=*/0, "full");
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

    seedCacheItem(allocator, config, "linear", /*key=*/100);
    seedCacheItem(allocator, config, "full", /*key=*/200);
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

    auto seeded = seedCacheItem(allocator, config, "full", /*key=*/300);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 1u);

    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(config);
    // Same block listed twice in the same group should only be released once;
    // NULL_BLOCK_IDX entries should be skipped.
    batch->mutableBlockIds(0, "full").assign(BlockIndicesType{seeded, seeded, NULL_BLOCK_IDX});
    EXPECT_NO_THROW(allocator->blockCacheFree(batch));
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Group 0 (linear) gets 6 blocks (5 usable), group 1 (full) gets 4 blocks (3 usable).
    // total_available = 8. Set reserve = 4.
    // Expected per-group reserve: floor(4 * 5/8) = 2 for "linear", floor(4 * 3/8) = 1 for "full".
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
    // Force a group whose available_blocks < need + group_reserve_blocks.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
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

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    constexpr size_t reserve_blocks = 6;
    allocator->setReserveBlocksNum(reserve_blocks);

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ("linear", snapshots[0].tag);
    EXPECT_EQ("full", snapshots[1].tag);
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
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, "full")), 4u);

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
    // "linear" has enough room for the LINEAR tail block; "full" cannot satisfy
    // the 3 FULL blocks needed for seq_len=9. Whichever stage rejects -- the
    // per-group capacity preflight or initMallocForCommonLen's group loop --
    // both pools must end up exactly as they started.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/3, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto counters_before = snapshotPoolCounters(allocator, config);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    // A 3-block pool exposes 2 usable blocks (block 0 is the null sentinel), so 3 FULL blocks can
    // never fit: the per-group total-capacity test must report PERMANENT so the scheduler errors
    // the stream out instead of parking it in WAITING forever.
    EXPECT_EQ(result.status, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, "linear"), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    expectPoolCountersEq(allocator, config, counters_before);
}

// The same request that a live holder makes un-satisfiable must come back RETRYABLE (stream stays
// WAITING) rather than PERMANENT, and must actually succeed once the holder releases its blocks.
TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocReportsRetryablePerGroupCapacityShortage) {
    // Each pool has enough empty-engine capacity for seq_len=8. A live holder
    // leaves the FULL pool one block short, so only the current admission is
    // retryable; the request is not permanently oversized.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto holder_resource = makeBatchResource(/*batch_size=*/1, config);
    holder_resource->setBatchCacheKeys(0, CacheKeysType{100});
    auto       holder_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo holder_info{holder_resource, holder_tokens};
    holder_info.enable_device_cache = false;
    holder_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(holder_info).success);

    auto deferred_resource = makeBatchResource(/*batch_size=*/1, config);
    deferred_resource->setBatchCacheKeys(0, CacheKeysType{200, 201});
    auto       deferred_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo deferred_info{deferred_resource, deferred_tokens};
    deferred_info.enable_device_cache = false;
    deferred_info.reuse_cache         = false;
    deferred_info.verbose             = false;

    auto deferred_result = allocator->malloc(deferred_info);
    EXPECT_FALSE(deferred_result.success);
    EXPECT_EQ(deferred_result.status, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(deferred_resource->curBlocksNum(), 0u);

    allocator->free(FreeInfo{holder_resource, holder_tokens});
    auto retry_result = allocator->malloc(deferred_info);
    EXPECT_TRUE(retry_result.success);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesDeviceReuseReferencesOnReserveReject) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto linear_cached = seedCacheItem(allocator, config, "linear", /*key=*/100);
    const auto full_cached   = seedCacheItem(allocator, config, "full", /*key=*/100);
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    ASSERT_EQ(allocator->requestRefBlocksNum(), 0u);
    ASSERT_EQ(allocator->blockCacheRefBlocksNum(), 2u);

    const size_t available_before = allocator->availableBlocksNum();
    const auto   counters_before  = snapshotPoolCounters(allocator, config);
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
    EXPECT_EQ(batch_res->blocksNum(0, "linear"), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 0u);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 2u);
    expectPoolCountersEq(allocator, config, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeFailingAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 1u);
    const auto linear_block_before = batch_res->blocks(0, "linear")[0];
    const auto full_block_before   = batch_res->blocks(0, "full")[0];
    const auto counters_before     = snapshotPoolCounters(allocator, config);
    const auto device_reuse_before = batch_res->cacheResource(0).deviceReuseBlockNum();
    allocator->failure_enabled     = true;

    // The injected FULL failure occurs only after the preceding LINEAR strategy
    // has run. Both pools have capacity, so this is independent of pool pressure.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 1u);
    EXPECT_EQ(batch_res->blocks(0, "linear")[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, "full")[0], full_block_before);
    EXPECT_EQ(batch_res->cacheResource(0).deviceReuseBlockNum(), device_reuse_before);
    expectPoolCountersEq(allocator, config, counters_before);
    EXPECT_EQ(allocator->observed_tags, (std::vector<std::string>{"linear", "full"}));
    EXPECT_GT(allocator->linear_blocks_at_failure.size(), 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackRestoresLinearBackfilledSlots) {
    // Block 0 is reserved by each pool, so FULL needs three configured blocks
    // to provide the two request blocks used by the initial allocation.
    auto config = makeTinyMultiPoolHybridConfig(
        /*linear_block_num=*/4, /*full_block_num=*/3, CacheGroupType::FULL, /*linear_active_tail_blocks=*/2);
    auto allocator = makeFailingAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto linear_reused = seedCacheItem(allocator, config, "linear", /*key=*/100, /*is_resident=*/true);
    const auto full_reused   = seedCacheItem(allocator, config, "full", /*key=*/100, /*is_resident=*/true);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = true;
    init_info.reuse_cache         = true;
    ASSERT_TRUE(allocator->malloc(init_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2u);
    EXPECT_EQ(batch_res->blocks(0, "linear")[0], linear_reused);
    EXPECT_EQ(batch_res->blocks(0, "full")[0], full_reused);

    const auto keys_before         = batch_res->cacheKeys(0);
    const auto dependencies_before = batch_res->cacheResource(0).blockDependencies();
    const auto device_reuse_before = batch_res->cacheResource(0).deviceReuseBlockNum();
    auto&      linear_ids          = batch_res->mutableBlockIds(0, "linear");
    auto       removed_block_id    = linear_ids.blocks()[1];
    ASSERT_FALSE(isNullBlockIdx(removed_block_id));
    allocator->blockPool("linear")->requestFree({removed_block_id});
    linear_ids.setAt(1, NULL_BLOCK_IDX);
    const auto blocks_before   = std::map<std::string, BlockIndicesType>{{"linear", batch_res->blocks(0, "linear")},
                                                                         {"full", batch_res->blocks(0, "full")}};
    const auto counters_before = snapshotPoolCounters(allocator, config);
    allocator->failure_enabled = true;

    // The resource already owns reused prefix references in both groups.
    // LINEAR first backfills the old sparse tail and appends a new tail block;
    // FULL then fails through the narrow test seam. Rollback must
    // restore the historical NULL slot, original logical length, cache refs,
    // request refs, and dependency timeline by tag.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    EXPECT_FALSE(allocator->malloc(incr_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2u);
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, "linear")[1]));
    EXPECT_EQ(batch_res->blocks(0, "linear"), blocks_before.at("linear"));
    EXPECT_EQ(batch_res->blocks(0, "full"), blocks_before.at("full"));
    EXPECT_EQ(batch_res->cacheResource(0).deviceReuseBlockNum(), device_reuse_before);
    EXPECT_EQ(batch_res->cacheKeys(0), keys_before);
    const auto& dependencies_after = batch_res->cacheResource(0).blockDependencies();
    ASSERT_EQ(dependencies_after.size(), dependencies_before.size());
    for (size_t i = 0; i < dependencies_before.size(); ++i) {
        EXPECT_EQ(dependencies_after[i].has_parent, dependencies_before[i].has_parent);
        EXPECT_EQ(dependencies_after[i].parent_key, dependencies_before[i].parent_key);
        EXPECT_EQ(dependencies_after[i].ordinal, dependencies_before[i].ordinal);
    }
    expectPoolCountersEq(allocator, config, counters_before);
    EXPECT_EQ(allocator->observed_tags, (std::vector<std::string>{"linear", "full"}));
    ASSERT_GE(allocator->linear_blocks_at_failure.size(), 2u);
    EXPECT_FALSE(isNullBlockIdx(allocator->linear_blocks_at_failure[1]));
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
    // Sum of per-pool totals must equal aggregated totalBlocksNum.
    size_t expected_total = 0;
    for (const auto& group : config.groups()) {
        expected_total += allocator->blockPool(group.tag)->totalBlocksNum();
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
    EXPECT_EQ(allocator->availableBlocksNum(), expected_total);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    for (const auto& group : config.groups()) {
        EXPECT_EQ(allocator->blockPool(group.tag)->where(), MemoryType::MEMORY_GPU) << "tag=" << group.tag;
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator     = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const std::string hca_state_tag = "hca_state";
    ASSERT_EQ(groupTagSet(config).count(hca_state_tag), 1u);
    ASSERT_EQ(config.group(hca_state_tag).tag, "hca_state");
    const size_t hca_free_before = allocator->blockPool("hca_state")->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, hca_state_tag);
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - allocator->blockPool("hca_state")->freeBlocksNum(), 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsIgnoreSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    const std::string hca_state_tag = "hca_state";
    ASSERT_EQ(config.group(hca_state_tag).tag, "hca_state");
    auto block_nums           = blockNumsByTag(config);
    block_nums[hca_state_tag] = 2;
    setGroupBlockNums(config, block_nums);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    const auto hca_state_tokens = allocator->blockPool("hca_state")->totalBlocksNum() * config.seq_size_per_block;
    EXPECT_LT(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->maxAvailableTokensNum());
    EXPECT_EQ(allocator->totalTokensNum(), allocator->maxAvailableTokensNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigUsesGroupOwnedBytesForPagedBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, 0);

    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_paged_bytes = 0;
    size_t expected_paged_bytes     = 0;
    for (const auto& group : config.groups()) {
        const auto type = group.policy.group_type;
        const auto expected_group_bytes =
            config.groupLayerIds(group.tag).size() * (group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
        EXPECT_EQ(config.blockSizeBytes(group.tag), expected_group_bytes) << "tag=" << group.tag;
        if (!config.usesExplicitIndependentBlocks(group.tag)
            && (type == CacheGroupType::FULL || type == CacheGroupType::LINEAR)) {
            expected_paged_bytes += expected_group_bytes;
        } else {
            expected_non_paged_bytes += expected_group_bytes;
        }
    }

    EXPECT_GT(expected_non_paged_bytes, 0u);
    EXPECT_GT(expected_paged_bytes, 0u);

    EXPECT_EQ(config.pagedBlockSizeBytes(), expected_paged_bytes);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioExcludesExplicitIndependentPools) {
    auto              config                   = makeDSV4HybridPoolConfig(/*block_num=*/200);
    const std::string explicit_independent_tag = firstExplicitIndependentGroupTag(config);
    ASSERT_FALSE(explicit_independent_tag.empty());
    ASSERT_TRUE(config.usesExplicitIndependentBlocks(explicit_independent_tag));

    constexpr int64_t reserve_ratio = 10;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());

    size_t reservable_available = 0;
    size_t all_available        = 0;
    for (const auto& group : config.groups()) {
        const size_t available = allocator->blockPool(group.tag)->availableBlocksNum();
        all_available += available;
        if (!config.usesExplicitIndependentBlocks(group.tag)) {
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
    auto              config       = makeDSV4HybridPoolConfig(/*block_num=*/50, 50);
    const std::string explicit_tag = firstExplicitIndependentGroupTag(config);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (const auto& group : config.groups()) {
        const uint32_t expected = group.policy.explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }

    const size_t expected_reserve = 50u * config.blockSizeBytes(explicit_tag);
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123, 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (const auto& group : config.groups()) {
        EXPECT_EQ(group.block_num, 123u) << "tag=" << group.tag;
    }
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto              config       = makeDSV4HybridPoolConfig(/*block_num=*/50, 50);
    const std::string explicit_tag = firstExplicitIndependentGroupTag(config);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (const auto& group : config.groups()) {
        const uint32_t expected = group.policy.explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }
    const size_t expected_reserve = 50u * config.blockSizeBytes(explicit_tag);
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4StateSwaPoolsWithoutExplicitBlocksScaleWithLinearStep) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, 0);
    config.linear_step = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (const auto& group : config.groups()) {
        const uint32_t expected = group.policy.group_type == CacheGroupType::SWA ? 32u : 128u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }
    EXPECT_EQ(config.explicitlySizedPoolReserveBytes(), 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, FinalizeNonExplicitSwaBlocksUsesCeilDivision) {
    auto config        = makeTinySwaMultiPoolHybridConfig();
    config.linear_step = 4;
    RuntimeConfig rt;

    config.finalizeBlockNums(/*global_block_num=*/1, rt);
    EXPECT_EQ(config.group("linear").block_num, 1u);
    EXPECT_EQ(config.group("swa").block_num, 1u);

    config.finalizeBlockNums(/*global_block_num=*/8, rt);
    EXPECT_EQ(config.group("linear").block_num, 8u);
    EXPECT_EQ(config.group("swa").block_num, 2u);

    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.group("linear").block_num, 9u);
    EXPECT_EQ(config.group("swa").block_num, 3u);

    config.linear_step = 1;
    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.group("linear").block_num, 9u);
    EXPECT_EQ(config.group("swa").block_num, 9u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        const auto& layer_tags = config.groupsForLayer(static_cast<int>(l));
        if (std::find(layer_tags.begin(), layer_tags.end(), "csa_kv") != layer_tags.end()) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // The tag selects the CSA group's pool directly.
    auto addr_csa = allocator->convertIndexToAddrByTag(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);

    auto addr_swa = allocator->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);
    EXPECT_THROW((void)allocator->convertIndexToAddrByTag(csa_layer, "missing", 1), std::exception);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, /*block_id=*/1), std::exception);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        const auto& layer_tags = config.groupsForLayer(static_cast<int>(l));
        if (std::find(layer_tags.begin(), layer_tags.end(), "csa_kv") != layer_tags.end()) {
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

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4SharedBlockCacheIsUnifiedAcrossGroups) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // All groups share a single SharedBlockCache owned by the allocator.
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Inserting a cache item for one group is visible via the shared cache.
    auto pool0  = allocator->blockPool("swa_kv");
    auto blocks = pool0->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    shared_cache->put(/*cache_key=*/42, {{"swa_kv", blocks[0]}}, /*is_resident=*/false);
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
    // 10 full keys under cp_size=2 subsample to 5 canonical match keys;
    // initMallocForCommonLen always drops the last canonical match key (it may
    // be a partial tail, and fully reusing the input would leave no prefill
    // tokens to compute), so only 4 canonical blocks are reusable.
    EXPECT_EQ(result.reuse_len, 4 * spb * 2);

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

    auto                  seed_res = makeBatchResource(/*batch_size=*/1, config);
    BlockDependenciesType full_dependencies;
    full_dependencies.reserve(full_keys.size());
    for (size_t i = 0; i < full_keys.size(); ++i) {
        full_dependencies.push_back(
            BlockDependency{true, 5000 + static_cast<CacheKeyType>(i), static_cast<uint32_t>(7 + i * 11)});
    }
    seed_res->cacheResource(0).setCacheKeysAndBlockDependencies(full_keys, full_dependencies);
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
    ASSERT_FALSE(evicted->cacheKeys(0).empty());
    const auto& dependencies = evicted->cacheResource(0).blockDependencies();
    ASSERT_EQ(dependencies.size(), evicted->cacheKeys(0).size());
    for (size_t i = 0; i < dependencies.size(); ++i) {
        EXPECT_NE(std::find(expected_canonical.begin(), expected_canonical.end(), evicted->cacheKeys(0)[i]),
                  expected_canonical.end());
        const size_t source_pos = static_cast<size_t>(evicted->cacheKeys(0)[i] - full_keys.front());
        ASSERT_LT(source_pos, full_dependencies.size());
        EXPECT_EQ(dependencies[i].has_parent, full_dependencies[source_pos].has_parent);
        EXPECT_EQ(dependencies[i].parent_key, full_dependencies[source_pos].parent_key);
        EXPECT_EQ(dependencies[i].ordinal, full_dependencies[source_pos].ordinal);
    }
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
