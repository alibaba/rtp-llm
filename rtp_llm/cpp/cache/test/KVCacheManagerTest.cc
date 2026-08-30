#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {
constexpr int kDsv4PoolNum = 7;
// Group storage order is deterministic but carries no business meaning, so the
// expectation is the tag set.
const std::set<std::string> kDsv4Tags = {
    "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            old_value_ = old_value;
            had_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_value_ = false;
};
}  // namespace

class KVCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        rtp_llm::initLogger();
        createDevice();
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

private:
    bool old_core_dump_on_exception_{false};
};

static void assertBlockBytesEq(const std::shared_ptr<rtp_llm::KVCacheManager>& cache_manager,
                               int                                             layer_id,
                               int                                             block_id,
                               const std::vector<int8_t>&                      expected) {
    auto addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
    ASSERT_NE(addr_info.kv_addr, nullptr);
    auto dev_t = torch::from_blob(
        addr_info.kv_addr, {(int64_t)expected.size()}, torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
    auto        host_t = dev_t.cpu();
    const auto* ptr    = host_t.data_ptr<int8_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(ptr[i], expected[i]) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id;
    }
}

static void assertScaleEq(const std::shared_ptr<rtp_llm::KVCacheManager>& cache_manager,
                          int                                             layer_id,
                          int                                             block_id,
                          const std::vector<float>&                       expected_k,
                          const std::vector<float>&                       expected_v) {
    auto addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
    ASSERT_NE(addr_info.kv_scale_addr, nullptr);
    ASSERT_EQ(expected_k.size(), expected_v.size());

    const size_t kv_scale_stride_bytes = cache_manager->cacheConfig().soleGroupForLayer(layer_id).kv_scale_stride_bytes;
    ASSERT_GT(kv_scale_stride_bytes, 0u);
    const size_t kv_scale_block_bytes = kv_scale_stride_bytes / 2;
    void*        v_scale_addr = static_cast<void*>(static_cast<char*>(addr_info.kv_scale_addr) + kv_scale_block_bytes);

    auto dev_k_t = torch::from_blob(addr_info.kv_scale_addr,
                                    {(int64_t)expected_k.size()},
                                    torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto dev_v_t = torch::from_blob(
        v_scale_addr, {(int64_t)expected_v.size()}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));

    auto host_k_t = dev_k_t.cpu();
    auto host_v_t = dev_v_t.cpu();

    const float* k_ptr = host_k_t.data_ptr<float>();
    const float* v_ptr = host_v_t.data_ptr<float>();
    for (size_t i = 0; i < expected_k.size(); ++i) {
        ASSERT_FLOAT_EQ(k_ptr[i], expected_k[i])
            << "k scale mismatch i=" << i << " layer=" << layer_id << " block=" << block_id;
        ASSERT_FLOAT_EQ(v_ptr[i], expected_v[i])
            << "v scale mismatch i=" << i << " layer=" << layer_id << " block=" << block_id;
    }
}

static ModelConfig makeDSV4ManagerFlashModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 43;
    mc.hidden_size                  = 4096;
    mc.attn_config.head_num         = 64;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.tokens_per_block = 128;
    std::vector<int> ratios         = {0, 0};
    for (int i = 2; i < 43; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    setDsv4KvCacheSpecs(mc, ratios);
    return mc;
}

// Override named group capacities through the test-only topology helper.
static void setGroupBlockNumsForTest(CacheConfig& config, const std::map<std::string, uint32_t>& block_nums_by_tag) {
    std::vector<uint32_t> block_nums;
    std::vector<size_t>   kv_strides;
    std::vector<size_t>   scale_strides;
    for (const auto& group : config.groups()) {
        const auto it = block_nums_by_tag.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(it != block_nums_by_tag.end(), "no test block count for tag=%s", group.tag.c_str());
        block_nums.push_back(it->second);
        kv_strides.push_back(group.kv_block_stride_bytes);
        scale_strides.push_back(group.kv_scale_stride_bytes);
    }
    setGroupBlockLayout(config, block_nums, kv_strides, scale_strides);
}

// Same block count for every cache group.
static void setUniformGroupBlockNumsForTest(CacheConfig& config, uint32_t block_num) {
    std::map<std::string, uint32_t> block_nums;
    for (const auto& group : config.groups()) {
        block_nums[group.tag] = block_num;
    }
    setGroupBlockNumsForTest(config, block_nums);
}

static CacheConfig makeTwoLinearGroupManagerConfig() {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = 2;
    config.block_num          = 4;
    config.seq_size_per_block = 2;
    config.linear_step        = 2;

    auto linear0 = makeResolvedLinearSpec(config.dtype,
                                          /*local_num_k_heads=*/1,
                                          /*local_num_v_heads=*/1,
                                          /*head_k_dim=*/2,
                                          /*head_v_dim=*/2,
                                          /*conv_kernel_dim=*/2,
                                          /*seq_size_per_block=*/2,
                                          config.dtype,
                                          config.dtype,
                                          "linear0");
    auto linear1 = makeResolvedLinearSpec(config.dtype,
                                          /*local_num_k_heads=*/1,
                                          /*local_num_v_heads=*/1,
                                          /*head_k_dim=*/2,
                                          /*head_v_dim=*/2,
                                          /*conv_kernel_dim=*/2,
                                          /*seq_size_per_block=*/2,
                                          config.dtype,
                                          config.dtype,
                                          "linear1");
    rtp_llm::test::assignCacheConfigFromGroupedSpecs(config,
                                                     config.layer_num,
                                                     {linear0, linear1},
                                                     {{0}, {1}},
                                                     {CacheGroupType::LINEAR, CacheGroupType::LINEAR},
                                                     {"linear0", "linear1"});
    setGroupBlockNumsForTest(config, {{"linear0", 4}, {"linear1", 4}});
    return config;
}

static CacheConfig makeCompactDSV4ManagerConfig(uint32_t block_num = 16) {
    ParallelismConfig pc;
    auto              mc = makeDSV4ManagerFlashModelConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config      = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, 0);
    config.block_num = block_num;
    setUniformGroupBlockNumsForTest(config, block_num);
    return config;
}

static bool isFullGroup(const CacheConfig& config, std::string_view tag) {
    return config.group(tag).policy.group_type == CacheGroupType::FULL;
}

static bool isFixedTailGroup(const CacheConfig& config, std::string_view tag) {
    return config.group(tag).policy.group_type != CacheGroupType::FULL;
}

static bool isHcaStateGroup(std::string_view tag) {
    return tag == "hca_state";
}

static std::vector<std::string> dsv4GroupTagsByType(const CacheConfig& config, CacheGroupType type) {
    std::vector<std::string> tags;
    for (const auto& group : config.groups()) {
        if (group.policy.group_type == type) {
            tags.push_back(group.tag);
        }
    }
    return tags;
}

static std::vector<std::string> dsv4FixedTailGroupTags(const CacheConfig& config) {
    std::vector<std::string> tags;
    for (const auto& group : config.groups()) {
        if (isFixedTailGroup(config, group.tag)) {
            tags.push_back(group.tag);
        }
    }
    return tags;
}

static int dsv4ActiveTailBlocks(std::string_view tag) {
    return isHcaStateGroup(tag) ? 1 : 2;
}

static void expectDsv4SwaAllocatedBlocks(const CacheConfig&      config,
                                         const BlockIndicesType& blocks,
                                         std::string_view        tag,
                                         const std::string&      label,
                                         bool                    enable_reuse_cache = false) {
    const int  active_tail_blocks = dsv4ActiveTailBlocks(tag);
    const int  tail_begin         = std::max(static_cast<int>(blocks.size()) - active_tail_blocks, 0);
    const int  linear_step        = std::max(1, config.linear_step);
    const bool effective_reuse    = enable_reuse_cache && !isHcaStateGroup(tag);
    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
        const bool should_allocate = i >= tail_begin || (effective_reuse && ((i + 1) % linear_step == 0));
        if (should_allocate) {
            EXPECT_FALSE(isNullBlockIdx(blocks[static_cast<size_t>(i)])) << label << " group " << tag << " pos " << i;
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[static_cast<size_t>(i)])) << label << " group " << tag << " pos " << i;
        }
    }
}

// Creates an intentionally tight DSV4 config for eviction stress tests: FULL
// groups use a large paged pool, while SWA groups use a small independent pool.
static CacheConfig makeDSV4ConfigWithConcurrencyPool(uint32_t full_block_num, uint32_t swa_batch_size) {
    ParallelismConfig pc;
    auto              mc = makeDSV4ManagerFlashModelConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config      = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, 0);
    config.block_num = full_block_num;
    std::map<std::string, uint32_t> block_nums;
    for (const auto& group : config.groups()) {
        block_nums[group.tag] = isFullGroup(config, group.tag) ? full_block_num : (2u * swa_batch_size);
    }
    setGroupBlockNumsForTest(config, block_nums);
    return config;
}

static CacheConfig
makeProductionDSV4Config(uint32_t full_block_num, uint32_t max_concurrency, uint32_t hca_state_pool_blocks = 4) {
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = full_block_num;
    auto mc                        = makeDSV4ManagerFlashModelConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", hca_state_pool_blocks);
    runtime_config.max_generate_batch_size                      = max_concurrency;
    runtime_config.fifo_scheduler_config.max_context_batch_size = max_concurrency;
    return CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);
}

static BatchKVCacheResourcePtr makeDSV4BatchResource(const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(1);
    res->initGroups(config);
    return res;
}

static CompleteTokenIdsPtr makeDSV4CompleteTokenIds(int initial_seq_len, int max_seq_len, int seq_size_per_block) {
    auto input_ids      = torch::arange(max_seq_len, torch::kInt32);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();

    auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, max_seq_len + 16, seq_size_per_block);
    complete_token_ids->init(gi);
    complete_token_ids->setSeqLength(initial_seq_len);
    return complete_token_ids;
}

static CompleteTokenIdsPtr makeSingleManagerCompleteTokenIds(int batch_size, int seq_len, int seq_size_per_block) {
    auto input_ids                  = torch::arange(seq_len, torch::kInt32);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_len + 16, seq_size_per_block);
    complete_token_ids->init(generate_input);
    return complete_token_ids;
}

static BatchKVCacheResourcePtr makeSingleManagerBatchResource(int batch_size, const CacheConfig& config) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(batch_size);
    resource->initGroups(config);
    return resource;
}

struct SingleManagerPoolCounters {
    size_t free_blocks;
    size_t available_blocks;
    size_t request_refs;
    size_t block_cache_refs;
    size_t connector_refs;
};

static SingleManagerPoolCounters snapshotSingleManagerPoolCounters(const BlockPoolPtr& pool) {
    return {pool->freeBlocksNum(),
            pool->availableBlocksNum(),
            pool->requestRefBlocksNum(),
            pool->blockCacheRefBlocksNum(),
            pool->connectorRefBlocksNum()};
}

static void expectSingleManagerPoolCountersEq(const BlockPoolPtr& pool, const SingleManagerPoolCounters& expected) {
    EXPECT_EQ(pool->freeBlocksNum(), expected.free_blocks);
    EXPECT_EQ(pool->availableBlocksNum(), expected.available_blocks);
    EXPECT_EQ(pool->requestRefBlocksNum(), expected.request_refs);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), expected.block_cache_refs);
    EXPECT_EQ(pool->connectorRefBlocksNum(), expected.connector_refs);
}

static CacheConfig cloneManagerConfig(const CacheConfig& source) {
    CacheConfig target(source.groups(), source.layers(), source.layer_num);
    target.use_typed_cache_regions                  = source.use_typed_cache_regions;
    target.use_opaque_kv_cache_store                = source.use_opaque_kv_cache_store;
    target.disable_decode_first_malloc_device_reuse = source.disable_decode_first_malloc_device_reuse;
    target.dtype                                    = source.dtype;
    target.use_mla                                  = source.use_mla;
    target.enable_hybrid_attention                  = source.enable_hybrid_attention;
    target.is_sparse                                = source.is_sparse;
    target.block_num                                = source.block_num;
    target.seq_size_per_block                       = source.seq_size_per_block;
    target.linear_step                              = source.linear_step;
    target.mtp_sub_configs                          = source.mtp_sub_configs;
    return target;
}

static void expectInvalidSharedTopologyRejectedAtAllocatorConstruction(const CacheConfig& config,
                                                                       const std::string& expected_message) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_rank                            = 0;
    parallelism_config.tp_size                            = 2;
    parallelism_config.prefill_cp_config.kv_cache_sharded = true;

    auto manager = std::make_shared<KVCacheManager>(cloneManagerConfig(config),
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    parallelism_config,
                                                    RuntimeConfig{},
                                                    SpeculativeExecutionConfig{},
                                                    PDSepConfig{},
                                                    CacheStoreConfig{},
                                                    /*use_cuda_malloc_block_pool=*/true);
    const_cast<CacheGroup&>(manager->config_.soleGroupForLayer(0)).spec.reset();
    try {
        manager->init();
        FAIL() << "expected invalid shared topology to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos) << e.what();
    }

    EXPECT_EQ(manager->allocator_, nullptr);
}

static void expectOrdinarySingleManagerUsesHybridPoolWithReuseAndRollback(const CacheConfig& roomy_config,
                                                                          const CacheConfig& tight_config,
                                                                          KVCacheSpecType    expected_spec_type) {
    ASSERT_EQ(roomy_config.groupNums(), 1);
    const std::string sole_tag = roomy_config.soleGroupForLayer(0).tag;
    ASSERT_EQ(roomy_config.group(sole_tag).spec->type, expected_spec_type);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reuse_cache         = true;
    kv_cache_config.reserve_block_ratio = 20;
    auto manager                        = std::make_shared<KVCacheManager>(cloneManagerConfig(roomy_config),
                                                    /*warmup=*/false,
                                                    /*metrics_reporter=*/nullptr,
                                                    kv_cache_config);
    ASSERT_TRUE(manager->init());

    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    const auto pool = allocator->blockPool(roomy_config.groups().front().tag);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->sharedBlockCache(), manager->allocator_->sharedBlockCache());
    ASSERT_NE(allocator->sharedBlockCache(), nullptr);
    EXPECT_EQ(manager->reserveBlocksNum(),
              static_cast<size_t>(kv_cache_config.reserve_block_ratio) * allocator->availableBlocksNum()
                  / static_cast<size_t>(100));

    const int  seq_size_per_block = static_cast<int>(roomy_config.seq_size_per_block);
    const int  seq_len            = 3 * seq_size_per_block + 1;
    auto       seed_resource      = makeSingleManagerBatchResource(/*batch_size=*/1, roomy_config);
    auto       seed_tokens        = makeSingleManagerCompleteTokenIds(/*batch_size=*/1, seq_len, seq_size_per_block);
    MallocInfo seed_malloc{seed_resource, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = true;
    ASSERT_TRUE(manager->malloc(seed_malloc).success);
    ASSERT_GE(seed_resource->blocksNum(0, sole_tag), 2u);
    const auto cached_prefix_block = seed_resource->blocks(0, sole_tag)[0];

    manager->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{seed_resource, seed_tokens});
    ASSERT_EQ(allocator->requestRefBlocksNum(), 0u);
    ASSERT_GT(allocator->blockCacheRefBlocksNum(), 0u);

    auto       reuse_resource = makeSingleManagerBatchResource(/*batch_size=*/1, roomy_config);
    auto       reuse_tokens   = makeSingleManagerCompleteTokenIds(/*batch_size=*/1, seq_len, seq_size_per_block);
    MallocInfo reuse_malloc{reuse_resource, reuse_tokens};
    reuse_malloc.enable_device_cache = true;
    reuse_malloc.reuse_cache         = true;
    const auto reuse_result          = manager->malloc(reuse_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_GE(reuse_result.reuse_len, seq_size_per_block);
    ASSERT_FALSE(reuse_resource->blocks(0, sole_tag).empty());
    EXPECT_EQ(reuse_resource->blocks(0, sole_tag)[0], cached_prefix_block);
    manager->free(FreeInfo{reuse_resource, reuse_tokens});
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);

    ASSERT_EQ(tight_config.groupNums(), 1);
    const std::string tight_sole_tag = tight_config.soleGroupForLayer(0).tag;
    ASSERT_EQ(tight_config.group(tight_sole_tag).spec->type, expected_spec_type);
    auto rollback_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(tight_config), /*warmup=*/false);
    ASSERT_TRUE(rollback_manager->init());
    auto rollback_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(rollback_manager->allocator_);
    ASSERT_NE(rollback_allocator, nullptr);
    const auto rollback_pool = rollback_allocator->blockPool(tight_config.groups().front().tag);

    auto target_resource = makeSingleManagerBatchResource(/*batch_size=*/2, tight_config);
    auto target_tokens   = makeSingleManagerCompleteTokenIds(/*batch_size=*/2, seq_size_per_block, seq_size_per_block);
    MallocInfo target_init{target_resource, target_tokens};
    target_init.enable_device_cache = false;
    target_init.reuse_cache         = false;
    ASSERT_TRUE(rollback_manager->malloc(target_init).success);

    auto holder_resource = makeSingleManagerBatchResource(/*batch_size=*/1, tight_config);
    auto holder_tokens   = makeSingleManagerCompleteTokenIds(/*batch_size=*/1, seq_size_per_block, seq_size_per_block);
    MallocInfo holder_init{holder_resource, holder_tokens};
    holder_init.enable_device_cache = false;
    holder_init.reuse_cache         = false;
    ASSERT_TRUE(rollback_manager->malloc(holder_init).success);
    ASSERT_EQ(rollback_pool->freeBlocksNum(), 1u);

    const auto batch0_before   = target_resource->blocks(0, tight_sole_tag);
    const auto batch1_before   = target_resource->blocks(1, tight_sole_tag);
    const auto counters_before = snapshotSingleManagerPoolCounters(rollback_pool);

    target_tokens->setSeqLength(2 * seq_size_per_block);
    MallocInfo target_incr{target_resource, target_tokens};
    target_incr.enable_device_cache = false;
    target_incr.reuse_cache         = false;
    target_incr.verbose             = false;
    const auto failure_result       = rollback_manager->malloc(target_incr);
    EXPECT_FALSE(failure_result.success);
    EXPECT_EQ(failure_result.status, MallocStatus::INTERNAL_ERROR);
    EXPECT_EQ(target_resource->blocks(0, tight_sole_tag), batch0_before);
    EXPECT_EQ(target_resource->blocks(1, tight_sole_tag), batch1_before);
    expectSingleManagerPoolCountersEq(rollback_pool, counters_before);

    rollback_manager->free(FreeInfo{holder_resource, holder_tokens});
    rollback_manager->free(FreeInfo{target_resource, target_tokens});
}

static void writeDsv4RegionPattern(const std::shared_ptr<KVCacheManager>& manager,
                                   int                                    block_id,
                                   int                                    layer_id,
                                   const std::string&                     tag,
                                   size_t                                 bytes,
                                   uint8_t                                pattern) {
    auto addr_info = manager->convertIndexToAddrByTag(block_id, layer_id, tag);
    ASSERT_NE(addr_info.kv_addr, nullptr);

    auto dst =
        torch::from_blob(addr_info.kv_addr, {(int64_t)bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto src = torch::full({(int64_t)bytes}, pattern, torch::TensorOptions(torch::kUInt8).device(torch::kCPU));
    dst.copy_(src);
    runtimeSyncAndCheck();
}

static void assertDsv4RegionPatternEq(const std::shared_ptr<KVCacheManager>& manager,
                                      int                                    block_id,
                                      int                                    layer_id,
                                      const std::string&                     tag,
                                      size_t                                 bytes,
                                      uint8_t                                expected) {
    auto addr_info = manager->convertIndexToAddrByTag(block_id, layer_id, tag);
    ASSERT_NE(addr_info.kv_addr, nullptr);

    auto dev_t =
        torch::from_blob(addr_info.kv_addr, {(int64_t)bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host_t = dev_t.cpu();
    const auto* ptr    = host_t.data_ptr<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        ASSERT_EQ(ptr[i], expected) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id
                                    << " tag=" << tag;
    }
}

TEST_F(KVCacheManagerTest, WarmupConfigSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/true);
    ASSERT_TRUE(cache_manager->init());

    EXPECT_EQ(cache_manager->cacheConfig().block_num, 1);

    EXPECT_EQ(cache_manager->totalBlocksNum(), 0);
    EXPECT_EQ(cache_manager->freeBlocksNum(), 0);
}

TEST_F(KVCacheManagerTest, InitAcceptsSingleLinearGroup) {
    auto cache_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_BF16);

    auto cache_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_NE(std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager->allocator_), nullptr);
}

TEST_F(KVCacheManagerTest, InitRejectsSingleNullSpecAtAllocatorConstruction) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_BF16);
    expectInvalidSharedTopologyRejectedAtAllocatorConstruction(cache_config, "CacheConfig got null spec at group 0");
}

TEST_F(KVCacheManagerTest, InitAcceptsMultiGroupWithoutFullAttention) {
    auto cache_config = makeTwoLinearGroupManagerConfig();

    auto cache_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());
    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    EXPECT_NE(allocator->blockPool("linear0"), allocator->blockPool("linear1"));
}

TEST_F(KVCacheManagerTest, ConstructionRematerializesSynchronizedCapacityIntoMtpViews) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_BF16);
    auto mtp_config = std::make_shared<CacheConfig>(makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/2, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_BF16));
    cache_config.mtp_sub_configs.push_back(mtp_config);

    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/false);

    EXPECT_EQ(cache_manager->cacheConfig().block_num, 4u);
    ASSERT_EQ(cache_manager->cacheConfig().mtp_sub_configs.size(), 1u);
    const auto& synchronized_mtp = cache_manager->cacheConfig().mtp_sub_configs.front();
    ASSERT_NE(synchronized_mtp, nullptr);
    EXPECT_EQ(synchronized_mtp->block_num, 4u);
    EXPECT_EQ(synchronized_mtp->group("default").block_num, 4u);
}

TEST_F(KVCacheManagerTest, IndependentLinearOnlyMultiGroupInsertFreeReusesDeviceCache) {
    auto cache_config = makeTwoLinearGroupManagerConfig();
    auto manager      = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    const auto pool0 = allocator->blockPool("linear0");
    const auto pool1 = allocator->blockPool("linear1");
    ASSERT_NE(pool0, pool1);
    const std::map<std::string, BlockPoolPtr> pools{{"linear0", pool0}, {"linear1", pool1}};

    const int  seq_size_per_block = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int  seq_len            = 3 * seq_size_per_block + 1;
    auto       seed_resource      = makeSingleManagerBatchResource(/*batch_size=*/1, manager->cacheConfig());
    auto       seed_tokens        = makeSingleManagerCompleteTokenIds(/*batch_size=*/1, seq_len, seq_size_per_block);
    MallocInfo seed_malloc{seed_resource, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = true;
    ASSERT_TRUE(manager->malloc(seed_malloc).success);

    ASSERT_EQ(seed_resource->groupNums(), 2);
    ASSERT_EQ(seed_resource->cacheKeys(0).size(), 4u);
    const auto cached_key        = seed_resource->cacheKeys(0)[1];
    const auto parent_key        = seed_resource->cacheKeys(0)[0];
    const auto cached_dependency = seed_resource->cacheResource(0).blockDependencies()[1];
    ASSERT_TRUE(cached_dependency.has_parent);
    EXPECT_EQ(cached_dependency.parent_key, parent_key);
    EXPECT_EQ(cached_dependency.ordinal, 1u);

    std::map<std::string, BlockIdxType> cached_blocks;
    for (const auto& group : manager->cacheConfig().groups()) {
        const auto& tag    = group.tag;
        const auto& blocks = seed_resource->blocks(0, tag);
        ASSERT_EQ(blocks.size(), 4u) << "group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "group " << tag;
        ASSERT_FALSE(isNullBlockIdx(blocks[1])) << "group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "group " << tag;
        ASSERT_FALSE(isNullBlockIdx(blocks[3])) << "group " << tag;
        cached_blocks[tag] = blocks[1];
        EXPECT_EQ(pools.at(tag)->requestRefBlocksNum(), 2u);
        EXPECT_EQ(pools.at(tag)->blockCacheRefBlocksNum(), 0u);
    }

    manager->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
    ASSERT_EQ(allocator->sharedBlockCache()->allCacheKeys(), (CacheKeysType{cached_key}));
    EXPECT_EQ(allocator->sharedBlockCache()->version(), 0);
    for (const auto& [tag, pool] : pools) {
        EXPECT_EQ(pool->requestRefBlocksNum(), 2u) << "group " << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), 1u) << "group " << tag;
    }

    manager->free(FreeInfo{seed_resource, seed_tokens});
    for (const auto& [tag, pool] : pools) {
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u) << "group " << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), 1u) << "group " << tag;
    }

    auto       reuse_resource = makeSingleManagerBatchResource(/*batch_size=*/1, manager->cacheConfig());
    auto       reuse_tokens   = makeSingleManagerCompleteTokenIds(/*batch_size=*/1, seq_len, seq_size_per_block);
    MallocInfo reuse_malloc{reuse_resource, reuse_tokens};
    reuse_malloc.enable_device_cache = true;
    reuse_malloc.reuse_cache         = true;
    const auto reuse_result          = manager->malloc(reuse_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_EQ(reuse_result.reuse_len, 2 * seq_size_per_block);
    EXPECT_EQ(reuse_resource->cacheResource(0).deviceReuseBlockNum(), 2);
    ASSERT_EQ(allocator->sharedBlockCache()->allCacheKeys(), (CacheKeysType{cached_key}));
    for (const auto& group : cache_config.groups()) {
        const auto& tag    = group.tag;
        const auto& blocks = reuse_resource->blocks(0, tag);
        ASSERT_EQ(blocks.size(), 4u) << "group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "group " << tag;
        EXPECT_EQ(blocks[1], cached_blocks.at(tag)) << "group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "group " << tag;
        EXPECT_FALSE(isNullBlockIdx(blocks[3])) << "group " << tag;
        EXPECT_EQ(pools.at(tag)->requestRefBlocksNum(), 2u);
        EXPECT_EQ(pools.at(tag)->blockCacheRefBlocksNum(), 1u);
    }

    manager->free(FreeInfo{reuse_resource, reuse_tokens});
    for (const auto& [tag, pool] : pools) {
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u) << "group " << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), 1u) << "group " << tag;
    }

    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/1);
    ASSERT_NE(evicted, nullptr);
    ASSERT_EQ(evicted->cacheKeys(0), (CacheKeysType{cached_key}));
    ASSERT_EQ(evicted->cacheResource(0).blockDependencies().size(), 1u);
    const auto evicted_dependency = evicted->cacheResource(0).blockDependencies()[0];
    EXPECT_EQ(evicted_dependency.has_parent, cached_dependency.has_parent);
    EXPECT_EQ(evicted_dependency.parent_key, cached_dependency.parent_key);
    EXPECT_EQ(evicted_dependency.ordinal, cached_dependency.ordinal);
    for (const auto& group : manager->cacheConfig().groups()) {
        const auto& tag = group.tag;
        ASSERT_EQ(evicted->blocks(0, tag).size(), 1u) << "group " << tag;
        EXPECT_EQ(evicted->blocks(0, tag)[0], cached_blocks.at(tag)) << "group " << tag;
    }
    EXPECT_TRUE(allocator->sharedBlockCache()->allCacheKeys().empty());
    manager->blockCacheFree(evicted);
    for (const auto& [tag, pool] : pools) {
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u) << "group " << tag;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), 0u) << "group " << tag;
    }
}

TEST_F(KVCacheManagerTest, InitAcceptsFullAndLinearGroups) {
    auto cache_config = makeSimpleHybridMhaCacheConfig(
        /*layer_num=*/4, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_BF16);

    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());
    EXPECT_NE(cache_manager->convertIndexToAddr(/*block_index=*/1, /*layer_id=*/0).kv_addr, nullptr);
    EXPECT_NE(cache_manager->convertIndexToAddr(/*block_index=*/1, /*layer_id=*/3).kv_addr, nullptr);
}

TEST_F(KVCacheManagerTest, ProductionHybridConfigUsesHybridPoolWithDistinctPhysicalPools) {
    ModelConfig model_config;
    model_config.num_layers                                      = 4;
    model_config.max_seq_len                                     = 64;
    model_config.data_type                                       = DataType::TYPE_FP16;
    model_config.attn_config.head_num                            = 2;
    model_config.attn_config.kv_head_num                         = 2;
    model_config.attn_config.size_per_head                       = 16;
    model_config.attn_config.tokens_per_block                    = 4;
    model_config.hybrid_attention_config.enable_hybrid_attention = true;
    model_config.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    model_config.linear_attention_config.linear_conv_kernel_dim = 2;
    model_config.linear_attention_config.linear_key_head_dim    = 8;
    model_config.linear_attention_config.linear_value_head_dim  = 8;
    model_config.linear_attention_config.linear_num_key_heads   = 2;
    model_config.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(model_config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 6;
    auto cache_config =
        CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
    auto cache_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), /*warmup=*/false);

    ASSERT_TRUE(cache_manager->init());
    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    ASSERT_EQ(cache_config.groups().size(), 2u);
    EXPECT_NE(allocator->blockPool(cache_config.groups()[0].tag), allocator->blockPool(cache_config.groups()[1].tag));
}

TEST_F(KVCacheManagerTest, OrdinarySingleMhaUsesHybridPoolWithReuseAndRollback) {
    expectOrdinarySingleManagerUsesHybridPoolWithReuseAndRollback(makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                                                           /*block_num=*/10,
                                                                                           /*tokens_per_block=*/4,
                                                                                           DataType::TYPE_FP16,
                                                                                           /*local_head_num_kv=*/2,
                                                                                           /*size_per_head=*/8),
                                                                  makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                                                           /*block_num=*/4,
                                                                                           /*tokens_per_block=*/4,
                                                                                           DataType::TYPE_FP16,
                                                                                           /*local_head_num_kv=*/2,
                                                                                           /*size_per_head=*/8),
                                                                  KVCacheSpecType::MultiHeadAttention);
}

TEST_F(KVCacheManagerTest, OrdinarySingleMlaUsesHybridPoolWithReuseAndRollback) {
    auto make_config = [](int block_num) {
        auto spec = makeResolvedMlaSpec(DataType::TYPE_FP16,
                                        /*kv_lora_rank=*/8,
                                        /*rope_head_dim=*/4,
                                        /*seq_size_per_block=*/4,
                                        /*tag=*/"default");
        return makeSingleGroupCacheConfig(std::move(spec), CacheGroupType::FULL, /*layer_num=*/2, block_num, "default");
    };
    expectOrdinarySingleManagerUsesHybridPoolWithReuseAndRollback(
        make_config(/*block_num=*/10), make_config(/*block_num=*/4), KVCacheSpecType::MultiHeadLatentAttention);
}

TEST_F(KVCacheManagerTest, OrdinarySinglePreservesHybridPoolConstructorAndCudaBacking) {
    auto cache_config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                 /*block_num=*/4,
                                                 /*tokens_per_block=*/2,
                                                 DataType::TYPE_FP16,
                                                 /*local_head_num_kv=*/2,
                                                 /*size_per_head=*/8);

    auto metrics_tags = kmonitor::MetricsTags();
    auto reporter     = std::make_shared<kmonitor::MetricsReporter>("", "", metrics_tags);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 20;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;

    auto manager = std::make_shared<KVCacheManager>(std::move(cache_config),
                                                    /*warmup=*/false,
                                                    reporter,
                                                    kv_cache_config,
                                                    ParallelismConfig{},
                                                    RuntimeConfig{},
                                                    SpeculativeExecutionConfig{},
                                                    pd_sep_config,
                                                    CacheStoreConfig{},
                                                    /*use_cuda_malloc_block_pool=*/true);
    ASSERT_TRUE(manager->init());

    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->allocation_type_, AllocationType::DEVICE);
    EXPECT_EQ(allocator->metrics_reporter_, reporter);
    EXPECT_EQ(allocator->role_type_, RoleType::DECODE);
    EXPECT_TRUE(allocator->use_cuda_malloc_block_pool_);
    const auto pool = allocator->blockPool(cache_config.groups().front().tag);
    ASSERT_NE(pool, nullptr);
    EXPECT_TRUE(pool->use_cuda_malloc_backing_);
}

TEST_F(KVCacheManagerTest, OrdinarySinglePreservesLegacyBatchZeroForwardInsertionAndEvictionTrace) {
    auto          cache_config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                 /*block_num=*/8,
                                                 /*tokens_per_block=*/2,
                                                 DataType::TYPE_FP16,
                                                 /*local_head_num_kv=*/2,
                                                 /*size_per_head=*/8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reuse_cache            = true;
    kv_cache_config.reserve_block_ratio    = 0;
    kv_cache_config.enable_gpu_prefix_tree = false;
    auto manager                           = std::make_shared<KVCacheManager>(std::move(cache_config),
                                                    /*warmup=*/false,
                                                    /*metrics_reporter=*/nullptr,
                                                    kv_cache_config);
    ASSERT_TRUE(manager->init());

    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(allocator, nullptr);
    const auto pool         = allocator->soleGroupBlockPool();
    const auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);
    ASSERT_FALSE(shared_cache->prefixTreeEnabled());

    const auto blocks = pool->malloc(/*num_blocks=*/6);
    ASSERT_EQ(blocks.size(), 6u);
    auto              resource = makeSingleManagerBatchResource(/*batch_size=*/2, manager->cacheConfig());
    const std::string sole_tag = manager->cacheConfig().soleGroupForLayer(0).tag;
    resource->setBatchBlocks(/*batch_id=*/0, sole_tag, {blocks[0], blocks[1], blocks[2]});
    resource->setBatchBlocks(/*batch_id=*/1, sole_tag, {blocks[3], blocks[4], blocks[5]});
    resource->cacheResource(0).setCacheKeysAndBlockDependencies(
        {100, 101, 102},
        {BlockDependency{false, 0, 7}, BlockDependency{true, 100, 11}, BlockDependency{true, 101, 13}});
    resource->cacheResource(1).setCacheKeysAndBlockDependencies(
        {200, 201, 202},
        {BlockDependency{false, 0, 17}, BlockDependency{true, 200, 23}, BlockDependency{true, 201, 29}});
    auto tokens = makeSingleManagerCompleteTokenIds(/*batch_size=*/2, /*seq_len=*/6, /*seq_size_per_block=*/2);

    ASSERT_EQ(pool->requestRefBlocksNum(), 6u);
    ASSERT_EQ(pool->blockCacheRefBlocksNum(), 0u);
    manager->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});

    EXPECT_EQ(shared_cache->allCacheKeys(), (CacheKeysType{101, 100}));
    EXPECT_EQ(shared_cache->version(), 1);
    EXPECT_FALSE(shared_cache->contains(200));
    EXPECT_FALSE(shared_cache->contains(201));
    EXPECT_EQ(pool->requestRefBlocksNum(), 6u);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 2u);

    manager->free(FreeInfo{resource, tokens});
    EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 2u);

    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/1);
    ASSERT_NE(evicted, nullptr);
    ASSERT_EQ(evicted->cacheKeys(0), (CacheKeysType{100}));
    ASSERT_EQ(evicted->cacheResource(0).blockDependencies().size(), 1u);
    const auto dependency = evicted->cacheResource(0).blockDependencies()[0];
    EXPECT_FALSE(dependency.has_parent);
    EXPECT_EQ(dependency.parent_key, 0);
    EXPECT_EQ(dependency.ordinal, 0u);
    EXPECT_EQ(shared_cache->allCacheKeys(), (CacheKeysType{101}));
    EXPECT_EQ(shared_cache->version(), 1);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 2u);

    manager->blockCacheFree(evicted);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 1u);
    auto remaining = manager->popBlocksFromCache(/*min_blocks_to_free=*/8);
    ASSERT_NE(remaining, nullptr);
    manager->blockCacheFree(remaining);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
}

TEST_F(KVCacheManagerTest, MultiGroupRemoteFailsBeforeAllocatorInitialization) {
    auto cache_config = makeSimpleHybridMhaCacheConfig(
        /*layer_num=*/4, /*block_num=*/6, /*tokens_per_block=*/2, DataType::TYPE_BF16);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reuse_cache         = true;
    kv_cache_config.enable_remote_cache = true;
    auto cache_manager                  = std::make_shared<KVCacheManager>(std::move(cache_config),
                                                          /*warmup=*/false,
                                                          nullptr,
                                                          kv_cache_config,
                                                          ParallelismConfig{},
                                                          RuntimeConfig{});

    EXPECT_THROW(cache_manager->init(), std::runtime_error);
    EXPECT_EQ(cache_manager->allocator_, nullptr);
    EXPECT_EQ(cache_manager->coordinator_, nullptr);
}

TEST_F(KVCacheManagerTest, DSV4IndependentPoolsUseGpuBacking) {
    auto expect_pool_backing = [](RoleType role_type) {
        auto config = makeCompactDSV4ManagerConfig(/*block_num=*/8);

        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        KVCacheConfig kv_cache_config;
        auto          cache_manager = std::make_shared<KVCacheManager>(std::move(config),
                                                              /*warmup=*/false,
                                                              nullptr,
                                                              kv_cache_config,
                                                              ParallelismConfig{},
                                                              RuntimeConfig{},
                                                              SpeculativeExecutionConfig{},
                                                              pd_sep_config);
        ASSERT_TRUE(cache_manager->init());

        auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager->allocator_);
        ASSERT_NE(allocator, nullptr);
        for (const auto& group : config.groups()) {
            EXPECT_EQ(allocator->blockPool(group.tag)->where(), MemoryType::MEMORY_GPU)
                << "role=" << static_cast<int>(role_type) << " tag=" << group.tag;
        }
    };

    expect_pool_backing(RoleType::PREFILL);
    expect_pool_backing(RoleType::DECODE);
    expect_pool_backing(RoleType::PDFUSION);
}

TEST_F(KVCacheManagerTest, MetricsThreadSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);

    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/true, reporter);

    ASSERT_TRUE(cache_manager->init());
    EXPECT_TRUE(cache_manager->metrics_reporter_thread_.joinable());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    cache_manager.reset();
}

TEST_F(KVCacheManagerTest, SetKVBlockValueAndBlockCopy) {
    // Use non-warmup config so we have usable blocks.
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().soleGroupForLayer(0).spec;
    const size_t k_bytes = spec->k_block_size_bytes();
    const size_t v_bytes = spec->v_block_size_bytes();
    ASSERT_GT(k_bytes, 0u);
    ASSERT_GT(v_bytes, 0u);

    const int block_src = 1;
    const int block_dst = 3;

    std::vector<int8_t> k_vec(k_bytes, 7);
    std::vector<int8_t> v_vec(v_bytes, 9);
    auto                k_t = torch::from_blob(k_vec.data(), {(int64_t)k_bytes}, torch::kInt8).clone();
    auto                v_t = torch::from_blob(v_vec.data(), {(int64_t)v_bytes}, torch::kInt8).clone();

    ASSERT_TRUE(cache_manager->writeKVBlockForTest(block_src, k_t, v_t));

    std::vector<int8_t> expected_block(k_bytes + v_bytes, 0);
    std::fill(expected_block.begin(), expected_block.begin() + k_bytes, 7);
    std::fill(expected_block.begin() + k_bytes, expected_block.end(), 9);

    // Check both layers in source block
    assertBlockBytesEq(cache_manager, /*layer_id=*/0, block_src, expected_block);
    assertBlockBytesEq(cache_manager, /*layer_id=*/1, block_src, expected_block);

    // Copy src -> dst and validate
    cache_manager->blockCopy(block_src, block_dst);
    assertBlockBytesEq(cache_manager, /*layer_id=*/0, block_dst, expected_block);
    assertBlockBytesEq(cache_manager, /*layer_id=*/1, block_dst, expected_block);

    // Now overwrite only layer 0 on dst block; layer 1 should remain unchanged.
    std::vector<int8_t> k2_vec(k_bytes, 1);
    std::vector<int8_t> v2_vec(v_bytes, 2);
    auto                k2_t = torch::from_blob(k2_vec.data(), {(int64_t)k_bytes}, torch::kInt8).clone();
    auto                v2_t = torch::from_blob(v2_vec.data(), {(int64_t)v_bytes}, torch::kInt8).clone();
    ASSERT_TRUE(cache_manager->writeKVBlockForTest(block_dst, /*layer_id=*/0, k2_t, v2_t));

    std::vector<int8_t> expected_layer0(k_bytes + v_bytes, 0);
    std::fill(expected_layer0.begin(), expected_layer0.begin() + k_bytes, 1);
    std::fill(expected_layer0.begin() + k_bytes, expected_layer0.end(), 2);
    assertBlockBytesEq(cache_manager, /*layer_id=*/0, block_dst, expected_layer0);
    assertBlockBytesEq(cache_manager, /*layer_id=*/1, block_dst, expected_block);
}

TEST_F(KVCacheManagerTest, BlockCopyAlsoCopiesScaleWhenQuantized) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    const int    block_src   = 1;
    const int    block_dst   = 4;
    const size_t scale_elems = 2;  // local_head_num_kv(=1) * tokens_per_block(=2)

    std::vector<float> src_k = {0.5f, 0.6f};
    std::vector<float> src_v = {1.5f, 1.6f};
    ASSERT_EQ(src_k.size(), scale_elems);
    ASSERT_EQ(src_v.size(), scale_elems);

    for (int layer_id = 0; layer_id < 2; ++layer_id) {
        auto addr = cache_manager->convertIndexToAddr(block_src, layer_id);
        ASSERT_NE(addr.kv_scale_addr, nullptr);

        auto host_k_t = torch::tensor(src_k, torch::kFloat32);
        auto host_v_t = torch::tensor(src_v, torch::kFloat32);

        const size_t kv_scale_stride_bytes =
            cache_manager->cacheConfig().soleGroupForLayer(layer_id).kv_scale_stride_bytes;
        ASSERT_GT(kv_scale_stride_bytes, 0u);
        const size_t kv_scale_block_bytes = kv_scale_stride_bytes / 2;
        void*        v_scale_addr = static_cast<void*>(static_cast<char*>(addr.kv_scale_addr) + kv_scale_block_bytes);

        auto dst_k_t = torch::from_blob(
            addr.kv_scale_addr, {(int64_t)scale_elems}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        auto dst_v_t = torch::from_blob(
            v_scale_addr, {(int64_t)scale_elems}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));

        CopyParams cp_k{dst_k_t, host_k_t};
        CopyParams cp_v{dst_v_t, host_v_t};
        runtimeCopy(cp_k);
        runtimeCopy(cp_v);
    }
    runtimeSyncAndCheck();

    // Copy should include both K/V scales.
    cache_manager->blockCopy(block_src, block_dst);
    runtimeSyncAndCheck();

    for (int layer_id = 0; layer_id < 2; ++layer_id) {
        assertScaleEq(cache_manager, layer_id, block_dst, src_k, src_v);
    }
}

TEST_F(KVCacheManagerTest, BlockBatchCopy) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/10, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().soleGroupForLayer(0).spec;
    const size_t k_bytes = spec->k_block_size_bytes();
    const size_t v_bytes = spec->v_block_size_bytes();

    const int src_blocks_num = 2;
    const int dst_blocks_num = 4;

    // Initialize src blocks with distinct patterns.
    for (int i = 0; i < src_blocks_num; ++i) {
        const int           block_id = 1 + i;
        std::vector<int8_t> k_vec(k_bytes, static_cast<int8_t>(block_id));
        std::vector<int8_t> v_vec(v_bytes, static_cast<int8_t>(block_id + 10));
        auto                k_t = torch::from_blob(k_vec.data(), {(int64_t)k_bytes}, torch::kInt8).clone();
        auto                v_t = torch::from_blob(v_vec.data(), {(int64_t)v_bytes}, torch::kInt8).clone();
        ASSERT_TRUE(cache_manager->writeKVBlockForTest(block_id, k_t, v_t));
    }

    std::vector<BlockIdPair> mapping;
    mapping.reserve(dst_blocks_num);
    for (int j = 0; j < dst_blocks_num; ++j) {
        const int dst_block = 1 + src_blocks_num + j;
        const int src_block = 1 + (j % src_blocks_num);
        mapping.push_back({src_block, dst_block});
    }

    cache_manager->blockBatchCopy(mapping);

    // Validate copied blocks for both layers.
    for (int j = 0; j < dst_blocks_num; ++j) {
        const int dst_block = 1 + src_blocks_num + j;
        const int src_block = 1 + (j % src_blocks_num);

        std::vector<int8_t> expected(k_bytes + v_bytes, 0);
        std::fill(expected.begin(), expected.begin() + k_bytes, static_cast<int8_t>(src_block));
        std::fill(expected.begin() + k_bytes, expected.end(), static_cast<int8_t>(src_block + 10));

        assertBlockBytesEq(cache_manager, /*layer_id=*/0, dst_block, expected);
        assertBlockBytesEq(cache_manager, /*layer_id=*/1, dst_block, expected);
    }
}

TEST_F(KVCacheManagerTest, DSV4MallocIncrFreeExposesSevenTypedRegions) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    const int    spb         = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    auto         resource    = makeDSV4BatchResource(manager->cacheConfig());
    auto         tokens      = makeDSV4CompleteTokenIds(/*initial_seq_len=*/2 * spb + 17,
                                           /*max_seq_len=*/4 * spb + 32,
                                           spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    auto malloc_result              = manager->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(resource->groupNums(), kDsv4PoolNum);

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(resource->blocksNum(0, group.tag), 3) << "group " << group.tag;
        const auto& blocks = resource->blocks(0, group.tag);
        if (isFullGroup(manager->cacheConfig(), group.tag)) {
            EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "paged group " << group.tag;
            EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "paged group " << group.tag;
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "paged group " << group.tag;
        } else {
            expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), blocks, group.tag, "tail group");
        }
    }

    tokens->setSeqLength(4 * spb);
    MallocInfo incr_info{resource, tokens};
    incr_info.reuse_cache         = false;
    incr_info.enable_device_cache = false;
    auto incr_result              = manager->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    for (const auto& group : manager_config.groups()) {
        EXPECT_EQ(resource->blocksNum(0, group.tag), 4) << "group " << group.tag;
    }

    auto layout = manager->getMainModelCacheLayerLayout();
    ASSERT_EQ(layout.groups().size(), static_cast<size_t>(kDsv4PoolNum));
    std::set<std::string> layout_tags;
    for (const auto& [tag, group_layout] : layout.groups()) {
        (void)group_layout;
        layout_tags.insert(tag);
    }
    EXPECT_EQ(layout_tags, kDsv4Tags);
    EXPECT_EQ(manager_config.layers().size(), static_cast<size_t>(manager_config.layer_num));

    const int csa_layer = manager_config.groupLayerIds("csa_kv")[0];
    const int hca_layer = manager_config.groupLayerIds("hca_kv")[0];
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "csa_kv")[0], csa_layer, "csa_kv").kv_addr, nullptr);
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "indexer_kv")[0], csa_layer, "indexer_kv").kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "csa_state")[2], csa_layer, "csa_state").kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "hca_state").back(), hca_layer, "hca_state").kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "hca_kv")[0], hca_layer, "hca_kv").kv_addr, nullptr);
    EXPECT_NE(manager->convertIndexToAddrByTag(resource->blocks(0, "swa_kv")[2], csa_layer, "swa_kv").kv_addr, nullptr);
    EXPECT_ANY_THROW((void)manager->convertIndexToAddrByTag(resource->blocks(0, "hca_kv")[0], csa_layer, "hca_kv"));

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

TEST_F(KVCacheManagerTest, DSV4LayerRegionBlockTablesMatchInferenceAccessPattern) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    auto      resource = makeDSV4BatchResource(manager->cacheConfig());
    auto      tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/3 * spb + 17,
                                           /*max_seq_len=*/4 * spb + 32,
                                           spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_info).success);

    // A tag declared by a layer must resolve to that layer's group, and the
    // per-layer block view must be the same storage as the whole-group view.
    auto expectTagGroup = [&](int layer_id, const std::string& tag) {
        EXPECT_EQ(manager->cacheConfig().groupForLayer(layer_id, tag).tag, tag)
            << "layer=" << layer_id << " tag=" << tag;
        EXPECT_TRUE(resource->layerOwnsTag(/*batch_id=*/0, layer_id, tag)) << "layer=" << layer_id << " tag=" << tag;
        EXPECT_EQ(resource->blocksForLayer(/*batch_id=*/0, layer_id, tag), resource->blocks(0, tag))
            << "layer=" << layer_id << " tag=" << tag;
    };

    // Flash DSV4 layers 0/1 are SWA-only. Inference resolves typed block tables by semantic tag.
    expectTagGroup(/*layer_id=*/0, "swa_kv");
    EXPECT_THROW((void)manager->cacheConfig().groupForLayer(/*layer_id=*/0, "csa_kv"), std::exception);
    EXPECT_THROW((void)manager->cacheConfig().groupForLayer(/*layer_id=*/0, "hca_kv"), std::exception);

    // Layer 2 is CSA: CSA_KV + INDEXER_KV + INDEXER_STATE + CSA_STATE + SWA_KV.
    const int csa_layer = manager->cacheConfig().groupLayerIds("csa_kv")[0];
    expectTagGroup(csa_layer, "csa_kv");
    expectTagGroup(csa_layer, "indexer_kv");
    expectTagGroup(csa_layer, "indexer_state");
    expectTagGroup(csa_layer, "csa_state");
    expectTagGroup(csa_layer, "swa_kv");
    EXPECT_THROW((void)manager->cacheConfig().groupForLayer(csa_layer, "hca_kv"), std::exception);

    // Layer 3 is HCA: HCA_KV + HCA_STATE + SWA_KV.
    const int hca_layer = manager->cacheConfig().groupLayerIds("hca_kv")[0];
    expectTagGroup(hca_layer, "hca_kv");
    expectTagGroup(hca_layer, "hca_state");
    expectTagGroup(hca_layer, "swa_kv");
    EXPECT_THROW((void)manager->cacheConfig().groupForLayer(hca_layer, "csa_kv"), std::exception);

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
}

TEST_F(KVCacheManagerTest, DSV4BlockCopyPreservesTypedRegionBytes) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/8);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int seq_len  = 3 * spb + 1;
    auto      resource = makeDSV4BatchResource(manager->cacheConfig());
    auto      tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_info).success);

    const int csa_layer      = manager_config.groupLayerIds("csa_kv")[0];
    const int hca_layer      = manager_config.groupLayerIds("hca_kv")[0];
    const int swa_only_layer = 0;

    auto hybrid_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(manager->allocator_);
    ASSERT_NE(hybrid_allocator, nullptr);

    std::map<std::string, std::pair<BlockIdxType, BlockIdxType>> copy_blocks_by_tag;
    for (const auto& group : manager_config.groups()) {
        const auto& blocks = resource->blocks(0, group.tag);
        const auto  src_it =
            std::find_if(blocks.begin(), blocks.end(), [](BlockIdxType id) { return !isNullBlockIdx(id); });
        ASSERT_NE(src_it, blocks.end()) << "group " << group.tag;

        const auto dst_blocks = hybrid_allocator->blockPool(group.tag)->malloc(1);
        ASSERT_EQ(dst_blocks.size(), 1u) << "group " << group.tag;
        ASSERT_NE(*src_it, dst_blocks[0]) << "group " << group.tag;
        copy_blocks_by_tag.emplace(group.tag, std::make_pair(*src_it, dst_blocks[0]));
    }

    struct RegionCase {
        std::string tag;
        int         layer_id;
        uint8_t     pattern;
    };

    const std::vector<RegionCase> cases = {
        {"swa_kv", csa_layer, 0x11},
        {"csa_kv", csa_layer, 0x22},
        {"indexer_kv", csa_layer, 0x33},
        {"indexer_state", csa_layer, 0x44},
        {"csa_state", csa_layer, 0x55},
        {"hca_kv", hca_layer, 0x66},
        {"hca_state", hca_layer, 0x77},
        {"swa_kv", swa_only_layer, 0x88},
    };

    for (const auto& region_case : cases) {
        const auto [src_block, dst_block] = copy_blocks_by_tag.at(region_case.tag);
        const size_t bytes                = manager->cacheConfig().group(region_case.tag).spec->block_size_bytes();
        ASSERT_GT(bytes, 0u);
        writeDsv4RegionPattern(manager, src_block, region_case.layer_id, region_case.tag, bytes, region_case.pattern);
        writeDsv4RegionPattern(manager, dst_block, region_case.layer_id, region_case.tag, bytes, 0);
        assertDsv4RegionPatternEq(
            manager, src_block, region_case.layer_id, region_case.tag, bytes, region_case.pattern);
        assertDsv4RegionPatternEq(manager, dst_block, region_case.layer_id, region_case.tag, bytes, 0);
    }

    std::vector<TaggedBlockIdPair> copy_mapping;
    copy_mapping.reserve(manager->cacheConfig().groups().size());
    for (const auto& group : manager->cacheConfig().groups()) {
        const auto [src_block, dst_block] = copy_blocks_by_tag.at(group.tag);
        copy_mapping.push_back({group.tag, src_block, dst_block});
    }
    manager->blockBatchCopyByTag(copy_mapping);
    runtimeSyncAndCheck();

    for (const auto& region_case : cases) {
        const auto [src_block, dst_block] = copy_blocks_by_tag.at(region_case.tag);
        (void)src_block;
        const size_t bytes = manager_config.group(region_case.tag).spec->block_size_bytes();
        assertDsv4RegionPatternEq(
            manager, dst_block, region_case.layer_id, region_case.tag, bytes, region_case.pattern);
    }

    for (const auto& group : manager_config.groups()) {
        const auto [src_block, dst_block] = copy_blocks_by_tag.at(group.tag);
        (void)src_block;
        hybrid_allocator->blockPool(group.tag)->requestFree(dst_block);
    }

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
}

TEST_F(KVCacheManagerTest, DSV4InsertIntoDeviceBlockCacheThenReuseSamePrefix) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb     = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int seq_len = 3 * spb + 17;

    auto first_resource = makeDSV4BatchResource(manager->cacheConfig());
    auto first_tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo first_malloc{first_resource, first_tokens};
    first_malloc.reuse_cache         = true;
    first_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(first_malloc).success);

    std::map<std::string, BlockIndicesType> first_blocks;
    for (const auto& group : manager->cacheConfig().groups()) {
        first_blocks[group.tag] = first_resource->blocks(0, group.tag);
    }

    InsertInfo insert_info{first_resource, first_tokens, /*is_resident=*/false};
    manager->insertIntoCache(insert_info);

    FreeInfo first_free{first_resource, first_tokens};
    manager->free(first_free);

    auto second_resource = makeDSV4BatchResource(manager->cacheConfig());
    auto second_tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo second_malloc{second_resource, second_tokens};
    second_malloc.reuse_cache         = true;
    second_malloc.enable_device_cache = true;
    auto reuse_result                 = manager->malloc(second_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_GE(reuse_result.reuse_len, spb);

    for (const auto& tag : dsv4GroupTagsByType(manager->cacheConfig(), CacheGroupType::FULL)) {
        ASSERT_GE(second_resource->blocksNum(0, tag), 3) << "paged group " << tag;
        EXPECT_EQ(second_resource->blocks(0, tag)[0], first_blocks.at(tag)[0]);
        EXPECT_EQ(second_resource->blocks(0, tag)[1], first_blocks.at(tag)[1]);
    }
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        if (manager->cacheConfig().group(tag).policy.enable_prefix_reuse == false) {
            continue;
        }
        ASSERT_GE(second_resource->blocksNum(0, tag), 3) << "tail group " << tag;
        EXPECT_EQ(second_resource->blocks(0, tag)[2], first_blocks.at(tag)[2]);
    }

    FreeInfo second_free{second_resource, second_tokens};
    manager->free(second_free);
}

TEST_F(KVCacheManagerTest, DSV4InitReuseKeepsSWAPrefixTailBlock) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/64);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb = static_cast<int>(manager->cacheConfig().seq_size_per_block);

    auto first_resource = makeDSV4BatchResource(manager->cacheConfig());
    auto first_tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/4 * spb, /*max_seq_len=*/4 * spb + 1, spb);

    MallocInfo first_malloc{first_resource, first_tokens};
    first_malloc.reuse_cache         = false;
    first_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(first_malloc).success);

    std::map<std::string, BlockIdxType> first_swa_tail_blocks;
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        ASSERT_EQ(first_resource->blocksNum(0, tag), 4) << "first SWA group " << tag;
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), first_resource->blocks(0, tag), tag, "first SWA");
        first_swa_tail_blocks[tag] = first_resource->blocks(0, tag)[3];
    }

    // Simulate one generated token before inserting into the device cache, so
    // the fourth full block is cached and can be reused by the next prefill.
    first_tokens->setSeqLength(4 * spb + 1);
    manager->insertIntoCache(InsertInfo{first_resource, first_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{first_resource, first_tokens});

    auto second_resource = makeDSV4BatchResource(manager->cacheConfig());
    auto second_tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/24 * spb, /*max_seq_len=*/24 * spb, spb);

    MallocInfo second_malloc{second_resource, second_tokens};
    second_malloc.reuse_cache                  = true;
    second_malloc.enable_device_cache          = true;
    second_malloc.enable_remove_skipped_blocks = false;
    auto reuse_result                          = manager->malloc(second_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_EQ(reuse_result.reuse_len, 4 * spb);

    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        if (manager->cacheConfig().group(tag).policy.enable_prefix_reuse == false) {
            continue;
        }
        const auto& blocks = second_resource->blocks(0, tag);
        ASSERT_EQ(blocks.size(), 24u) << "second SWA group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "SWA reuse prefix penultimate block is NULL (no prev lookup)";
        EXPECT_EQ(blocks[3], first_swa_tail_blocks.at(tag)) << "SWA reuse prefix tail block must stay readable";
        EXPECT_FALSE(isNullBlockIdx(blocks[22])) << "second SWA group " << tag << " fresh tail block 22";
        EXPECT_FALSE(isNullBlockIdx(blocks[23])) << "second SWA group " << tag << " fresh tail block 23";
    }

    manager->free(FreeInfo{second_resource, second_tokens});
}

TEST_F(KVCacheManagerTest, DSV4PopCachedBlocksPreservesGroupShape) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int seq_len  = 3 * spb + 1;
    auto      resource = makeDSV4BatchResource(manager->cacheConfig());
    auto      tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_info).success);

    InsertInfo insert_info{resource, tokens, /*is_resident=*/false};
    manager->insertIntoCache(insert_info);
    FreeInfo free_info{resource, tokens};
    manager->free(free_info);

    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/10);
    ASSERT_NE(evicted, nullptr);
    ASSERT_TRUE(evicted->hasCacheKeys());
    EXPECT_EQ(evicted->groupNums(), kDsv4PoolNum);
    EXPECT_EQ(evicted->cacheResource(0).blocksByGroup().size(), manager->cacheConfig().groups().size());

    bool saw_paged_block = false;
    bool saw_tail_block  = false;
    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(evicted->blocksNum(0, group.tag), static_cast<int>(evicted->cacheKeys(0).size()))
            << "group " << group.tag;
        for (auto block : evicted->blocks(0, group.tag)) {
            if (!isNullBlockIdx(block)) {
                if (isFullGroup(manager->cacheConfig(), group.tag)) {
                    saw_paged_block = true;
                } else {
                    saw_tail_block = true;
                }
            }
        }
    }
    EXPECT_TRUE(saw_paged_block);
    EXPECT_TRUE(saw_tail_block);

    manager->blockCacheFree(evicted);
}

TEST_F(KVCacheManagerTest, Init_ReturnTrue_WhenMemoryCacheDisabled) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache = false;

    auto kv_cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), false, nullptr, kv_cache_config);
    EXPECT_TRUE(kv_cache_manager->init());
    ASSERT_NE(kv_cache_manager->coordinator_, nullptr);
    ASSERT_NE(kv_cache_manager->coordinator_->update_thread_, nullptr);
}

TEST_F(KVCacheManagerTest, Init_Throws_WhenMemoryCacheEnabledButSizeMissing) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache = true;
    kv_cache_config.reuse_cache = true;  // coordinator init only enables memory connector when reuse_cache is true
    kv_cache_config.memory_cache_size_mb         = 0;
    kv_cache_config.memory_cache_sync_timeout_ms = 1;

    auto kv_cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), false, nullptr, kv_cache_config);
    EXPECT_THROW(kv_cache_manager->init(), std::runtime_error);
    // KVCacheManager::initConnectorCoordinator assigns coordinator_ before RTP_LLM_CHECK throws.
    ASSERT_NE(kv_cache_manager->coordinator_, nullptr);
    EXPECT_EQ(kv_cache_manager->coordinator_->update_thread_, nullptr);
}

TEST_F(KVCacheManagerTest, Init_Throws_WhenMemoryCacheEnabledButSyncTimeoutInvalid) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.reuse_cache                  = true;
    kv_cache_config.memory_cache_size_mb         = 10;
    kv_cache_config.memory_cache_sync_timeout_ms = 0;  // mock coordinator init failed

    auto kv_cache_manager = std::make_shared<KVCacheManager>(std::move(cache_config), false, nullptr, kv_cache_config);
    EXPECT_THROW(kv_cache_manager->init(), std::runtime_error);
    ASSERT_NE(kv_cache_manager->coordinator_, nullptr);
    EXPECT_EQ(kv_cache_manager->coordinator_->update_thread_, nullptr);
}

TEST_F(KVCacheManagerTest, Init_ReturnTrue_WhenMemoryCacheEnabledAndConfigValid) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;

    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.reuse_cache                  = true;
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.memory_cache_sync_timeout_ms = 1;
    runtime_config.worker_grpc_addrs             = {"127.0.0.1:12345"};

    auto kv_cache_manager = std::make_shared<KVCacheManager>(
        std::move(cache_config), false, nullptr, kv_cache_config, ParallelismConfig{}, runtime_config);
    EXPECT_TRUE(kv_cache_manager->init());

    auto coordinator = kv_cache_manager->coordinator_;
    ASSERT_NE(coordinator, nullptr);
    EXPECT_EQ(coordinator->connectors_.size(), 1u);
}

TEST_F(KVCacheManagerTest, AsyncLoadCache_ReturnFromCoordinator_Success) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    auto          allocator = std::make_shared<MockKVCacheAllocator>(cache_config);
    auto          mock_coordinator =
        std::make_shared<MockKVCacheConnectorCoordinator>(cache_config, kv_cache_config, runtime_config, allocator);

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    kv_cache_manager->coordinator_ = mock_coordinator;

    auto mock_context       = std::make_shared<MockKVCacheConnectorReadWriteContext>();
    auto mock_async_context = std::make_shared<MockAsyncContext>();

    EXPECT_CALL(*mock_coordinator, asyncRead(std::shared_ptr<KVCacheConnectorReadWriteContext>(mock_context)))
        .WillOnce(::testing::Return(mock_async_context));

    EXPECT_EQ(kv_cache_manager->asyncLoadCache(mock_context), mock_async_context);
}

TEST_F(KVCacheManagerTest, AsyncStoreCache_ReturnFromCoordinator_Success) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    auto          allocator = std::make_shared<MockKVCacheAllocator>(cache_config);
    auto          mock_coordinator =
        std::make_shared<MockKVCacheConnectorCoordinator>(cache_config, kv_cache_config, runtime_config, allocator);

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    kv_cache_manager->coordinator_ = mock_coordinator;

    auto mock_context       = std::make_shared<MockKVCacheConnectorReadWriteContext>();
    auto mock_async_context = std::make_shared<MockAsyncContext>();

    EXPECT_CALL(*mock_coordinator, asyncWrite(std::shared_ptr<KVCacheConnectorReadWriteContext>(mock_context)))
        .WillOnce(::testing::Return(mock_async_context));

    EXPECT_EQ(kv_cache_manager->asyncStoreCache(mock_context), mock_async_context);
}

TEST_F(KVCacheManagerTest, ExecuteFunction_ReturnFalse_CoordinatorReturnFalse) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    auto          allocator = std::make_shared<MockKVCacheAllocator>(cache_config);
    auto          mock_coordinator =
        std::make_shared<MockKVCacheConnectorCoordinator>(cache_config, kv_cache_config, runtime_config, allocator);

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    kv_cache_manager->coordinator_ = mock_coordinator;

    FunctionRequestPB request;
    request.mutable_mem_request();
    FunctionResponsePB response;

    EXPECT_CALL(*mock_coordinator, executeFunction(::testing::_, ::testing::_)).WillOnce(::testing::Return(false));

    EXPECT_FALSE(kv_cache_manager->executeFunction(request, response));
}

TEST_F(KVCacheManagerTest, ExecuteFunction_ReturnTrue_Success) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 4, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    auto          allocator = std::make_shared<MockKVCacheAllocator>(cache_config);
    auto          mock_coordinator =
        std::make_shared<MockKVCacheConnectorCoordinator>(cache_config, kv_cache_config, runtime_config, allocator);

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    kv_cache_manager->coordinator_ = mock_coordinator;

    FunctionRequestPB request;
    request.mutable_mem_request();
    FunctionResponsePB response;

    EXPECT_CALL(*mock_coordinator, executeFunction(::testing::_, ::testing::_)).WillOnce(::testing::Return(true));

    EXPECT_TRUE(kv_cache_manager->executeFunction(request, response));
}

TEST_F(KVCacheManagerTest, GetKVCacheInfo_MergesDeviceAndMemoryKeys_Dedup) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 8, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache = false;  // avoid starting real memory connector in coordinator->init()
    kv_cache_config.reuse_cache         = false;

    auto kv_cache_manager =
        std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), false, nullptr, kv_cache_config);
    ASSERT_TRUE(kv_cache_manager->init());
    ASSERT_NE(kv_cache_manager->allocator_, nullptr);
    ASSERT_NE(kv_cache_manager->coordinator_, nullptr);

    // Seed device block cache with keys: 10, 11, 12 (put makes MRU at front => snapshot order: 12,11,10)
    auto shared_cache = kv_cache_manager->allocator_->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);
    {
        const auto& tag = cache_config.soleGroupForLayer(0).tag;
        shared_cache->put(10, {{tag, 1}}, false);
        shared_cache->put(11, {{tag, 2}}, false);
        shared_cache->put(12, {{tag, 3}}, false);
    }

    // Inject a lightweight memory connector with a MemoryBlockCache snapshot:
    // put 11 then 13 => MRU order: 13,11 (11 duplicates device key)
    auto mem_connector = std::make_shared<KVCacheMemoryConnector>(
        cache_config, kv_cache_config, kv_cache_manager->allocator_, std::vector<std::string>{});
    mem_connector->block_cache_ = std::make_shared<MemoryDiskBlockCache>();
    {
        MemoryBlockCache::CacheItem item;
        item.cache_key   = 11;
        item.block_index = 101;
        item.block_size  = 1;
        item.is_resident = false;
        ASSERT_TRUE(mem_connector->block_cache_->put(item).first);
        item.cache_key   = 13;
        item.block_index = 102;
        ASSERT_TRUE(mem_connector->block_cache_->put(item).first);
    }
    kv_cache_manager->coordinator_->memory_connector_ = mem_connector;

    // latest_version=-1 forces SharedBlockCache snapshot to return all current keys.
    auto info = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true);

    // Current implementation uses unordered_set -> assign, so order is not stable.
    // Only validate de-dup and set-equality.
    std::vector<CacheKeyType> got = info.cached_keys;
    std::sort(got.begin(), got.end());
    std::vector<CacheKeyType> expected = {10, 11, 12, 13};
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(got, expected);
}

TEST_F(KVCacheManagerTest, GetKVCacheInfo_UsesSnapshotForCacheKeysWhenEnabled) {
    ScopedEnvVar snapshot_env("RTP_LLM_CACHE_STATUS_SNAPSHOT", "1");

    auto          cache_config = makeSimpleMhaCacheConfig(1, 8, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache = false;
    kv_cache_config.reuse_cache         = false;

    auto kv_cache_manager =
        std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config), false, nullptr, kv_cache_config);
    ASSERT_TRUE(kv_cache_manager->init());

    auto shared_cache = kv_cache_manager->allocator_->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    const auto& tag = cache_config.soleGroupForLayer(0).tag;
    shared_cache->put(10, {{tag, 1}}, false);
    shared_cache->put(11, {{tag, 2}}, false);

    kv_cache_manager->refreshKVCacheInfoSnapshot();

    auto first = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true);
    ASSERT_GE(first.version, 0);
    auto first_keys = first.cached_keys;
    std::sort(first_keys.begin(), first_keys.end());
    EXPECT_EQ(first_keys, (std::vector<CacheKeyType>{10, 11}));

    shared_cache->put(12, {{tag, 3}}, false);

    auto unchanged = kv_cache_manager->getKVCacheInfo(first.version, /*need_cache_keys=*/true);
    EXPECT_EQ(unchanged.version, first.version);
    auto unchanged_keys = unchanged.cached_keys;
    std::sort(unchanged_keys.begin(), unchanged_keys.end());
    EXPECT_EQ(unchanged_keys, (std::vector<CacheKeyType>{10, 11}));

    auto stale = kv_cache_manager->getKVCacheInfo(first.version - 1, /*need_cache_keys=*/true);
    EXPECT_EQ(stale.version, first.version);
    auto stale_keys = stale.cached_keys;
    std::sort(stale_keys.begin(), stale_keys.end());
    EXPECT_EQ(stale_keys, (std::vector<CacheKeyType>{10, 11}));

    kv_cache_manager->refreshKVCacheInfoSnapshot();

    auto updated = kv_cache_manager->getKVCacheInfo(first.version, /*need_cache_keys=*/true);
    EXPECT_GT(updated.version, first.version);
    auto updated_keys = updated.cached_keys;
    std::sort(updated_keys.begin(), updated_keys.end());
    EXPECT_EQ(updated_keys, (std::vector<CacheKeyType>{10, 11, 12}));

    auto current = kv_cache_manager->getKVCacheInfo(updated.version, /*need_cache_keys=*/true);
    EXPECT_EQ(current.version, updated.version);
    auto current_keys = current.cached_keys;
    std::sort(current_keys.begin(), current_keys.end());
    EXPECT_EQ(current_keys, (std::vector<CacheKeyType>{10, 11, 12}));
}

TEST_F(KVCacheManagerTest, GetKVCacheInfo_UsesSmallestHybridPoolTokenCapacity) {
    auto cache_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/16, /*swa_batch_size=*/3);

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    ASSERT_TRUE(kv_cache_manager->init());

    auto hybrid_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(kv_cache_manager->allocator_);
    ASSERT_NE(hybrid_allocator, nullptr);

    size_t expected_total_tokens     = std::numeric_limits<size_t>::max();
    size_t expected_available_tokens = std::numeric_limits<size_t>::max();
    ASSERT_GT(cache_config.groups().size(), 1u);

    for (const auto& group : cache_config.groups()) {
        const auto& pool = hybrid_allocator->blockPool(group.tag);
        ASSERT_NE(pool, nullptr);
        const size_t seq_size     = cache_config.seq_size_per_block;
        expected_total_tokens     = std::min(expected_total_tokens, pool->totalBlocksNum() * seq_size);
        expected_available_tokens = std::min(expected_available_tokens, pool->availableBlocksNum() * seq_size);
    }

    auto info = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);

    EXPECT_EQ(info.total_kv_cache, expected_total_tokens);
    EXPECT_EQ(info.available_kv_cache, expected_available_tokens);
    EXPECT_LT(info.total_kv_cache, kv_cache_manager->totalBlocksNum() * cache_config.seq_size_per_block);
}

TEST_F(KVCacheManagerTest, MaxAvailableTokensNumUsesCPVirtualBlockSizeForHybridPoolFullGroups) {
    auto cache_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/16, /*swa_batch_size=*/3);

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cloneManagerConfig(cache_config));
    ASSERT_TRUE(kv_cache_manager->init());

    auto hybrid_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(kv_cache_manager->allocator_);
    ASSERT_NE(hybrid_allocator, nullptr);

    const size_t physical_capacity = hybrid_allocator->maxAvailableTokensNum();
    auto         cp_slot_mapper =
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, static_cast<int>(cache_config.seq_size_per_block));
    kv_cache_manager->cp_slot_mapper_ = cp_slot_mapper;
    hybrid_allocator->setCPSlotMapper(cp_slot_mapper);

    size_t expected_logical_capacity = std::numeric_limits<size_t>::max();
    for (const auto& group : cache_config.groups()) {
        if (group.policy.group_type != CacheGroupType::FULL) {
            continue;
        }
        expected_logical_capacity = std::min(expected_logical_capacity,
                                             hybrid_allocator->blockPool(group.tag)->totalBlocksNum()
                                                 * static_cast<size_t>(cache_config.seq_size_per_block * 2));
    }

    EXPECT_EQ(kv_cache_manager->maxAvailableTokensNum(), expected_logical_capacity);
    EXPECT_GT(kv_cache_manager->maxAvailableTokensNum(), physical_capacity);
}

TEST_F(KVCacheManagerTest, GetKVCacheInfo_IncludesMemoryBlocksInTotalAndAvailable) {
    auto          cache_config = makeSimpleMhaCacheConfig(1, 8, 2, rtp_llm::DataType::TYPE_INT8);
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;

    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.reuse_cache                  = true;
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.memory_cache_sync_timeout_ms = 1;
    runtime_config.worker_grpc_addrs             = {"127.0.0.1:12345"};

    auto kv_cache_manager = std::make_shared<KVCacheManager>(
        std::move(cache_config), false, nullptr, kv_cache_config, ParallelismConfig{}, runtime_config);
    ASSERT_TRUE(kv_cache_manager->init());

    // With memory cache enabled, getKVCacheInfo() should include memory block pool stats.
    auto info = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);

    // The "device-only" kv cache would be totalBlocksNum() * seq_size_per_block.
    // With memory cache enabled, total_kv_cache/available_kv_cache should be >= device-only.
    const size_t device_only_total =
        kv_cache_manager->allocator_->totalBlocksNum() * kv_cache_manager->cacheConfig().seq_size_per_block;
    const size_t device_only_available =
        kv_cache_manager->allocator_->availableBlocksNum() * kv_cache_manager->cacheConfig().seq_size_per_block;

    EXPECT_GE(info.total_kv_cache, device_only_total);
    EXPECT_GE(info.available_kv_cache, device_only_available);
}

TEST_F(KVCacheManagerTest, DSV4EvictionTriggeredWhenPoolExhaustedByCache) {
    // This test verifies that when block pools are exhausted by cached (but freed) requests,
    // a new allocation correctly triggers LRU eviction from each group's independent BlockCache.
    //
    // Setup: block_num=8 → 7 usable blocks per group (block 0 reserved).
    // Request seq_len = 3*spb. FULL groups allocate 3 blocks. Reusable SWA groups allocate
    // linear-step blocks (step=1 here, so all 3), while HCA_STATE keeps only its active tail block.
    // insertIntoCache drops the active tail slot, so each completed request caches:
    //   FULL groups: 2 blocks per group
    //   SWA/state groups: fixed-window cached blocks; HCA_STATE skips reuse.
    //
    // After 3 requests are cached and request-freed:
    //   FULL groups (0,1,2): 6 blocks cached, 1 free → new request needs 3, triggers eviction
    //   SWA/state groups (3,4,5,6): reusable groups may also evict under their independent pools.
    //
    // The fourth allocation MUST succeed via eviction on FULL groups.
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/8);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int    spb         = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int    seq_len     = 3 * spb;
    const size_t free_before = manager->freeBlocksNum();
    // 7 groups × 7 usable blocks = 49 total free.
    EXPECT_EQ(free_before, 7u * 7u);

    // Helper: create CompleteTokenIds with a token-value offset so each request gets unique cache keys.
    auto makeTokens = [&](int offset) {
        auto input_ids      = torch::arange(offset, offset + seq_len, torch::kInt32);
        auto gi             = std::make_shared<GenerateInput>();
        gi->input_ids       = input_ids;
        gi->generate_config = std::make_shared<GenerateConfig>();
        auto cti            = std::make_shared<CompleteTokenIds>(1, 1, seq_len + 16, spb);
        cti->init(gi);
        cti->setSeqLength(seq_len);
        return cti;
    };

    // --- Request A: allocate, cache, free request reference ---
    auto       res_a    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_a = makeTokens(/*offset=*/0);
    MallocInfo malloc_a{res_a, tokens_a};
    malloc_a.reuse_cache         = true;
    malloc_a.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_a).success);

    InsertInfo insert_a{res_a, tokens_a, /*is_resident=*/false};
    manager->insertIntoCache(insert_a);
    FreeInfo free_a{res_a, tokens_a};
    manager->free(free_a);

    const size_t free_after_a = manager->freeBlocksNum();
    EXPECT_LT(free_after_a, free_before);

    // --- Request B: different tokens → different cache keys ---
    auto       res_b    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_b = makeTokens(/*offset=*/10000);
    MallocInfo malloc_b{res_b, tokens_b};
    malloc_b.reuse_cache         = true;
    malloc_b.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_b).success);

    InsertInfo insert_b{res_b, tokens_b, /*is_resident=*/false};
    manager->insertIntoCache(insert_b);
    FreeInfo free_b{res_b, tokens_b};
    manager->free(free_b);

    const size_t free_after_b = manager->freeBlocksNum();
    EXPECT_LT(free_after_b, free_after_a);

    // --- Request C: still fits, but leaves FULL groups with only one free block ---
    auto       res_c    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_c = makeTokens(/*offset=*/20000);
    MallocInfo malloc_c{res_c, tokens_c};
    malloc_c.reuse_cache         = true;
    malloc_c.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_c).success);

    InsertInfo insert_c{res_c, tokens_c, /*is_resident=*/false};
    manager->insertIntoCache(insert_c);
    FreeInfo free_c{res_c, tokens_c};
    manager->free(free_c);

    const size_t free_after_c = manager->freeBlocksNum();
    EXPECT_LE(free_after_c, free_after_b);

    // --- Request D: triggers eviction on FULL groups ---
    auto       res_d    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_d = makeTokens(/*offset=*/30000);
    MallocInfo malloc_d{res_d, tokens_d};
    malloc_d.reuse_cache         = true;
    malloc_d.enable_device_cache = false;

    // This allocation MUST succeed — FULL groups trigger ensureFreeBlocks → evict from cache.
    auto result_d = manager->malloc(malloc_d);
    ASSERT_TRUE(result_d.success) << "Fourth allocation should succeed via eviction";

    // Verify block structure for request D.
    ASSERT_EQ(res_d->groupNums(), kDsv4PoolNum);
    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(res_d->blocksNum(0, group.tag), 3) << "group " << group.tag;
        const auto& blocks = res_d->blocks(0, group.tag);
        if (isFullGroup(manager->cacheConfig(), group.tag)) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << group.tag << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(
                manager->cacheConfig(), blocks, group.tag, "fixed group", /*enable_reuse_cache=*/true);
        }
    }

    EXPECT_LE(manager->freeBlocksNum(), free_after_c) << "Pool should be tighter after D allocated";

    // --- Free D and verify blocks return to pool ---
    FreeInfo free_d{res_d, tokens_d};
    manager->free(free_d);

    // After freeing D, its blocks (request_ref→0, cache_ref=0 since we did not insert D into cache)
    // return to the free pool.
    // But cached blocks from eviction of A are fully freed (both refs=0) so they also count.
    // Expect freeBlocksNum >= free_after_c (at least as good as before D was allocated).
    EXPECT_GE(manager->freeBlocksNum(), free_after_c);

    // --- Pop all remaining cached blocks and verify full pool recovery ---
    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/100);
    if (evicted) {
        manager->blockCacheFree(evicted);
    }
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

TEST_F(KVCacheManagerTest, DSV4MaxConcurrencyOneReuseOneBlockAndAllocTwoTailBlocks) {
    auto manager_config =
        makeProductionDSV4Config(/*full_block_num=*/8, /*max_concurrency=*/1, /*hca_state_pool_blocks=*/12);
    ASSERT_EQ(manager_config.groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& tag : dsv4FixedTailGroupTags(manager_config)) {
        const uint32_t expected = isHcaStateGroup(tag) ? 12u : 8u;
        ASSERT_EQ(manager_config.group(tag).block_num, expected) << "group " << tag;
    }

    auto manager = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    EXPECT_EQ(free_before, 6u * 7u + 11u);
    const int spb = static_cast<int>(manager->cacheConfig().seq_size_per_block);

    auto makeTokens = [&](int seq_len) {
        auto input_ids      = torch::arange(0, seq_len, torch::kInt32);
        auto gi             = std::make_shared<GenerateInput>();
        gi->input_ids       = input_ids;
        gi->generate_config = std::make_shared<GenerateConfig>();
        auto cti            = std::make_shared<CompleteTokenIds>(1, 1, /*max_seq_len=*/4 * spb, spb);
        cti->init(gi);
        cti->setSeqLength(seq_len);
        return cti;
    };

    // Seed one reusable SWA/state block per independent pool. For a 2-block request,
    // insertIntoCache keeps only the first full block; the active tail is not cached.
    auto       seed_res    = makeDSV4BatchResource(manager->cacheConfig());
    auto       seed_tokens = makeTokens(2 * spb);
    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = false;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(seed_malloc).success);

    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        ASSERT_EQ(seed_res->blocksNum(0, tag), 2) << "seed group " << tag;
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), seed_res->blocks(0, tag), tag, "seed group");
    }

    manager->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{seed_res, seed_tokens});

    // Same prefix, one more block. This hits one cached independent-pool block and
    // must still have room for the two fresh tail blocks.  The matched block is
    // then skipped out of the active SWA tail by the decode allocation path.
    auto       reuse_res    = makeDSV4BatchResource(manager->cacheConfig());
    auto       reuse_tokens = makeTokens(3 * spb);
    MallocInfo reuse_malloc{reuse_res, reuse_tokens};
    reuse_malloc.reuse_cache         = true;
    reuse_malloc.enable_device_cache = true;
    auto reuse_result                = manager->malloc(reuse_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_EQ(reuse_result.reuse_len, 2 * spb);

    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        if (manager->cacheConfig().group(tag).policy.enable_prefix_reuse == false) {
            continue;
        }
        const auto& blocks = reuse_res->blocks(0, tag);
        ASSERT_EQ(blocks.size(), 3u) << "reuse group " << tag;
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "reuse group " << tag << " skipped reused prefix";
        EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "reuse group " << tag << " tail block 1";
        EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "reuse group " << tag << " tail block 2";
    }

    manager->free(FreeInfo{reuse_res, reuse_tokens});
    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/100);
    if (evicted) {
        manager->blockCacheFree(evicted);
    }
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

TEST_F(KVCacheManagerTest, DSV4EvictionOnSWAGroupsDuringInferenceWithDecodeContinuation) {
    // This test simulates full DSV4 inference including SWA group eviction.
    //
    // Tight stress layout:
    //   FULL groups (0,1,2): large paged pool (block_num=8, 7 usable)
    //   SWA  groups (3,4,5,6): small independent pool with 3 usable blocks
    //
    // SWA pools are sized by concurrency, NOT by global block_num. This test verifies that
    // eviction is triggered independently on SWA groups when concurrent requests exhaust
    // the independent pool, and that decode-phase removeSkippedBlocks interacts correctly with eviction.
    //
    // Lifecycle:
    //   Phase 1: 2 requests complete and get cached → SWA pools nearly full (2 of 3 cached)
    //   Phase 2: 3rd request triggers eviction on SWA groups
    //   Phase 3: Decode-phase incrKVBlock triggers further FULL/SWA eviction + removeSkippedBlocks
    //   Phase 4: Free and verify pool recovery
    auto manager_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/8, /*swa_batch_size=*/4);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb     = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    const int seq_len = 3 * spb;

    // Verify differentiated pool sizes.
    const size_t free_before = manager->freeBlocksNum();
    EXPECT_EQ(free_before, 3u * 7u + 4u * 7u);

    // Helper: create tokens with unique offset for distinct cache keys.
    auto makeTokens = [&](int offset) {
        auto input_ids      = torch::arange(offset, offset + seq_len, torch::kInt32);
        auto gi             = std::make_shared<GenerateInput>();
        gi->input_ids       = input_ids;
        gi->generate_config = std::make_shared<GenerateConfig>();
        auto cti            = std::make_shared<CompleteTokenIds>(1, 1, /*max_seq_len=*/10 * spb, spb);
        cti->init(gi);
        cti->setSeqLength(seq_len);
        return cti;
    };

    // === Phase 1: Fill caches with 2 completed requests ===
    auto       res_a    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_a = makeTokens(/*offset=*/0);
    MallocInfo malloc_a{res_a, tokens_a};
    malloc_a.reuse_cache         = true;
    malloc_a.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_a).success);
    InsertInfo insert_a{res_a, tokens_a, /*is_resident=*/false};
    manager->insertIntoCache(insert_a);
    manager->free(FreeInfo{res_a, tokens_a});

    auto       res_b    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_b = makeTokens(/*offset=*/10000);
    MallocInfo malloc_b{res_b, tokens_b};
    malloc_b.reuse_cache         = true;
    malloc_b.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_b).success);
    InsertInfo insert_b{res_b, tokens_b, /*is_resident=*/false};
    manager->insertIntoCache(insert_b);
    manager->free(FreeInfo{res_b, tokens_b});

    const size_t free_after_cache = manager->freeBlocksNum();
    EXPECT_LT(free_after_cache, free_before);

    // === Phase 2: 3rd request triggers eviction on SWA groups ===
    auto       res_c    = makeDSV4BatchResource(manager->cacheConfig());
    auto       tokens_c = makeTokens(/*offset=*/20000);
    MallocInfo malloc_c{res_c, tokens_c};
    malloc_c.reuse_cache         = true;
    malloc_c.enable_device_cache = false;

    // FULL needs 3, has exactly 3 free → no FULL eviction yet.
    // SWA needs 2, only 1 free → ensureFreeBlocks evicts 1 from SWA cache.
    auto result_c = manager->malloc(malloc_c);
    ASSERT_TRUE(result_c.success) << "3rd allocation must succeed via SWA eviction";

    // Verify block structure.
    ASSERT_EQ(res_c->groupNums(), kDsv4PoolNum);
    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(res_c->blocksNum(0, group.tag), 3) << "group " << group.tag;
        const auto& blocks = res_c->blocks(0, group.tag);
        if (isFullGroup(manager->cacheConfig(), group.tag)) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << group.tag << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(
                manager->cacheConfig(), blocks, group.tag, "SWA group", /*enable_reuse_cache=*/true);
        }
    }

    // === Phase 3: Decode incrKVBlock → SWA removeSkippedBlocks + further SWA eviction ===

    // --- Incr to 4*spb ---
    // Non-HCA SWA state starts from the reusable linear-step allocation and then keeps the active tail window.
    // HCA_STATE skips reuse and keeps only its active tail block.
    // FULL pool after Phase 2: 4 cached + 3 request = 7 used, 0 free → ensureFreeBlocks evicts 1.
    tokens_c->setSeqLength(4 * spb);
    MallocInfo incr1{res_c, tokens_c};
    incr1.reuse_cache         = false;
    incr1.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr1).success) << "First incr must succeed via eviction";

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(res_c->blocksNum(0, group.tag), 4) << "group " << group.tag << " after incr to 4*spb";
    }
    // SWA/state fixed groups retain the current tail window.
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), res_c->blocks(0, tag), tag, "SWA group");
    }

    // --- Incr to 5*spb ---
    // Non-HCA SWA removes blocks before the active two-block tail; HCA_STATE keeps a one-block tail.
    // SWA pools may need another eviction if no free block remains.
    tokens_c->setSeqLength(5 * spb);
    MallocInfo incr2{res_c, tokens_c};
    incr2.reuse_cache         = false;
    incr2.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr2).success) << "Second incr must succeed (removeSkipped frees block)";

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(res_c->blocksNum(0, group.tag), 5) << "group " << group.tag << " after incr to 5*spb";
    }
    // SWA/state fixed groups keep only the active tail window.
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), res_c->blocks(0, tag), tag, "SWA group");
    }

    // === Phase 4: Free all and verify full pool recovery ===
    manager->free(FreeInfo{res_c, tokens_c});

    // Pop remaining cached blocks to restore pool.
    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/100);
    if (evicted) {
        manager->blockCacheFree(evicted);
    }
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}
TEST_F(KVCacheManagerTest, DSV4InitThenIncrWithRemoveSkippedBlocksFullLifecycle) {
    // This test exercises the full lifecycle of a DSV4 request:
    //   1. initKVBlock (first malloc with 4 blocks)
    //   2. Multiple incrKVBlock calls (decode phase) that trigger removeSkippedBlocks
    //   3. Verify SWA groups free old non-tail blocks during incr
    //   4. Final free returns all blocks to pool
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/32);
    auto manager        = std::make_shared<KVCacheManager>(std::move(manager_config), /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    const int    spb         = static_cast<int>(manager->cacheConfig().seq_size_per_block);
    auto         resource    = makeDSV4BatchResource(manager->cacheConfig());

    // --- Phase 1: initKVBlock with 4 blocks (simulates prefill completion) ---
    const int init_seq_len = 4 * spb;
    auto      tokens       = makeDSV4CompleteTokenIds(init_seq_len, /*max_seq_len=*/10 * spb, spb);

    MallocInfo init_info{resource, tokens};
    init_info.reuse_cache         = false;
    init_info.enable_device_cache = false;
    auto init_result              = manager->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    ASSERT_EQ(resource->groupNums(), kDsv4PoolNum);

    // After init: FULL groups (0,1,2) have 4 real blocks each.
    //             SWA groups keep the active tail window; HCA_STATE keeps a one-block tail.
    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(resource->blocksNum(0, group.tag), 4) << "group " << group.tag;
        const auto& blocks = resource->blocks(0, group.tag);
        if (isFullGroup(manager->cacheConfig(), group.tag)) {
            for (int i = 0; i < 4; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << group.tag << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), blocks, group.tag, "SWA group");
        }
    }

    // Record block IDs allocated after init for later validation.
    std::map<std::string, BlockIndicesType> init_blocks;
    for (const auto& group : manager->cacheConfig().groups()) {
        init_blocks[group.tag] = resource->blocks(0, group.tag);
    }
    const size_t free_after_init = manager->freeBlocksNum();

    // --- Phase 2: First incrKVBlock (4 → 5 blocks) ---
    // removeSkippedBlocks on SWA groups: [NULL, NULL, A, B] → keep_begin=2, loop i=0..1 both NULL → no free.
    // Then allocate 1 new block per group.
    tokens->setSeqLength(5 * spb);
    MallocInfo incr1_info{resource, tokens};
    incr1_info.reuse_cache         = false;
    incr1_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr1_info).success);

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(resource->blocksNum(0, group.tag), 5) << "group " << group.tag << " after incr1";
    }
    // FULL groups: all 5 blocks should be real.
    for (const auto& tag : dsv4GroupTagsByType(manager->cacheConfig(), CacheGroupType::FULL)) {
        const auto& blocks = resource->blocks(0, tag);
        for (int i = 0; i < 5; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << tag << " pos " << i << " after incr1";
        }
        // Original init blocks should be preserved.
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(blocks[i], init_blocks.at(tag)[i]) << "FULL group " << tag << " pos " << i << " changed";
        }
    }
    // SWA/state fixed groups keep the current tail window.
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        const auto& blocks = resource->blocks(0, tag);
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), blocks, tag, "SWA group after incr1");
        if (!isHcaStateGroup(tag)) {
            EXPECT_EQ(blocks[3], init_blocks.at(tag)[3]) << "SWA group " << tag << " old tail pos 3";
        }
    }

    // Four fixed groups freed one stale block and all seven groups allocated one new block.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_init - 7 + 4);
    const size_t free_after_incr1 = manager->freeBlocksNum();

    // Record SWA tail blocks after incr1 for the next step.
    std::map<std::string, BlockIdxType> swa_new_C;
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        swa_new_C[tag] = resource->blocks(0, tag)[4];
    }

    // --- Phase 3: Second incrKVBlock (5 → 6 blocks) — triggers removeSkippedBlocks ---
    // SWA removeSkippedBlocks on [NULL, NULL, A, B, C] (size=5): keep_begin = 5-2 = 3.
    //   Loop i=0: NULL → skip.
    //   Loop i=1: NULL → skip.
    //   Loop i=2: A (real block) → FREE it, set to NULL.
    // After remove: [NULL, NULL, NULL, B, C]
    // Then malloc allocates 1 new block D → [NULL, NULL, NULL, B, C, D]
    tokens->setSeqLength(6 * spb);
    MallocInfo incr2_info{resource, tokens};
    incr2_info.reuse_cache         = false;
    incr2_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr2_info).success);

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(resource->blocksNum(0, group.tag), 6) << "group " << group.tag << " after incr2";
    }

    // FULL groups: all 6 blocks real, first 4 unchanged.
    for (const auto& tag : dsv4GroupTagsByType(manager->cacheConfig(), CacheGroupType::FULL)) {
        const auto& blocks = resource->blocks(0, tag);
        for (int i = 0; i < 6; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << tag << " pos " << i << " after incr2";
        }
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(blocks[i], init_blocks.at(tag)[i]) << "FULL group " << tag << " init block preserved";
        }
    }

    // SWA/state fixed groups after incr2 keep their configured active tail window.
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        const auto& blocks = resource->blocks(0, tag);
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), blocks, tag, "SWA group after incr2");
        if (!isHcaStateGroup(tag)) {
            EXPECT_EQ(blocks[4], swa_new_C.at(tag)) << "SWA group " << tag << " pos 4 = old C";
        }
    }

    // Free block accounting: SWA freed 1 block per SWA group (4 groups) at removeSkippedBlocks,
    // then allocated 1 new block per group (7 groups). Net change: -7 + 4 = -3.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_incr1 - 7 + 4);
    const size_t free_after_incr2 = manager->freeBlocksNum();

    // --- Phase 4: Third incrKVBlock (6 → 7 blocks) — triggers another removeSkippedBlocks ---
    // SWA removeSkippedBlocks on [NULL, NULL, NULL, B, C, D] (size=6): keep_begin = 6-2 = 4.
    //   Loop i=0..2: all NULL → skip.
    //   Loop i=3: B (real block) → FREE it, set to NULL.
    // After remove: [NULL, NULL, NULL, NULL, C, D]
    // Then malloc allocates 1 new block E → [NULL, NULL, NULL, NULL, C, D, E]
    tokens->setSeqLength(7 * spb);
    MallocInfo incr3_info{resource, tokens};
    incr3_info.reuse_cache         = false;
    incr3_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr3_info).success);

    for (const auto& group : manager->cacheConfig().groups()) {
        ASSERT_EQ(resource->blocksNum(0, group.tag), 7) << "group " << group.tag << " after incr3";
    }

    // SWA/state fixed groups after incr3 keep their configured active tail window.
    for (const auto& tag : dsv4FixedTailGroupTags(manager->cacheConfig())) {
        const auto& blocks = resource->blocks(0, tag);
        expectDsv4SwaAllocatedBlocks(manager->cacheConfig(), blocks, tag, "SWA group after incr3");
    }

    // SWA freed 1 block per SWA group (4) and allocated 1 per all groups (7). Net: -7+4 = -3.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_incr2 - 7 + 4);

    // --- Phase 5: Free all — all blocks should return to pool ---
    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

}  // namespace test
}  // namespace rtp_llm
