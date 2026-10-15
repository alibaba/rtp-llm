#include <gtest/gtest.h>
#include <memory>
#include <algorithm>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTestConfig(int block_num = 20, int seq_size_per_block = 4) {
    return makeSimpleMhaCacheConfig(
        /*layer_num=*/2,
        block_num,
        /*tokens_per_block=*/static_cast<size_t>(seq_size_per_block),
        rtp_llm::DataType::TYPE_FP16,
        /*local_head_num_kv=*/1,
        /*size_per_head=*/16);
}

static CompleteTokenIdsPtr makeTokenIds(int batch_size, int seq_len, int block_size) {
    auto  ids       = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_len + 100, block_size);
    auto  input_ids = torch::empty({(int64_t)seq_len}, torch::kInt32);
    auto* ptr       = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i)
        ptr[i] = i + 1;
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();
    ids->init(gi);
    return ids;
}

static BatchKVCacheResourcePtr makeResource(int batch_size, const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.topologyPtr());
    return res;
}

static void resolveManagerCpLayout(CacheConfig& config, ParallelismConfig parallelism) {
    auto policies = std::vector<CacheGroupPolicy>{};
    for (const auto& group : config.topology().groups()) {
        auto policy = group.policy;
        if (policy.group_type == CacheGroupType::FULL) {
            policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
        }
        policies.push_back(policy);
    }
    setTestGroupPolicies(config, policies);
    config.finalizeStorageLayout(parallelism);
}

class KVCacheManagerCPLayoutTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// A non-sharded direct manager keeps the ordinary request layout contract.
TEST_F(KVCacheManagerCPLayoutTest, NoCPShardingKeepsOrdinaryRequestLayout) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = false;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());
    EXPECT_FALSE(mgr->cacheConfig().request_projection_required);
}

// One rank needs no sharded request projection even if the CP flag is set.
TEST_F(KVCacheManagerCPLayoutTest, SingleRankKeepsOrdinaryRequestLayout) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 1;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());
    EXPECT_FALSE(mgr->cacheConfig().request_projection_required);
}

// Sharded direct construction consumes a cache-finalized storage layout.
TEST_F(KVCacheManagerCPLayoutTest, CPShardingConsumesFinalizedLayout) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 1;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, par);

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());
    EXPECT_TRUE(mgr->cacheConfig().request_projection_required);
    ASSERT_TRUE(mgr->cacheConfig().cache_info_block_tokens.has_value());
    EXPECT_EQ(*mgr->cacheConfig().cache_info_block_tokens, static_cast<size_t>(seq_size_per_block * 2));
}

TEST_F(KVCacheManagerCPLayoutTest, CPShardingRequiresFinalizedStorageConfig) {
    auto config = makeTestConfig();
    ParallelismConfig par;
    par.tp_size = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    EXPECT_ANY_THROW((void)std::make_shared<KVCacheManager>(
        config, /*warmup=*/true, nullptr, KVCacheConfig{}, par));
}

TEST_F(KVCacheManagerCPLayoutTest, CPShardingEnabled_CacheInfoReportsVirtualBlockSize) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 4;
    par.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, par);

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto info = mgr->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);
    EXPECT_EQ(info.block_size, static_cast<size_t>(seq_size_per_block * par.tp_size));
}

TEST_F(KVCacheManagerCPLayoutTest, StorageLayoutAlignsCacheInfoAndFullCapacityUnits) {
    for (int cp_size : {1, 2, 4}) {
        auto config = makeTestConfig(/*block_num=*/20, /*seq_size_per_block=*/4);
        ParallelismConfig par;
        par.tp_rank = 0;
        par.tp_size = cp_size;
        par.prefill_cp_config.kv_cache_sharded = true;

        resolveManagerCpLayout(config, par);
        ASSERT_TRUE(config.cache_info_block_tokens.has_value());
        EXPECT_EQ(*config.cache_info_block_tokens, static_cast<size_t>(4 * cp_size));
        EXPECT_EQ(config.capacity_tokens_per_block.at("default"), static_cast<size_t>(4 * cp_size));

        auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
        ASSERT_TRUE(manager->init());
        const auto info = manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);
        EXPECT_EQ(info.block_size, static_cast<size_t>(4 * cp_size));
        // Upstream b252e385 makes CacheInfo use the same virtual FULL
        // bundle span as maxAvailableTokensNum.
        // One sentinel block is excluded from this 20-block fixture's pool.
        EXPECT_EQ(info.total_kv_cache, static_cast<size_t>(76 * cp_size));
        EXPECT_EQ(info.available_kv_cache, static_cast<size_t>(76 * cp_size));
        EXPECT_EQ(manager->maxAvailableTokensNum(), static_cast<size_t>(76 * cp_size));
    }
}

TEST_F(KVCacheManagerCPLayoutTest, TheoryWindowSnapshotPreservesPerBatchTailAndCanonicalKeys) {
    for (int cp_size : {1, 2, 4}) {
        auto config = makeTestConfig();
        ParallelismConfig par;
        par.tp_rank = 0;
        par.tp_size = cp_size;
        par.prefill_cp_config.kv_cache_sharded = true;
        config.finalizeStorageLayout(par);

        auto resource = makeResource(2, config);
        resource->setBatchCacheKeys(0, {10, 11, 12, 13});
        resource->setBatchCacheKeys(1, {20, 21, 22, 23, 24});
        resource->setLastBlockAligned(false);
        auto tokens = makeTokenIds(2, /*seq_len=*/20, /*block_size=*/4);
        const auto without_metric = snapshotGlobalCacheRequest(*resource, *tokens);
        EXPECT_FALSE(without_metric->theory_window_keys.has_value());
        const auto with_metric = snapshotGlobalCacheRequest(*resource, *tokens, -1, true);
        ASSERT_TRUE(with_metric->theory_window_keys.has_value());
        ASSERT_EQ(with_metric->theory_window_keys->size(), 2u);
        CacheKeysType canonical_theory;
        for (const auto& batch_keys : *with_metric->theory_window_keys) {
            const auto canonical_batch = config.canonicalCacheKeys(batch_keys);
            canonical_theory.insert(canonical_theory.end(), canonical_batch.begin(), canonical_batch.end());
        }
        const CacheKeysType expected = cp_size == 1 ? CacheKeysType{10, 11, 12, 20, 21, 22, 23} :
                                       cp_size == 2 ? CacheKeysType{11, 21, 23} : CacheKeysType{23};
        EXPECT_EQ(canonical_theory, expected);
        EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{10, 11, 12, 13}));
        EXPECT_EQ(resource->cacheKeys(1), (CacheKeysType{20, 21, 22, 23, 24}));
    }
}

// Partial tails may be allocated as live KV blocks before they become cacheable
// full blocks. CP invariants must therefore be based on logical sequence length,
// not cacheKeys().size().
TEST_F(KVCacheManagerCPLayoutTest, CPShardedMallocAllowsPartialTailWithoutCacheKey) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    ParallelismConfig layout_parallelism;
    layout_parallelism.role_type                = RoleType::PREFILL;
    layout_parallelism.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    layout_parallelism.tp_size                  = 2;
    layout_parallelism.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, layout_parallelism);
    par = layout_parallelism;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto resource  = makeResource(1, config);
    auto token_ids = makeTokenIds(1, /*seq_len=*/1, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    info.request_layout = snapshotGlobalCacheRequest(*resource, *token_ids);

    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, "default"), 1);

    token_ids->setSeqLength(2);
    info.request_layout = snapshotGlobalCacheRequest(*resource, *token_ids);
    result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, "default"), 1);
    EXPECT_EQ(resource->cacheKeys(0).size(), 0);
}

TEST_F(KVCacheManagerCPLayoutTest, DomainModulesShareCpKeyGridButOwnIndependentRowsAndBacking) {
    auto              main  = makeTestConfig(20, 4);
    auto              draft = makeSimpleMhaCacheConfig(1, 20, 4, DataType::TYPE_FP16, 1, 8);
    ParallelismConfig parallelism;
    parallelism.tp_rank                            = 1;
    parallelism.tp_size                            = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(main, parallelism);
    resolveManagerCpLayout(draft, parallelism);
    CacheDomainConfig domain(std::vector<CacheConfig>{main, draft, draft}, 2);
    auto              manager = std::make_shared<KVCacheManager>(domain, false, nullptr, KVCacheConfig{}, parallelism);
    ASSERT_TRUE(manager->init());
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(domain);
    auto       tokens = makeTokenIds(1, 16, 4);
    MallocInfo info{resource, tokens};
    info.request_layout = snapshotGlobalCacheRequest(*resource, *tokens);
    ASSERT_TRUE(manager->malloc(info).success);
    std::vector<void*> addresses;
    for (size_t m = 0; m < domain.moduleCount(); ++m) {
        const ModuleGroupKey key{CacheModuleId(m), "default"};
        EXPECT_EQ(resource->blocksNum(0, key), 2);
        const auto buffers = manager->convertIndexToBuffer(key.module_id, 0, key.tag, resource->blocks(0, key).front());
        ASSERT_FALSE(buffers.empty());
        addresses.push_back(buffers.front().addr);
        // Each rank stores 4 tokens per block; the shared key stride spans both CP ranks.
        EXPECT_EQ(domain.module(key.module_id).group("default").cacheKeyTokenStride(), 8u);
    }
    EXPECT_NE(addresses[0], addresses[1]);
    EXPECT_NE(addresses[1], addresses[2]);
    EXPECT_EQ(resource->cacheResource().activeModuleCount(), 2u);
    EXPECT_EQ(resource->cacheKeys(0).size(), 4u);
}

// malloc() uses the immutable global request snapshot and cache-owned storage mapping.
// With CP sharding (cp_size=2, block_size=4), virtual_block_size=8.
// A sequence of 16 tokens needs ceil(16/8)=2 physical blocks per batch (not 4).
// Keep this pre-existing multi-rank case disabled pending its dedicated harness.
TEST_F(KVCacheManagerCPLayoutTest, DISABLED_MallocAutoInjectReducesBlockCount) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, par);

    // The fixture has complete capacities; manager construction does not run collectives.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    info.request_layout = snapshotGlobalCacheRequest(*resource, *token_ids);
    auto       result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks
    EXPECT_EQ(resource->blocksNum(0, "default"), 2);
}

// Without CP sharding, the same seq_len should allocate more blocks.
// Keep this pre-existing multi-rank case disabled pending its dedicated harness.
TEST_F(KVCacheManagerCPLayoutTest, DISABLED_MallocWithoutCPAllocatesFullBlocks) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = false;

    // The fixture has complete capacities; manager construction does not run collectives.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // Without CP: ceil(16/4) = 4 blocks
    EXPECT_EQ(resource->blocksNum(0, "default"), 4);
}

// Cache-owned storage layout drives malloc sharding.
// Keep this pre-existing multi-rank case disabled pending its dedicated harness.
TEST_F(KVCacheManagerCPLayoutTest, DISABLED_StorageLayoutControlsMalloc) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/30, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    auto layout_parallelism = par;
    layout_parallelism.tp_size = 4;
    resolveManagerCpLayout(config, layout_parallelism);
    par = layout_parallelism;

    // The fixture has complete capacities; manager construction does not run collectives.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 64;
    auto      resource  = makeResource(1, config);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    // virtual_block_size = 4 * 4 = 16
    // effectiveSeqLenForAlloc(64) = ceil(64/16)*4 = 16 tokens => ceil(16/4) = 4 blocks

    MallocInfo info{resource, token_ids};
    info.request_layout = snapshotGlobalCacheRequest(*resource, *token_ids);
    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(resource->blocksNum(0, "default"), 4);
}

// insertIntoCache() derives canonical keys from the cache storage layout.
// DISABLED: same reason as above (multi-rank harness needed).
TEST_F(KVCacheManagerCPLayoutTest, DISABLED_InsertDerivesCanonicalKeys) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, par);

    KVCacheConfig kv_cfg;
    kv_cfg.reuse_cache         = true;
    kv_cfg.enable_device_cache = true;

    auto mgr = std::make_shared<KVCacheManager>(config, false, nullptr, kv_cfg, par);
    ASSERT_TRUE(mgr->init());
    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_cache_lookup = true;
    malloc_info.request_layout      = snapshotGlobalCacheRequest(*resource, *token_ids);
    auto result                     = mgr->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    // The cache derives canonical keys before publishing the sharded insertion.
    InsertInfo insert_info{resource, token_ids, /*is_resident=*/false};
    {
        size_t resident_prefix_length = 0;
        EXPECT_NO_THROW(mgr->insertIntoCache(insert_info, resident_prefix_length));
    }

    // Now try to malloc again with the same token_ids -- should get reuse hit.
    auto       resource2 = makeResource(1, config);
    MallocInfo malloc_info2{resource2, token_ids};
    malloc_info2.reuse_cache         = true;
    malloc_info2.enable_cache_lookup = true;
    malloc_info2.request_layout      = snapshotGlobalCacheRequest(*resource2, *token_ids);
    auto result2                     = mgr->malloc(malloc_info2);
    ASSERT_TRUE(result2.success);
    // With CP sharding (cp_size=2, block_size=4), virtual_block_size=8.
    // seq_len=16 produces 2 cache keys (each covering 8 tokens).
    // match drops the last key → 1 matched key → reuse_len = 1 * virtual_block_size = 8.
    // The sharded reuse_length adjustment ensures this is 1 * virtual_block_size = 8, not 1 * seq_size_per_block = 4.
    EXPECT_EQ(result2.reuse_len, seq_size_per_block * par.tp_size);  // = 4 * 2 = 8
}

}  // namespace test
}  // namespace rtp_llm
