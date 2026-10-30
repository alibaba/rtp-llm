#include <gtest/gtest.h>
#include <memory>
#include <limits>
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

TEST_F(KVCacheManagerCPLayoutTest, DecodeTpReplicatesFullPagesWhilePrefillCpShardsThem) {
    ModelConfig model;
    model.num_layers                          = 1;
    model.attn_config.tokens_per_block        = 256;
    model.attn_config.kernel_tokens_per_block = 128;
    model.attn_config.sliding_window          = 128;
    KVCacheSpecDesc full;
    full.tag               = "global_kv_2";
    full.cache_type        = KVCacheSpecType::OpaqueKV;
    full.entry_dtype       = DataType::TYPE_UINT8;
    full.entry_elems       = 288;
    full.compression_ratio = 2;
    full.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    KVCacheSpecDesc swa;
    swa.tag                          = "swa_kv";
    swa.cache_type                   = KVCacheSpecType::OpaqueState;
    swa.is_state_cache               = true;
    swa.entry_dtype                  = DataType::TYPE_UINT8;
    swa.entry_elems                  = 528;
    swa.explicit_entry_count         = 128;
    swa.block_stride_bytes_alignment = 16896;
    swa.cp                           = CacheCpPolicyDesc{};
    swa.cp->scale_seq_size           = true;
    swa.cp->align_payload            = true;
    swa.cp->prefill_slice_layout     = CpPrefillSliceLayout::BLOCK_STRIDE;
    swa.cp->slice                    = CpBlockSliceMode::EQUAL_BYTES;
    model.kv_cache_spec_descs        = {{full, swa}};

    for (const auto role : {RoleType::PREFILL, RoleType::DECODE}) {
        const auto ranks = std::vector<size_t>{0u, 1u};
        for (const size_t rank : ranks) {
            ParallelismConfig par;
            par.role_type = role;
            par.tp_size   = 2;
            par.tp_rank   = rank;
            par.prefill_cp_config.method =
                role == RoleType::PREFILL ? CPRotateMethod::ALL_GATHER : CPRotateMethod::PREFILL_CP;
            par.prefill_cp_config.kv_cache_sharded = true;
            par.prefill_cp_config.prefill_cp_size  = 2;
            PDSepConfig pd;
            pd.role_type = role;

            // Decode keeps every received FULL page on each TP rank — even at
            // decode TP2 the cache-owned storage layout stays CP1; the manager
            // only records the upstream producer CP size for transport.
            ParallelismConfig layout_parallelism = par;
            auto config = finalizeCacheConfig(CacheConfigCreator::createConfig(model, layout_parallelism, {}), 20);
            EXPECT_EQ(config.group(full.tag).seqSizePerBlock(), 256u);
            EXPECT_EQ(config.group(swa.tag).seqSizePerBlock(), 512u);
            EXPECT_EQ(config.request_projection_required, role == RoleType::PREFILL);
            EXPECT_EQ(config.cache_cp_size, role == RoleType::PREFILL ? 2u : 1u);
            EXPECT_EQ(config.canonicalSizePerBlock(), role == RoleType::PREFILL ? 512u : 256u);
            auto manager = std::make_shared<KVCacheManager>(
                config, false, nullptr, KVCacheConfig{}, par, RuntimeConfig{}, SpeculativeExecutionConfig{}, pd);
            ASSERT_TRUE(manager->init());
            EXPECT_EQ(manager->cacheConfig().request_projection_required, role == RoleType::PREFILL);
            auto       resource = makeResource(1, config);
            auto       tokens   = makeTokenIds(1, 783, 256);
            MallocInfo malloc_info{resource, tokens};
            const auto result = manager->malloc(malloc_info);
            ASSERT_TRUE(result.success);
            const size_t full_blocks = role == RoleType::PREFILL ? 2 : 4;
            EXPECT_EQ(resource->blocksNum(0, full.tag), full_blocks);
            EXPECT_EQ(resource->kernelBlocks(0, full.tag).size(), full_blocks * 2);
            EXPECT_TRUE(std::all_of(resource->kernelBlocks(0, full.tag).begin(),
                                    resource->kernelBlocks(0, full.tag).end(),
                                    [](int id) { return id > 0; }));
            EXPECT_EQ(resource->blocksNum(0, swa.tag), 2u);
            EXPECT_EQ(resource->kernelBlocks(0, swa.tag).size(), 2u);
            manager->free(FreeInfo{resource, tokens});
        }
    }
}

// The CP folding guard: only a true CP rotate method on a non-decode role
// folds local allocation; flag-only or decode configurations stay unfolded.
TEST(CacheKeyLayoutTest, CpFoldingGuardMatrix) {
    for (const auto method : {CPRotateMethod::DISABLED, CPRotateMethod::PREFILL_CP, CPRotateMethod::ALL_GATHER}) {
        for (const auto role : {RoleType::PREFILL, RoleType::DECODE}) {
            auto              config = makeTestConfig();
            ParallelismConfig par;
            par.role_type                          = role;
            par.tp_rank                            = 0;
            par.tp_size                            = 2;
            par.prefill_cp_config.method           = method;
            par.prefill_cp_config.kv_cache_sharded = true;
            config.finalizeStorageLayout(par);
            const bool folded = method == CPRotateMethod::ALL_GATHER && role == RoleType::PREFILL;
            EXPECT_EQ(config.cache_cp_size, folded ? 2u : 1u);
            EXPECT_EQ(config.request_projection_required, folded);
        }
    }
}

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
    ASSERT_TRUE(mgr->cacheConfig().cache_key_layout_resolved);
    EXPECT_EQ(mgr->cacheConfig().canonicalSizePerBlock(), static_cast<size_t>(seq_size_per_block * 2));
}

TEST_F(KVCacheManagerCPLayoutTest, CPShardingRequiresFinalizedStorageConfig) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.role_type                          = RoleType::PREFILL;
    par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    EXPECT_ANY_THROW((void)std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par));
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
        auto              config = makeTestConfig(/*block_num=*/20, /*seq_size_per_block=*/4);
        ParallelismConfig par;
        par.role_type                          = RoleType::PREFILL;
        par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        par.tp_rank                            = 0;
        par.tp_size                            = cp_size;
        par.prefill_cp_config.kv_cache_sharded = true;

        resolveManagerCpLayout(config, par);
        ASSERT_TRUE(config.cache_key_layout_resolved);
        EXPECT_EQ(config.canonicalSizePerBlock(), static_cast<size_t>(4 * cp_size));
        EXPECT_EQ(config.globalSizePerBlock("default"), static_cast<size_t>(4 * cp_size));

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

TEST(CacheKeyLayoutTest, NoFullGroupUsesCanonicalEndpointsAcrossPhysicalSpans) {
    for (size_t scale : {1u, 2u, 4u}) {
        CacheConfig config;
        config.dtype               = DataType::TYPE_FP16;
        config.layer_num           = 1;
        config.seq_size_per_block  = 4;
        const auto spec            = makeResolvedMhaSpec(config.dtype, 1, 1, 8, "swa");
        auto       policy          = defaultCacheGroupPolicy(CacheGroupType::SWA);
        policy.cp_mapping          = CpBlockMappingMode::NONE;
        policy.sliding_window_size = 16;
        config.fromGroupedSpecs({spec}, {{0}}, {CacheGroupType::SWA}, {"swa"}, {policy});
        ParallelismConfig parallelism;
        parallelism.tp_size                            = scale;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        config.finalizeStorageLayout(parallelism);

        EXPECT_EQ(config.cache_cp_size, scale);
        EXPECT_EQ(config.groupKeyScale("swa"), 2u);
        EXPECT_EQ(config.canonicalSizePerBlock(), 4u * scale);
        EXPECT_TRUE(config.supportsCanonicalReuse());
        EXPECT_EQ(config.localAllocTokens("swa", 29), 29);
        EXPECT_EQ(config.canonicalBlockPosition("swa", 1), scale - 1);
        EXPECT_EQ(config.globalSizePerBlock("swa"), 8u);
        EXPECT_ANY_THROW(
            config.canonicalBlockPosition("swa", std::numeric_limits<size_t>::max() / config.canonicalSizePerBlock()));
    }
}

TEST(CacheKeyLayoutTest, WarmupKeepsBaseKeyStrideWithLargerPhysicalGroup) {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = 1;
    config.seq_size_per_block = 4;
    const auto spec           = makeResolvedMhaSpec(config.dtype, 1, 1, 16, "full");
    auto       policy         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.cp_mapping         = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    config.fromGroupedSpecs({spec}, {{0}}, {CacheGroupType::FULL}, {"full"}, {policy});
    ParallelismConfig parallelism;
    parallelism.tp_size                            = 4;
    parallelism.tp_rank                            = 3;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(parallelism, /*warmup=*/true);

    EXPECT_FALSE(config.request_projection_required);
    EXPECT_EQ(config.cache_cp_size, 1u);
    EXPECT_EQ(config.groupKeyScale("full"), 4u);
    EXPECT_EQ(config.group("full").seqSizePerBlock(), 16u);
    EXPECT_EQ(config.canonicalSizePerBlock(), 4u);
    EXPECT_EQ(config.globalSizePerBlock("full"), 16u);
    EXPECT_FALSE(config.supportsCanonicalReuse());
    EXPECT_EQ(config.canonicalCacheKeys({100, 101, 102}), (CacheKeysType{100, 101, 102}));
    EXPECT_EQ(config.localAllocTokens("full", 29), 29);
}

TEST_F(KVCacheManagerCPLayoutTest, GlobalSpanCapacityTracksAllocationAndRecomputeReferences) {
    for (const int scale : {1, 2, 4}) {
        for (const auto mapping :
             {CpBlockMappingMode::BLOCK_ROUND_ROBIN, CpBlockMappingMode::NONE, CpBlockMappingMode::COMPACT_LAST_RANK}) {
            SCOPED_TRACE(scale);
            SCOPED_TRACE(static_cast<int>(mapping));
            auto config               = makeTestConfig(20, 8);
            config.seq_size_per_block = 4;
            auto policy               = config.group("default").policy;
            policy.cp_mapping         = mapping;
            setTestGroupPolicies(config, {policy});
            ParallelismConfig par;
            par.tp_size                            = scale;
            par.prefill_cp_config.kv_cache_sharded = true;
            par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
            config.finalizeStorageLayout(par);
            const size_t span = mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN ? 8u * scale : 8u;
            EXPECT_EQ(config.canonicalSizePerBlock(), 4u * scale);
            EXPECT_EQ(config.globalSizePerBlock("default"), span);

            auto manager = std::make_shared<KVCacheManager>(config, true, nullptr, KVCacheConfig{}, par);
            ASSERT_TRUE(manager->init());
            const auto allocator = manager->coordinator_manager_;
            const auto pool      = allocator->groupBlockPools().front();
            ASSERT_EQ(pool->totalBlocksNum(), 19u);
            EXPECT_EQ(allocator->totalTokensNum(), 19u * span);
            EXPECT_EQ(allocator->availableTokensNum(), 19u * span);
            auto       resource = makeResource(1, config);
            auto       tokens   = makeTokenIds(1, static_cast<int>(2 * span), 4);
            MallocInfo info{resource, tokens};
            info.enable_cache_lookup = false;
            ASSERT_TRUE(manager->malloc(info).success);
            const auto original = resource->blocks(0, "default");
            ASSERT_EQ(original.size(), 2u);
            EXPECT_EQ(pool->freeBlocksNum(), 17u);
            EXPECT_EQ(allocator->availableTokensNum(), 17u * span);
            EXPECT_EQ(allocator->totalTokensNum(), 19u * span);
            pool->incTreeRef(original, BlockTreeRefType::CACHE);
            EXPECT_EQ(pool->refCount(original[0]), 2u);
            EXPECT_EQ(pool->refCount(original[1]), 2u);
            ASSERT_TRUE(allocator->detachSharedBlocksForRecompute(resource, static_cast<int>(span)));
            const auto current = resource->blocks(0, "default");
            EXPECT_EQ(current[0], original[0]);
            EXPECT_NE(current[1], original[1]);
            EXPECT_EQ(pool->refCount(original[0]), 2u);
            EXPECT_EQ(pool->refCount(original[1]), 1u);
            EXPECT_EQ(pool->refCount(current[1]), 1u);
            EXPECT_EQ(allocator->availableTokensNum(), 17u * span);
            manager->free(FreeInfo{resource, tokens});
            pool->decTreeRef(original, BlockTreeRefType::CACHE);
            EXPECT_EQ(pool->referencedBlocksNum(), 0u);
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
            EXPECT_EQ(pool->freeBlocksNum(), 19u);
            EXPECT_EQ(allocator->availableTokensNum(), 19u * span);
        }
    }
}

TEST(CacheKeyLayoutTest, GlobalSpanRejectsInvalidScaleAndOverflowWithoutChangingEndpointDivision) {
    auto config               = makeTestConfig(20, 8);
    config.seq_size_per_block = 4;
    ParallelismConfig par;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, par);
    config.cache_cp_size = 0;
    EXPECT_ANY_THROW(config.globalSizePerBlock("default"));
    const size_t max     = std::numeric_limits<size_t>::max();
    config.cache_cp_size = max / 8 + 1;
    EXPECT_ANY_THROW(config.globalSizePerBlock("default"));
    EXPECT_ANY_THROW(config.globalSizePerBlock("missing"));
    config.cache_cp_size                                                                   = 1;
    std::const_pointer_cast<KVCacheSpec>(config.group("default").spec)->seq_size_per_block = 0;
    EXPECT_ANY_THROW(config.globalSizePerBlock("default"));
}

TEST_F(KVCacheManagerCPLayoutTest, GlobalSpanCapacityRejectsOverflowAndUnresolvedProjection) {
    for (int invalid_layout = 0; invalid_layout < 3; ++invalid_layout) {
        auto config               = makeTestConfig(20, 8);
        config.seq_size_per_block = 4;
        resolveManagerCpLayout(config, ParallelismConfig{});
        if (invalid_layout == 0) {
            config.cache_cp_size = std::numeric_limits<size_t>::max() / 8;
            EXPECT_NO_THROW(config.globalSizePerBlock("default"));
        } else {
            config.cache_key_layout_resolved   = false;
            config.cache_cp_size               = invalid_layout == 1 ? 2 : 1;
            config.request_projection_required = invalid_layout == 2;
        }
        auto allocator = std::make_shared<CoordinatorCacheManager>(config);
        ASSERT_TRUE(allocator->init());
        EXPECT_ANY_THROW(allocator->totalTokensNum());
        EXPECT_ANY_THROW(allocator->availableTokensNum());
    }
}

TEST(CacheKeyLayoutTest, GlobalSpanUsesMtpAliasTopologyWithoutMutatingDraft) {
    for (const int scale : {1, 2, 4}) {
        auto make_config = [&](const std::string& tag) {
            CacheConfig config;
            config.dtype              = DataType::TYPE_FP16;
            config.layer_num          = 1;
            config.seq_size_per_block = 4;
            auto policy               = defaultCacheGroupPolicy(CacheGroupType::FULL);
            policy.cp_mapping         = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
            config.fromGroupedSpecs(
                {makeResolvedMhaSpec(config.dtype, 1, 1, 8, tag)}, {{0}}, {CacheGroupType::FULL}, {tag}, {policy});
            ParallelismConfig par;
            par.tp_size                            = scale;
            par.prefill_cp_config.kv_cache_sharded = true;
            par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
            config.finalizeStorageLayout(par);
            return config;
        };
        auto       main           = make_config("full");
        const auto draft          = make_config("default");
        const auto draft_topology = draft.topologyPtr();
        const auto child          = main.mergeMTPModule(draft, 0, 1);
        ASSERT_NE(child, nullptr);
        EXPECT_EQ(child->groupForLayer(0, "full").tag, "full");
        EXPECT_EQ(child->globalSizePerBlock("full"), 8u * scale);
        EXPECT_EQ(child->canonicalSizePerBlock(), 4u * scale);
        EXPECT_ANY_THROW(child->globalSizePerBlock("default"));
        EXPECT_EQ(draft.topologyPtr(), draft_topology);
        EXPECT_EQ(draft.groupForLayer(0, "default").tag, "default");
        EXPECT_EQ(draft.globalSizePerBlock("default"), 8u * scale);
        EXPECT_EQ(draft.layer_all_num(), 1u);
    }
}

TEST_F(KVCacheManagerCPLayoutTest, TheoryWindowSnapshotPreservesPerBatchTailAndCanonicalKeys) {
    for (int cp_size : {1, 2, 4}) {
        auto              config = makeTestConfig();
        ParallelismConfig par;
        par.role_type                          = RoleType::PREFILL;
        par.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        par.tp_rank                            = 0;
        par.tp_size                            = cp_size;
        par.prefill_cp_config.kv_cache_sharded = true;
        config.finalizeStorageLayout(par);

        auto resource = makeResource(2, config);
        resource->setBatchCacheKeys(0, {10, 11, 12, 13});
        resource->setBatchCacheKeys(1, {20, 21, 22, 23, 24});
        resource->setLastBlockAligned(false);
        // Build canonical theory keys the same way reportPrefillCacheHitMetrics does:
        // drop unaligned tail key per batch, then CP-project.
        CacheKeysType canonical_theory;
        for (int batch = 0; batch < resource->batchSize(); ++batch) {
            CacheKeysType full_keys = resource->cacheKeys(batch);
            if (!resource->lastBlockAligned() && !full_keys.empty()) {
                full_keys.pop_back();
            }
            const auto canonical_batch = config.canonicalCacheKeys(full_keys);
            canonical_theory.insert(canonical_theory.end(), canonical_batch.begin(), canonical_batch.end());
        }
        const CacheKeysType expected = cp_size == 1 ? CacheKeysType{10, 11, 12, 20, 21, 22, 23} :
                                       cp_size == 2 ? CacheKeysType{11, 21, 23} :
                                                      CacheKeysType{23};
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
    par.role_type                                         = RoleType::PREFILL;
    par.prefill_cp_config.method                          = CPRotateMethod::ALL_GATHER;
    ParallelismConfig layout_parallelism                  = par;
    layout_parallelism.tp_size                            = 2;
    layout_parallelism.prefill_cp_config.kv_cache_sharded = true;
    resolveManagerCpLayout(config, layout_parallelism);
    par = layout_parallelism;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto resource  = makeResource(1, config);
    auto token_ids = makeTokenIds(1, /*seq_len=*/1, seq_size_per_block);

    MallocInfo info{resource, token_ids};

    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, "default"), 1);

    token_ids->setSeqLength(2);
    result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, "default"), 1);
    EXPECT_EQ(resource->cacheKeys(0).size(), 0);
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
    auto layout_parallelism                = par;
    layout_parallelism.tp_size             = 4;
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
    auto       result = mgr->malloc(info);
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
