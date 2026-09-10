#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include <numeric>
#include <optional>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
namespace test {

class BlockPoolTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        createDevice();
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
        block_pool_.reset();
    }

    std::shared_ptr<BlockPool> block_pool_;
    bool                       old_core_dump_on_exception_{false};
};

namespace {

static rtp_llm::ModelConfig makeTestModelConfig(uint32_t num_layers) {
    rtp_llm::ModelConfig m;
    m.num_layers                   = static_cast<int>(num_layers);
    m.max_seq_len                  = 128;
    m.hidden_size                  = 1;
    m.vocab_size                   = 1;
    m.data_type                    = rtp_llm::DataType::TYPE_FP16;
    m.attn_config.use_mla          = false;
    m.attn_config.tokens_per_block = 4;
    m.attn_config.kv_head_num      = 2;
    m.attn_config.size_per_head    = 1;
    m.attn_config.kv_cache_dtype   = KvCacheDataType::INT8;  // enable kv-scale
    m.attn_config.kv_lora_rank     = 0;
    m.attn_config.rope_head_dim    = 0;
    m.attn_config.head_num         = 2;
    // keep other fields default
    return m;
}

static rtp_llm::CacheConfig
makeMtpCacheConfigByCreateSpConfig(uint32_t main_layers, int mtp_module_num, uint32_t block_num) {
    auto score_model_config   = makeTestModelConfig(main_layers);
    auto propose_model_config = makeTestModelConfig(/*num_layers=*/1);

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    rtp_llm::RuntimeConfig runtime_config;

    rtp_llm::KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = static_cast<int>(block_num);

    rtp_llm::SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = mtp_module_num;

    // NOTE: createSpConfig will fill global_layer_ids for main + MTP sub-models.
    auto cfg = rtp_llm::CacheConfigCreator::createSpConfig(score_model_config,
                                                           propose_model_config,
                                                           parallelism_config,
                                                           runtime_config,
                                                           kv_cache_config,
                                                           sp_config,
                                                           /*warm_up_result=*/std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);
    return cfg;
}

}  // namespace

// Initialization Test
TEST_F(BlockPoolTest, ConstructorAndInit) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    ASSERT_NE(block_pool_, nullptr);

    bool init_result = block_pool_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, Glm52SharedIndexerKvCacheIsOptIn) {
    auto model_config                          = makeTestModelConfig(/*num_layers=*/4);
    model_config.model_type                    = "glm_5";
    model_config.attn_config.use_mla           = true;
    model_config.attn_config.is_sparse         = true;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.attn_config.kv_lora_rank      = 512;
    model_config.attn_config.rope_head_dim     = 64;
    model_config.attn_config.indexer_head_dim  = 128;
    model_config.glm52_indexer_kv_slot_mapping = {0, 1, 1, 1};

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    auto legacy_config = SingleConfigCreator::createSingleConfig(model_config, parallelism_config, /*is_mtp=*/false);
    EXPECT_TRUE(legacy_config.layer_to_indexer_kv_slot.empty());
    EXPECT_EQ(legacy_config.kv_scale_size_bytes, legacy_config.layer_num * legacy_config.kv_scale_stride_bytes);

    model_config.enable_glm52_shared_indexer_kv_cache = true;
    auto compact_config = SingleConfigCreator::createSingleConfig(model_config, parallelism_config, /*is_mtp=*/false);
    EXPECT_EQ(compact_config.layer_to_indexer_kv_slot, std::vector<int>({0, 1, 1, 1}));
    EXPECT_EQ(compact_config.kv_scale_size_bytes, 2u * compact_config.kv_scale_stride_bytes);
    EXPECT_LT(compact_config.block_size_bytes, legacy_config.block_size_bytes);
    model_config.glm52_indexer_kv_slot_mapping = {-1, 0, 0, 0};
    EXPECT_ANY_THROW(SingleConfigCreator::createSingleConfig(model_config, parallelism_config, /*is_mtp=*/false));
}

TEST_F(BlockPoolTest, PinnedMlaKeepsIndexerOnGpuAndVersionsRecycledBlocks) {
    auto model = makeTestModelConfig(4);
    model.model_type = "glm_5";
    model.attn_config.use_mla = true;
    model.attn_config.is_sparse = true;
    model.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    model.attn_config.kv_lora_rank = 512;
    model.attn_config.rope_head_dim = 64;
    model.attn_config.indexer_head_dim = 128;
    model.enable_glm52_shared_indexer_kv_cache = true;
    model.glm52_indexer_kv_slot_mapping = {0, 1, 1, 1};
    ParallelismConfig parallelism;
    auto config = SingleConfigCreator::createSingleConfig(model, parallelism, false);
    config.block_num = 8;
    config.dsa_mla_resident_tokens = config.seq_size_per_block;
    config.dsa_mla_hbm_blocks = 3;
    auto pool_ptr = std::make_shared<BlockPool>(BlockPoolConfigHelper::createConfig(config),
                                                AllocationType::DEVICE,
                                                /*use_pinned_cpu_backing=*/false,
                                                /*use_cuda_malloc_backing=*/true);
    auto& pool = *pool_ptr;
    ASSERT_TRUE(pool.init());
    const auto kv = pool.allLayerCacheBase();
    const auto hbm = pool.allLayerHbmCacheBase();
    const auto indexer = pool.allLayerScaleCacheBase();
    for (const auto& layer : kv) {
        EXPECT_FALSE(layer.is_cuda());
        EXPECT_TRUE(layer.is_pinned());
        EXPECT_EQ(layer.size(0), 5);
    }
    EXPECT_TRUE(indexer[0].is_cuda());
    EXPECT_EQ(indexer[0].size(0), 8);
    ASSERT_EQ(hbm.size(), 4u);
    EXPECT_EQ(hbm[0].size(0), 4);  // Three complete blocks + one working-set block.
    EXPECT_TRUE(hbm[0].is_cuda());
    for (int layer = 0; layer < 4; ++layer) {
        for (int id = 0; id < 8; ++id) {
            const auto parts = pool.convertIndexToBuffer(layer, id);
            EXPECT_EQ(parts[0].is_cuda, id < 3);
            const auto& storage = id < 3 ? hbm[layer] : kv[layer];
            EXPECT_EQ(parts[0].addr, storage[id < 3 ? id : id - 3].data_ptr());
            if (parts.size() > 1) {
                EXPECT_TRUE(parts[1].is_cuda);
                EXPECT_EQ(parts[1].addr, indexer[layer][id].data_ptr());
            }
        }
    }
    auto initial_metrics = pool.tierMetricsSnapshots();
    ASSERT_EQ(initial_metrics.size(), 2u);
    EXPECT_EQ(initial_metrics[0].total_blocks, 2u);  // Block zero is reserved.
    EXPECT_EQ(initial_metrics[1].total_blocks, 5u);
    EXPECT_EQ(initial_metrics[0].occupied_bytes, 0u);
    EXPECT_EQ(initial_metrics[1].occupied_bytes, 0u);
    EXPECT_TRUE(indexer[1].is_cuda());
    EXPECT_FALSE(indexer[2].defined());
    EXPECT_FALSE(indexer[3].defined());
#if USING_CUDA
    // Expandable VMM allocations cannot provide the legacy IPC/RDMA handle.
    cudaIpcMemHandle_t ipc_handle;
    EXPECT_EQ(cudaIpcGetMemHandle(&ipc_handle, indexer[0].data_ptr()), cudaSuccess);
#endif
    auto blocks = pool.malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const int block = blocks.front();
    const auto parts = pool.convertIndexToBuffer(0, block);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_TRUE(parts[0].is_cuda);
    EXPECT_TRUE(parts[1].is_cuda);
    const auto* generations = pool.blockGenerations().data_ptr<int64_t>();
    EXPECT_EQ(generations[block], 1);
    pool.requestFree(blocks);
    EXPECT_EQ(pool.malloc(1).front(), block);
    EXPECT_EQ(generations[block], 2);

    // Prefix-cache eviction only drops its own reference. Requests and an
    // in-flight transfer must keep both pinned MLA and HBM indexer alive.
    auto rest = pool.malloc(6);
    ASSERT_EQ(rest.size(), 6u);
    auto full_metrics = pool.tierMetricsSnapshots();
    EXPECT_EQ(full_metrics[0].free_blocks, 0u);
    EXPECT_EQ(full_metrics[1].free_blocks, 0u);
    EXPECT_EQ(full_metrics[0].occupied_bytes, full_metrics[0].capacity_bytes);
    EXPECT_EQ(full_metrics[1].occupied_bytes, full_metrics[1].capacity_bytes);
    SharedBlockCache cache;
    cache.init(1, {pool_ptr});
    cache.setPrefixTreeEnabled(false);
    cache.put(1, {block}, false);
    cache.put(2, {rest[0]}, false);
    cache.put(3, {rest[1]}, true);
    cache.put(4, {rest[2]}, false);
    pool.connectorReference(rest[0]);
    pool.requestFree({rest[0], rest[1], rest[2]});
    EXPECT_EQ(pool.freeBlocksNum(), 0u);
    EXPECT_EQ(cache.matchGroup(1, 0), block);  // Touch key 1, making key 2 oldest.

    SharedBlockCache::EvictResult evicted;
    cache.evictAndFreeForGroup(0, 1, &evicted);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    EXPECT_EQ(pool.freeBlocksNum(), 0u);
    EXPECT_EQ(generations[rest[0]], 1);
    pool.connectorFree(rest[0]);
    EXPECT_EQ(pool.freeBlocksNum(), 1u);
    ASSERT_EQ(pool.malloc(1), (BlockIndicesType{rest[0]}));
    EXPECT_EQ(generations[rest[0]], 2);

    cache.evictAndFreeForGroup(0, 1, &evicted);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{4}));
    ASSERT_EQ(pool.malloc(1), (BlockIndicesType{rest[2]}));
    EXPECT_EQ(generations[rest[2]], 2);
    cache.evictAndFreeForGroup(0, 1, &evicted);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    EXPECT_EQ(pool.freeBlocksNum(), 0u);  // Active request still owns this block.
    EXPECT_EQ(generations[block], 2);
    EXPECT_TRUE(cache.contains(3));  // Explicitly resident prefixes never evict.
    EXPECT_TRUE(cache.selectAndEvictForGroup(0, 1).evicted_keys.empty());
    pool.requestFree(block);
    ASSERT_EQ(pool.malloc(1), (BlockIndicesType{block}));
    EXPECT_EQ(generations[block], 3);

    // The default prefix-tree policy evicts a cold leaf before a shared parent.
    SharedBlockCache tree_cache;
    tree_cache.init(1, {pool_ptr});
    tree_cache.setPrefixTreeEnabled(true);
    tree_cache.put(10, {block}, false);
    BlockDependency child;
    child.has_parent = true;
    child.parent_key = 10;
    child.ordinal = 1;
    tree_cache.put(11, {rest[0]}, false, SharedBlockCache::kDefaultNamespace, child);
    tree_cache.put(12, {rest[2]}, false, SharedBlockCache::kDefaultNamespace, child);
    pool.requestFree({block, rest[0], rest[2]});
    EXPECT_EQ(tree_cache.matchGroup(11, 0), rest[0]);
    tree_cache.evictAndFreeForGroup(0, 1, &evicted);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{12}));
    EXPECT_TRUE(tree_cache.contains(10));
    EXPECT_TRUE(tree_cache.contains(11));
    ASSERT_EQ(pool.malloc(1), (BlockIndicesType{rest[2]}));
    EXPECT_EQ(generations[rest[2]], 3);
    tree_cache.evictAndFreeForGroup(0, 1, &evicted);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{10, 11}));
    EXPECT_EQ(pool.freeBlocksNum(), 2u);
    EXPECT_TRUE(tree_cache.empty());
}

// Scale the 48 x 16K + 16 x 192K workload to one/twelve blocks per query.
// Under long-first arrival, adaptive admission leaves HBM available for short queries.
TEST_F(BlockPoolTest, PinnedMlaLengthAwareAllocation) {
    auto model = makeTestModelConfig(1);
    model.model_type = "glm_5";
    model.attn_config.use_mla = true;
    model.attn_config.is_sparse = true;
    model.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    model.attn_config.kv_lora_rank = 512;
    model.attn_config.rope_head_dim = 64;
    model.attn_config.indexer_head_dim = 128;
    auto config = SingleConfigCreator::createSingleConfig(model, ParallelismConfig(), false);
    config.block_num = 241;
    config.dsa_mla_hbm_blocks = 49;
    config.dsa_mla_resident_tokens = config.seq_size_per_block;
    for (const auto denominator : {"0", "3"}) {
        autil::EnvGuard guard("RTP_LLM_DSA_MLA_HBM_SHARE_DENOMINATOR", denominator);
        auto pool = std::make_shared<BlockPool>(BlockPoolConfigHelper::createConfig(config));
        ASSERT_TRUE(pool->init());
        FullKVCacheGroup group({0}, config.cache_specs[0], pool, 0);
        ASSERT_TRUE(group.init());
        for (int round = 0; round < 3; ++round) {
            std::vector<BlockIds> requests(64);
            int host_queries = 0;
            for (int i = 0; i < 64; ++i) {
                const bool is_long = round == 1 ? i >= 48 : i < 16;
                ASSERT_TRUE(group.initMalloc(requests[i], (is_long ? 12 : 1) * config.seq_size_per_block));
                const auto& blocks = requests[i].blocks();
                const bool on_host = std::any_of(blocks.begin(), blocks.end(), [](int id) { return id >= 49; });
                host_queries += on_host;
            }
            EXPECT_EQ(host_queries, std::string(denominator) == "0" ? (round == 1 ? 16 : 60) :
                                                                    (round == 1 ? 18 : 38));
            EXPECT_EQ(pool->freeBlocksNum(), 0u);
            EXPECT_TRUE(pool->malloc(1, 48).empty());
            for (const auto& request : requests) {
                group.free(request.blocks());
            }
            EXPECT_EQ(pool->freeBlocksNum(), 240u);
        }
        const auto* generations = pool->blockGenerations().data_ptr<int64_t>();
        for (int id = 1; id < 241; ++id) {
            EXPECT_EQ(generations[id], 3);
        }
    }
}

TEST_F(BlockPoolTest, PinnedMlaLengthAwareGrowthAndFallback) {
    autil::EnvGuard guard("RTP_LLM_DSA_MLA_HBM_SHARE_DENOMINATOR", "3");
    auto model = makeTestModelConfig(1);
    model.model_type = "glm_5";
    model.attn_config.use_mla = true;
    model.attn_config.is_sparse = true;
    model.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    model.attn_config.kv_lora_rank = 512;
    model.attn_config.rope_head_dim = 64;
    model.attn_config.indexer_head_dim = 128;
    auto config = SingleConfigCreator::createSingleConfig(model, ParallelismConfig(), false);
    config.block_num = 20;
    config.dsa_mla_hbm_blocks = 10;
    config.dsa_mla_resident_tokens = config.seq_size_per_block;
    auto pool = std::make_shared<BlockPool>(BlockPoolConfigHelper::createConfig(config));
    ASSERT_TRUE(pool->init());
    FullKVCacheGroup group({0}, config.cache_specs[0], pool, 0);
    ASSERT_TRUE(group.init());
    BlockIds long_query, short_query;
    ASSERT_TRUE(group.malloc(long_query, 16));  // Four blocks > nine free HBM / 3.
    EXPECT_EQ(long_query.blocks(), (BlockIndicesType{10, 11, 12, 13}));
    ASSERT_TRUE(group.malloc(short_query, 8));
    EXPECT_EQ(short_query.blocks(), (BlockIndicesType{1, 2}));
    ASSERT_TRUE(group.malloc(long_query, 20));  // Incremental allocation follows host.
    EXPECT_EQ(long_query.blocks(), (BlockIndicesType{10, 11, 12, 13, 14}));
    ASSERT_TRUE(group.malloc(short_query, 12));  // HBM query keeps its tier.
    EXPECT_EQ(short_query.blocks(), (BlockIndicesType{1, 2, 3}));
    const auto free_before = pool->freeBlocksNum();
    EXPECT_TRUE(pool->malloc(12, 48).empty());
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
    EXPECT_EQ(pool->malloc(11, 48), (BlockIndicesType{15, 16, 17, 18, 19, 4, 5, 6, 7, 8, 9}));
    pool->connectorReference(10);
    group.free(long_query.blocks());
    EXPECT_EQ(pool->malloc(4, 8), (BlockIndicesType{11, 12, 13, 14}));  // HBM exhausted.
    EXPECT_TRUE(pool->malloc(1, 48).empty());  // In-flight RDMA still owns 10.
    pool->connectorFree(10);
    EXPECT_EQ(pool->malloc(1, 48), (BlockIndicesType{10}));
    EXPECT_EQ(pool->blockGenerations().data_ptr<int64_t>()[10], 2);

    // Every reference kind must update free HBM exactly once, even when
    // references overlap; freeing an active connector cannot inflate capacity.
    group.free(short_query.blocks());
    pool->requestFree({4, 5, 6, 7, 8, 9});
    pool->blockCacheReference({1, 2, 3});
    pool->connectorReference({1, 2, 3});
    pool->blockCacheFree({1, 2, 3});
    pool->requestFree({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    EXPECT_EQ(pool->malloc(3, 12), (BlockIndicesType{10, 11, 12}));  // Six free HBM / 3 = two.
    pool->connectorFree({1, 2, 3});
    EXPECT_EQ(pool->malloc(3, 12), (BlockIndicesType{1, 2, 3}));  // Exact nine / 3 boundary.
    BlockIds reused_query;
    group.reference(reused_query, {1, 2, 3});
    ASSERT_TRUE(group.initMalloc(reused_query, 16));
    EXPECT_EQ(reused_query.blocks(), (BlockIndicesType{1, 2, 3, 13}));  // Long suffix goes to host.
    EXPECT_EQ(pool->blockGenerations().data_ptr<int64_t>()[1], 2);  // Prefix was only referenced.

}

TEST_F(BlockPoolTest, PinnedMlaRoundsAutomaticWorkingSetToPhysicalBlocks) {
    autil::EnvGuard host_budget("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "256");
    autil::EnvGuard automatic_resident("RTP_LLM_DSA_MLA_RESIDENT_TOKENS", "0");
    auto model = makeTestModelConfig(4);
    model.model_type = "glm_5";
    model.attn_config.use_mla = true;
    model.attn_config.is_sparse = true;
    model.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    model.attn_config.kv_lora_rank = 512;
    model.attn_config.rope_head_dim = 64;
    model.attn_config.indexer_head_dim = 128;
    model.attn_config.indexer_topk = 2048;
    model.attn_config.tokens_per_block = 4096;
    model.enable_glm52_shared_indexer_kv_cache = true;
    model.glm52_indexer_kv_slot_mapping = {0, 1, 1, 1};
    ParallelismConfig parallelism;
    RuntimeConfig runtime;
    runtime.max_generate_batch_size = 3;
    KVCacheConfig cache_options;
    cache_options.seq_size_per_block = 4096;
    cache_options.kernel_seq_size_per_block = 64;
    cache_options.test_block_num = 16;
    auto create = [&] {
        return CacheConfigCreator::createConfig(model, parallelism, runtime, cache_options, std::nullopt);
    };

    const auto automatic = create();
    EXPECT_GE(automatic.dsa_mla_resident_tokens, 8192u);  // 3 * 2048 rounds up.
    EXPECT_EQ(automatic.dsa_mla_resident_tokens % 4096, 0u);
    const auto pool_config = BlockPoolConfigHelper::createConfig(automatic);
    EXPECT_GT(automatic.dsa_mla_hbm_blocks, 0u);
    EXPECT_GT(automatic.block_num, automatic.dsa_mla_hbm_blocks);
    EXPECT_LE(pool_config.total_size_bytes, 256u * 1024 * 1024);
    size_t hbm_bytes = 0;
    for (const auto& layout : pool_config.memory_layouts) {
        hbm_bytes += layout.mla_hbm_size_bytes + layout.kv_scale_pool_size_bytes;
    }
    EXPECT_LE(hbm_bytes, cache_options.test_block_num * automatic.block_size_bytes);
    {
        autil::EnvGuard explicit_resident("RTP_LLM_DSA_MLA_RESIDENT_TOKENS", "8192");
        EXPECT_EQ(create().dsa_mla_resident_tokens, 8192u);
    }
    {
        autil::EnvGuard oversized_pin_budget("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "1048576");
        EXPECT_GE(create().dsa_mla_hbm_blocks, 2u);  // Reserved zero + usable HBM.
    }
    for (const char* invalid : {"6144", "4096"}) {
        autil::EnvGuard invalid_resident("RTP_LLM_DSA_MLA_RESIDENT_TOKENS", invalid);
        EXPECT_ANY_THROW(create());  // Unaligned, or too small for the batch.
    }
}

TEST_F(BlockPoolTest, Glm52CompactScoreKeepsMtpIndexerLayoutIndependent) {
    auto score_model_config                                 = makeTestModelConfig(/*num_layers=*/4);
    score_model_config.model_type                           = "glm_5";
    score_model_config.attn_config.use_mla                  = true;
    score_model_config.attn_config.is_sparse                = true;
    score_model_config.attn_config.kv_cache_dtype           = KvCacheDataType::FP8;
    score_model_config.attn_config.kv_lora_rank             = 512;
    score_model_config.attn_config.rope_head_dim            = 64;
    score_model_config.attn_config.indexer_head_dim         = 128;
    score_model_config.enable_glm52_shared_indexer_kv_cache = true;
    score_model_config.glm52_indexer_kv_slot_mapping        = {0, 1, 1, 1};

    auto propose_model_config                                 = score_model_config;
    propose_model_config.num_layers                           = 1;
    propose_model_config.enable_glm52_shared_indexer_kv_cache = false;
    propose_model_config.glm52_indexer_kv_slot_mapping.clear();

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    rtp_llm::RuntimeConfig runtime_config;
    rtp_llm::KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 4;
    rtp_llm::SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;

    auto cache_config = CacheConfigCreator::createSpConfig(score_model_config,
                                                           propose_model_config,
                                                           parallelism_config,
                                                           runtime_config,
                                                           kv_cache_config,
                                                           sp_config,
                                                           /*warm_up_result=*/std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);
    auto pool_config  = BlockPoolConfigHelper::createConfig(cache_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);
    EXPECT_EQ(pool_config.memory_layouts[0].scale_layer_num, 2u);
    EXPECT_EQ(pool_config.memory_layouts[1].scale_layer_num, 1u);
    ASSERT_EQ(cache_config.mtp_sub_configs.size(), 1u);
    EXPECT_TRUE(cache_config.mtp_sub_configs[0]->layer_to_indexer_kv_slot.empty());
}

TEST_F(BlockPoolTest, PinnedMlaSharedIndexerMtpBudgetCoversAllQueries) {
    autil::EnvGuard host_budget("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "256");
    autil::EnvGuard resident_budget("RTP_LLM_DSA_MLA_RESIDENT_TOKENS", "0");
    auto score_model_config                                 = makeTestModelConfig(/*num_layers=*/4);
    score_model_config.model_type                           = "glm_5";
    score_model_config.attn_config.use_mla                  = true;
    score_model_config.attn_config.is_sparse                = true;
    score_model_config.attn_config.kv_cache_dtype           = KvCacheDataType::FP8;
    score_model_config.attn_config.kv_lora_rank             = 512;
    score_model_config.attn_config.rope_head_dim            = 64;
    score_model_config.attn_config.indexer_head_dim         = 128;
    score_model_config.enable_glm52_shared_indexer_kv_cache = true;
    score_model_config.glm52_indexer_kv_slot_mapping        = {0, 1, 1, 1};

    score_model_config.attn_config.indexer_topk = 2048;
    score_model_config.attn_config.tokens_per_block = 64;
    auto propose_model_config                                 = score_model_config;
    propose_model_config.num_layers                           = 1;
    propose_model_config.enable_glm52_shared_indexer_kv_cache = false;
    propose_model_config.glm52_indexer_kv_slot_mapping.clear();

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    rtp_llm::RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size = 8;
    rtp_llm::KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 2048;
    kv_cache_config.seq_size_per_block = 64;
    rtp_llm::SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 3;

    auto cache_config = CacheConfigCreator::createSpConfig(score_model_config,
                                                           propose_model_config,
                                                           parallelism_config,
                                                           runtime_config,
                                                           kv_cache_config,
                                                           sp_config,
                                                           /*warm_up_result=*/std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);
    auto pool_config  = BlockPoolConfigHelper::createConfig(cache_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 4u);
    EXPECT_EQ(pool_config.memory_layouts[0].scale_layer_num, 2u);
    EXPECT_EQ(pool_config.memory_layouts[1].scale_layer_num, 1u);
    ASSERT_EQ(cache_config.mtp_sub_configs.size(), 3u);
    EXPECT_TRUE(cache_config.mtp_sub_configs[0]->layer_to_indexer_kv_slot.empty());
    EXPECT_EQ(cache_config.dsa_mla_resident_tokens, 8u * 4 * 2048);
    EXPECT_GT(cache_config.dsa_mla_hbm_blocks, 0u);
    EXPECT_LE(pool_config.total_size_bytes, 256u * 1024 * 1024);
    for (const auto& layout : pool_config.memory_layouts) {
        EXPECT_EQ(layout.block_num, cache_config.block_num);
        EXPECT_EQ(layout.mla_hbm_blocks, cache_config.dsa_mla_hbm_blocks);
        EXPECT_EQ(layout.mla_resident_tokens, cache_config.dsa_mla_resident_tokens);
        EXPECT_EQ(layout.kv_scale_pool_size_bytes,
                  layout.scale_layer_num * layout.block_num * layout.kv_scale_stride_bytes);
    }

}

TEST_F(BlockPoolTest, MTPConvertIndexGlobalIdMapping) {
    // Use createSpConfig logic so that global_layer_ids is filled for main + sub-model layers.
    // main(2 layers) + mtp1(1 layer) + mtp2(1 layer)
    auto cache_cfg = makeMtpCacheConfigByCreateSpConfig(/*main_layers=*/2, /*mtp_module_num=*/2, /*block_num=*/4);

    ASSERT_FALSE(cache_cfg.global_layer_ids.empty());
    ASSERT_EQ(cache_cfg.global_layer_ids[0].size(), static_cast<size_t>(cache_cfg.layer_all_num));

    ASSERT_EQ(cache_cfg.mtp_sub_configs.size(), 2u);
    ASSERT_NE(cache_cfg.mtp_sub_configs[0], nullptr);
    ASSERT_NE(cache_cfg.mtp_sub_configs[1], nullptr);
    ASSERT_EQ(cache_cfg.mtp_sub_configs[0]->groupNums(), 1);
    ASSERT_EQ(cache_cfg.mtp_sub_configs[1]->groupNums(), 1);
    EXPECT_EQ(cache_cfg.mtp_sub_configs[0]->cache_specs[0]->block_size_bytes(),
              cache_cfg.mtp_sub_configs[1]->cache_specs[0]->block_size_bytes());

    ASSERT_FALSE(cache_cfg.mtp_sub_configs[0]->global_layer_ids.empty());
    ASSERT_FALSE(cache_cfg.mtp_sub_configs[1]->global_layer_ids.empty());
    ASSERT_EQ(cache_cfg.mtp_sub_configs[0]->global_layer_ids[0].size(), 1u);
    ASSERT_EQ(cache_cfg.mtp_sub_configs[1]->global_layer_ids[0].size(), 1u);
    EXPECT_EQ(cache_cfg.mtp_sub_configs[0]->global_layer_ids[0][0], 2);
    EXPECT_EQ(cache_cfg.mtp_sub_configs[1]->global_layer_ids[0][0], 3);

    auto pool_cfg = rtp_llm::BlockPoolConfigHelper::createConfig(cache_cfg);
    ASSERT_EQ(pool_cfg.memory_layouts.size(), 3u);
    ASSERT_EQ(pool_cfg.memory_layouts[0].layer_num, 2u);
    ASSERT_EQ(pool_cfg.memory_layouts[1].layer_num, 1u);
    ASSERT_EQ(pool_cfg.memory_layouts[2].layer_num, 1u);

    block_pool_ = std::make_shared<BlockPool>(pool_cfg);
    ASSERT_TRUE(block_pool_->init());

    const int global_main = 0;
    const int global_mtp1 = static_cast<int>(pool_cfg.memory_layouts[0].layer_num);
    const int global_mtp2 = global_mtp1 + 1;

    const int block_id  = 1;
    auto      base_addr = block_pool_->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/0);
    ASSERT_NE(base_addr.kv_addr, nullptr);
    ASSERT_NE(base_addr.kv_scale_addr, nullptr);
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_addr.kv_addr);

    auto verify_one = [&](int global_layer, size_t expect_layout_idx, int expect_local_layer) {
        const auto& layout_cfg = pool_cfg.memory_layouts[expect_layout_idx];
        auto        addr       = block_pool_->convertIndexToAddr(global_layer, block_id);
        ASSERT_NE(addr.kv_addr, nullptr);
        ASSERT_NE(addr.kv_scale_addr, nullptr);

        const size_t idx_in_layout = static_cast<size_t>(expect_local_layer) * static_cast<size_t>(layout_cfg.block_num)
                                     + static_cast<size_t>(block_id);
        const size_t expect_kv_off =
            layout_cfg.kv_cache_offset_bytes + idx_in_layout * layout_cfg.kv_block_stride_bytes;
        const size_t expect_sc_off =
            layout_cfg.kv_scale_offset_bytes + idx_in_layout * layout_cfg.kv_scale_stride_bytes;

        EXPECT_EQ(reinterpret_cast<uintptr_t>(addr.kv_addr) - base, expect_kv_off);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(addr.kv_scale_addr) - base, expect_sc_off);

        auto buf = block_pool_->convertIndexToBuffer(global_layer, block_id);
        ASSERT_EQ(buf.size(), 2u);
        ASSERT_NE(buf[0].addr, nullptr);
        ASSERT_NE(buf[1].addr, nullptr);
        EXPECT_EQ(buf[0].addr, addr.kv_addr);
        EXPECT_EQ(buf[0].size_bytes, layout_cfg.kv_block_stride_bytes);
        EXPECT_EQ(buf[1].addr, addr.kv_scale_addr);
        EXPECT_EQ(buf[1].size_bytes, layout_cfg.kv_scale_stride_bytes);
    };

    verify_one(global_main, /*expect_layout_idx=*/0, /*expect_local_layer=*/0);
    verify_one(global_mtp1, /*expect_layout_idx=*/1, /*expect_local_layer=*/0);
    verify_one(global_mtp2, /*expect_layout_idx=*/2, /*expect_local_layer=*/0);

    // Partitioned buffer correctness on mtp layer (heads=2, partition_count=2, partition_id=1)
    const auto& mtp_layout_cfg = pool_cfg.memory_layouts[1];
    auto        addr_mtp1      = block_pool_->convertIndexToAddr(global_mtp1, block_id);
    ASSERT_NE(addr_mtp1.kv_addr, nullptr);
    ASSERT_NE(addr_mtp1.kv_scale_addr, nullptr);
    auto parts = block_pool_->convertIndexToBuffer(global_mtp1, block_id, /*partition_count=*/2, /*partition_id=*/1);
    ASSERT_EQ(parts.size(), 4u);
    ASSERT_NE(parts[0].addr, nullptr);
    ASSERT_NE(parts[1].addr, nullptr);
    ASSERT_NE(parts[2].addr, nullptr);
    ASSERT_NE(parts[3].addr, nullptr);
    EXPECT_EQ(parts[0].size_bytes, mtp_layout_cfg.k_block_stride_bytes / 2);
    EXPECT_EQ(parts[1].size_bytes, mtp_layout_cfg.v_block_stride_bytes / 2);
    EXPECT_EQ(parts[2].size_bytes, mtp_layout_cfg.k_scale_stride_bytes / 2);
    EXPECT_EQ(parts[3].size_bytes, mtp_layout_cfg.v_scale_stride_bytes / 2);

    const size_t k_bytes_per_head = mtp_layout_cfg.k_block_stride_bytes / 2;
    const size_t v_bytes_per_head = mtp_layout_cfg.v_block_stride_bytes / 2;
    const size_t k_off            = k_bytes_per_head;
    const size_t v_off            = mtp_layout_cfg.k_block_stride_bytes + v_bytes_per_head;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(parts[0].addr) - reinterpret_cast<uintptr_t>(addr_mtp1.kv_addr), k_off);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(parts[1].addr) - reinterpret_cast<uintptr_t>(addr_mtp1.kv_addr), v_off);

    const size_t sc_bytes_per_head = mtp_layout_cfg.k_scale_stride_bytes / 2;
    const size_t sc_k_off          = sc_bytes_per_head;
    const size_t sc_v_off          = mtp_layout_cfg.k_scale_stride_bytes + sc_bytes_per_head;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(parts[2].addr) - reinterpret_cast<uintptr_t>(addr_mtp1.kv_scale_addr),
              sc_k_off);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(parts[3].addr) - reinterpret_cast<uintptr_t>(addr_mtp1.kv_scale_addr),
              sc_v_off);
}

// Allocation Test
TEST_F(BlockPoolTest, AllocSingleBlock) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(1);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_GE(blocks[0], 0);
    EXPECT_LT(blocks[0], static_cast<BlockIdxType>(config.block_num));
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 2);
}

TEST_F(BlockPoolTest, AllocMultipleBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int  alloc_count = 5;
    auto blocks      = block_pool_->malloc(alloc_count);
    EXPECT_EQ(blocks.size(), alloc_count);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - alloc_count - 1);

    std::set<BlockIdxType> unique_blocks(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_blocks.size(), alloc_count);
}

TEST_F(BlockPoolTest, AllocAllBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(config.block_num - 1);
    EXPECT_EQ(blocks.size(), config.block_num - 1);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 0);
}

TEST_F(BlockPoolTest, AllocMoreThanAvailable) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks1 = block_pool_->malloc(5);
    EXPECT_EQ(blocks1.size(), 5);

    auto blocks2 = block_pool_->malloc(10);
    EXPECT_EQ(blocks2.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);
}

// Free Test
TEST_F(BlockPoolTest, FreeBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreePartialBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    std::vector<BlockIdxType> partial_blocks(blocks.begin(), blocks.begin() + 3);
    block_pool_->requestFree(partial_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);
}

TEST_F(BlockPoolTest, ReferenceAndFree) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();
    auto total_blocks = block_pool_->freeBlocksNum();

    {
        auto blocks = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }

    // Blocks referred to by the block cache do not affect the freeblocks count.
    // Blocks referred to by the block cache do not affect the available blocks count.
    {
        auto blocks2 = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->blockCacheReference(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);

        block_pool_->blockCacheFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }

    {
        auto blocks = block_pool_->malloc(2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);

        block_pool_->blockCacheReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);

        block_pool_->connectorReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 2);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 2);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 2);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 0);

        block_pool_->connectorFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 0);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 0);

        block_pool_->blockCacheFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }
}

TEST_F(BlockPoolTest, MultipleReferencesAndFrees) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(2);

    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);

    // free for 4 times (1 + 3)
    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

// Convert Index to Addr Test
TEST_F(BlockPoolTest, ConvertIndexToAddr) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    const auto layer_num = static_cast<int>(config.memory_layouts[0].layer_num);
    for (int layer = 0; layer < layer_num; ++layer) {
        for (int block = 0; block < 3; ++block) {
            auto addr_info = block_pool_->convertIndexToAddr(layer, block);
            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(BlockPoolTest, ConvertIndexToBuffer) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int layer = 0;
    int block = 0;

    auto buffer_info = block_pool_->convertIndexToBuffer(layer, block);
    ASSERT_EQ(buffer_info.size(), 1u);
    EXPECT_NE(buffer_info[0].addr, nullptr);
}

TEST_F(BlockPoolTest, ConvertIndexToAddrAndBufferWithScale) {
    // dtype=int8 will enable kv-scale pool automatically in BlockPoolConfigHelper.
    auto config = createTestConfig(
        /*k_block_stride_bytes=*/512,
        /*v_block_stride_bytes=*/512,
        /*k_scale_stride_bytes=*/128,
        /*v_scale_stride_bytes=*/128,
        /*dtype=*/rtp_llm::DataType::TYPE_INT8,
        /*local_head_num_kv=*/2,
        /*seq_size_per_block=*/4);

    block_pool_ = std::make_shared<BlockPool>(config);
    ASSERT_TRUE(block_pool_->init());

    const auto& layout_cfg = config.memory_layouts[0];
    const int   layer      = 0;
    const int   block      = 0;
    auto        addr       = block_pool_->convertIndexToAddr(layer, block);
    EXPECT_NE(addr.kv_addr, nullptr);
    EXPECT_NE(addr.kv_scale_addr, nullptr);

    auto buf = block_pool_->convertIndexToBuffer(layer, block);
    ASSERT_EQ(buf.size(), 2u);
    EXPECT_NE(buf[0].addr, nullptr);
    EXPECT_NE(buf[1].addr, nullptr);
    EXPECT_EQ(buf[1].size_bytes, layout_cfg.kv_scale_stride_bytes);
}

// LayerCache Base Test
TEST_F(BlockPoolTest, LayerCacheBase) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto layer_tensors = block_pool_->allLayerCacheBase();
    EXPECT_EQ(layer_tensors.size(), config.memory_layouts[0].layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_GT(layer_tensors[i].numel(), 0);
    }
}

// Boundary Condition Test
TEST_F(BlockPoolTest, AllocZeroBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(0);
    EXPECT_EQ(blocks.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreeEmptyVector) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    std::vector<BlockIdxType> empty_blocks;
    block_pool_->requestFree(empty_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, OutOfRangeLayerId) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int invalid_layer = static_cast<int>(config.memory_layouts[0].layer_num) + 10;
    EXPECT_THROW((void)block_pool_->convertIndexToAddr(invalid_layer, 0), rtp_llm::RTPException);
}

TEST_F(BlockPoolTest, AllocFreeAllocCycle) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    for (int i = 0; i < 5; ++i) {
        auto blocks = block_pool_->malloc(5);
        EXPECT_EQ(blocks.size(), 5);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
    }
}

TEST_F(BlockPoolTest, MixedAllocFreeOperations) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    std::vector<std::vector<BlockIdxType>> allocated_blocks;

    allocated_blocks.push_back(block_pool_->malloc(2));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 7);

    allocated_blocks.push_back(block_pool_->malloc(3));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 4);

    block_pool_->requestFree(allocated_blocks[0]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 6);

    allocated_blocks.push_back(block_pool_->malloc(4));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 2);

    block_pool_->requestFree(allocated_blocks[1]);
    block_pool_->requestFree(allocated_blocks[2]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 9);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
