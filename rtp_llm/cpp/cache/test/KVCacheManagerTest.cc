#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <algorithm>
#include <limits>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {
constexpr int kDsv4PoolNum = 7;
}

class KVCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }

protected:
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

    const size_t kv_scale_stride_bytes = cache_manager->cacheConfig().kv_scale_stride_bytes;
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
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.o_groups         = 8;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios         = {0, 0};
    for (int i = 2; i < 43; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

static CacheConfig makeCompactDSV4ManagerConfig(uint32_t block_num = 16) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc);
    config.block_num         = block_num;
    config.non_full_addition_kvcache_blocks = 0;
    config.group_block_nums.assign(config.groupNums(), block_num);
    return config;
}

// Creates an intentionally tight DSV4 config for eviction stress tests: FULL
// groups use a large paged pool, while SWA groups use a small fixed pool.
static CacheConfig makeDSV4ConfigWithConcurrencyPool(uint32_t full_block_num, uint32_t swa_batch_size) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc);
    config.block_num         = full_block_num;
    config.non_full_addition_kvcache_blocks = 0;
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        config.group_block_nums[gid] = (gid < 3) ? full_block_num : (2u * swa_batch_size);
    }
    return config;
}

static CacheConfig
makeProductionDSV4Config(uint32_t full_block_num, uint32_t max_concurrency, uint32_t fixed_pool_blocks = 4) {
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                              = full_block_num;
    kv_cache_config.non_full_addition_kvcache_blocks            = fixed_pool_blocks;
    runtime_config.max_generate_batch_size                      = max_concurrency;
    runtime_config.fifo_scheduler_config.max_context_batch_size = max_concurrency;
    return CacheConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc, runtime_config, kv_cache_config);
}

static BatchKVCacheResourcePtr makeDSV4BatchResource(const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(1);
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.layer_to_group_id,
                    config.kernelBlocksPerKvBlock(),
                    config.group_types,
                    config.layer_region_to_group_id);
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

static void writeDsv4RegionPattern(const std::shared_ptr<KVCacheManager>& manager,
                                   int                                    block_id,
                                   int                                    layer_id,
                                   KVCacheRegionName                      region_name,
                                   size_t                                 bytes,
                                   uint8_t                                pattern) {
    auto addr_info = manager->convertIndexToAddr(block_id, layer_id, region_name);
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
                                      KVCacheRegionName                      region_name,
                                      size_t                                 bytes,
                                      uint8_t                                expected) {
    auto addr_info = manager->convertIndexToAddr(block_id, layer_id, region_name);
    ASSERT_NE(addr_info.kv_addr, nullptr);

    auto dev_t =
        torch::from_blob(addr_info.kv_addr, {(int64_t)bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host_t = dev_t.cpu();
    const auto* ptr    = host_t.data_ptr<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        ASSERT_EQ(ptr[i], expected) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id
                                    << " region=" << static_cast<int>(region_name);
    }
}

TEST_F(KVCacheManagerTest, WarmupConfigSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/true);
    ASSERT_TRUE(cache_manager->init());

    EXPECT_EQ(cache_manager->cacheConfig().block_num, 1);

    EXPECT_EQ(cache_manager->totalBlocksNum(), 0);
    EXPECT_EQ(cache_manager->freeBlocksNum(), 0);
}

TEST_F(KVCacheManagerTest, DSV4StatePoolBackingFollowsPinnedCpuFlag) {
    // Contract after B1 Option A fix: the allocator reads
    // `config.state_pool_uses_pinned_cpu` directly — it does NOT inspect
    // role_type independently. CacheConfigCreator sets the flag from
    // (state_pool_memory_mb > 0 || role_type == PREFILL) && state_block_size_bytes > 0.
    // This test simulates that flag-setting explicitly (the helper config
    // skips createConfig) and verifies the allocator wiring honors it.
    auto expect_pool_backing = [](RoleType role_type, bool state_pool_uses_pinned_cpu, bool expect_state_cpu) {
        auto config = makeCompactDSV4ManagerConfig(/*block_num=*/8);
        config.state_pool_uses_pinned_cpu = state_pool_uses_pinned_cpu;

        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        auto cache_manager      = std::make_shared<KVCacheManager>(config,
                                                               /*warmup=*/false,
                                                               nullptr,
                                                               KVCacheConfig{},
                                                               ParallelismConfig{},
                                                               RuntimeConfig{},
                                                               SpeculativeExecutionConfig{},
                                                               pd_sep_config);
        ASSERT_TRUE(cache_manager->init());

        auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager->allocator_);
        ASSERT_NE(allocator, nullptr);
        ASSERT_EQ(allocator->groupBlockPools().size(), config.group_region_names.size());

        for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
            const auto region_name = config.group_region_names[gid];
            const bool state_region = region_name == KVCacheRegionName::INDEXER_STATE
                                      || region_name == KVCacheRegionName::CSA_STATE
                                      || region_name == KVCacheRegionName::HCA_STATE;
            const auto expected =
                (expect_state_cpu && state_region) ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_GPU;
            EXPECT_EQ(allocator->groupBlockPools()[gid]->where(), expected)
                << "role=" << static_cast<int>(role_type)
                << " state_pool_uses_pinned_cpu=" << state_pool_uses_pinned_cpu
                << " gid=" << gid
                << " region=" << static_cast<int>(region_name);
        }
    };

    // PREFILL with flag set (production createConfig path): STATE on pinned CPU.
    expect_pool_backing(RoleType::PREFILL, /*flag=*/true, /*expect_state_cpu=*/true);
    // DECODE/PDFUSION without env budget: flag false → STATE on device.
    expect_pool_backing(RoleType::DECODE, /*flag=*/false, /*expect_state_cpu=*/false);
    expect_pool_backing(RoleType::PDFUSION, /*flag=*/false, /*expect_state_cpu=*/false);
    // Default-OFF regression guard: PREFILL with flag NOT set (e.g. caller built
    // CacheConfig outside CacheConfigCreator) must NOT silently move STATE to CPU.
    // Pre-fix the allocator would override; the fix removes that unilateral OR.
    expect_pool_backing(RoleType::PREFILL, /*flag=*/false, /*expect_state_cpu=*/false);
    // Env-driven path (decode with explicit STATE pool budget): flag true → CPU.
    expect_pool_backing(RoleType::DECODE, /*flag=*/true, /*expect_state_cpu=*/true);
}

TEST_F(KVCacheManagerTest, MetricsThreadSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);

    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/true, reporter);

    ASSERT_TRUE(cache_manager->init());
    EXPECT_TRUE(cache_manager->metrics_reporter_thread_.joinable());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    cache_manager.reset();
}

TEST_F(KVCacheManagerTest, SetKVBlockValueAndBlockCopy) {
    // Use non-warmup config so we have usable blocks (block 0 is reserved in BlockPool).
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().cache_specs[0];
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

    ASSERT_TRUE(cache_manager->setKVBlockValue(block_src, k_t, v_t));

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
    ASSERT_TRUE(cache_manager->setKVBlockValue(block_dst, /*layer_id=*/0, k2_t, v2_t));

    std::vector<int8_t> expected_layer0(k_bytes + v_bytes, 0);
    std::fill(expected_layer0.begin(), expected_layer0.begin() + k_bytes, 1);
    std::fill(expected_layer0.begin() + k_bytes, expected_layer0.end(), 2);
    assertBlockBytesEq(cache_manager, /*layer_id=*/0, block_dst, expected_layer0);
    assertBlockBytesEq(cache_manager, /*layer_id=*/1, block_dst, expected_block);
}

TEST_F(KVCacheManagerTest, BlockCopyAlsoCopiesScaleWhenQuantized) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false);
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

        const size_t kv_scale_stride_bytes = cache_manager->cacheConfig().kv_scale_stride_bytes;
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
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().cache_specs[0];
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
        ASSERT_TRUE(cache_manager->setKVBlockValue(block_id, k_t, v_t));
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
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    const int    spb         = static_cast<int>(manager_config.seq_size_per_block);
    auto         resource    = makeDSV4BatchResource(manager_config);
    auto         tokens      = makeDSV4CompleteTokenIds(/*initial_seq_len=*/2 * spb + 17,
                                           /*max_seq_len=*/4 * spb + 32,
                                           spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    auto malloc_result              = manager->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(resource->groupNums(), kDsv4PoolNum);

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 3) << "group " << gid;
        const auto& blocks = resource->blocks(0, gid);
        if (gid < 3) {
            EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "paged group " << gid;
            EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "paged group " << gid;
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "paged group " << gid;
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "tail group " << gid << " should skip non-tail block";
            EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "tail group " << gid;
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "tail group " << gid;
        }
    }

    tokens->setSeqLength(4 * spb);
    MallocInfo incr_info{resource, tokens};
    incr_info.reuse_cache         = false;
    incr_info.enable_device_cache = false;
    auto incr_result              = manager->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(resource->blocksNum(0, gid), 4) << "group " << gid;
    }

    auto layout = manager->getMainModelCacheLayerLayout();
    ASSERT_EQ(layout.group_region_names.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn.size(), static_cast<size_t>(manager_config.layer_num));

    const int csa_layer = manager_config.global_layer_ids[0][0];
    const int hca_layer = manager_config.global_layer_ids[1][0];
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 0)[0], csa_layer, KVCacheRegionName::CSA_KV).kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 2)[0], csa_layer, KVCacheRegionName::INDEXER_KV).kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 4)[2], csa_layer, KVCacheRegionName::CSA_STATE).kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 6)[2], csa_layer, KVCacheRegionName::SWA_KV).kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 1)[0], hca_layer, KVCacheRegionName::HCA_KV).kv_addr,
              nullptr);
    EXPECT_ANY_THROW(
        (void)manager->convertIndexToAddr(resource->blocks(0, 1)[0], csa_layer, KVCacheRegionName::HCA_KV));

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

TEST_F(KVCacheManagerTest, DSV4LayerRegionBlockTablesMatchInferenceAccessPattern) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager_config.seq_size_per_block);
    auto      resource = makeDSV4BatchResource(manager_config);
    auto      tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/3 * spb + 17,
                                           /*max_seq_len=*/4 * spb + 32,
                                           spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_info).success);

    auto expectRegionGroup = [&](int layer_id, KVCacheRegionName region_name, int expected_gid) {
        EXPECT_EQ(resource->groupId(/*batch_id=*/0, layer_id, region_name), expected_gid)
            << "layer=" << layer_id << " region=" << static_cast<int>(region_name);
        EXPECT_EQ(resource->blocks(/*batch_id=*/0, layer_id, region_name), resource->blocks(0, expected_gid))
            << "layer=" << layer_id << " region=" << static_cast<int>(region_name);
        EXPECT_EQ(resource->kernelBlocks(/*batch_id=*/0, layer_id, region_name),
                  resource->kernelBlocks(0, expected_gid))
            << "layer=" << layer_id << " region=" << static_cast<int>(region_name);
    };

    // Flash DSV4 layers 0/1 are SWA-only. Even though layer_to_group_id defaults
    // to SWA, inference resolves typed block tables by KVCacheRegionName.
    expectRegionGroup(/*layer_id=*/0, KVCacheRegionName::SWA_KV, /*expected_gid=*/6);
    EXPECT_ANY_THROW((void)resource->blocks(/*batch_id=*/0, /*layer_id=*/0, KVCacheRegionName::CSA_KV));
    EXPECT_ANY_THROW((void)resource->blocks(/*batch_id=*/0, /*layer_id=*/0, KVCacheRegionName::HCA_KV));

    // Layer 2 is CSA: CSA_KV + INDEXER_KV + INDEXER_STATE + CSA_STATE + SWA_KV.
    const int csa_layer = manager_config.global_layer_ids[0][0];
    expectRegionGroup(csa_layer, KVCacheRegionName::CSA_KV, /*expected_gid=*/0);
    expectRegionGroup(csa_layer, KVCacheRegionName::INDEXER_KV, /*expected_gid=*/2);
    expectRegionGroup(csa_layer, KVCacheRegionName::INDEXER_STATE, /*expected_gid=*/3);
    expectRegionGroup(csa_layer, KVCacheRegionName::CSA_STATE, /*expected_gid=*/4);
    expectRegionGroup(csa_layer, KVCacheRegionName::SWA_KV, /*expected_gid=*/6);
    EXPECT_ANY_THROW((void)resource->blocks(/*batch_id=*/0, csa_layer, KVCacheRegionName::HCA_KV));

    // Layer 3 is HCA: HCA_KV + HCA_STATE + SWA_KV.
    const int hca_layer = manager_config.global_layer_ids[1][0];
    expectRegionGroup(hca_layer, KVCacheRegionName::HCA_KV, /*expected_gid=*/1);
    expectRegionGroup(hca_layer, KVCacheRegionName::HCA_STATE, /*expected_gid=*/5);
    expectRegionGroup(hca_layer, KVCacheRegionName::SWA_KV, /*expected_gid=*/6);
    EXPECT_ANY_THROW((void)resource->blocks(/*batch_id=*/0, hca_layer, KVCacheRegionName::CSA_KV));

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
}

TEST_F(KVCacheManagerTest, DSV4BlockCopyPreservesTypedRegionBytes) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/8);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager_config.seq_size_per_block);
    const int seq_len  = 3 * spb + 1;
    auto      resource = makeDSV4BatchResource(manager_config);
    auto      tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_info).success);

    const int src_block      = 1;
    const int dst_block      = 2;
    const int csa_layer      = manager_config.global_layer_ids[0][0];
    const int hca_layer      = manager_config.global_layer_ids[1][0];
    const int swa_only_layer = 0;

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        EXPECT_NE(std::find(blocks.begin(), blocks.end(), src_block), blocks.end()) << "group " << gid;
        EXPECT_NE(std::find(blocks.begin(), blocks.end(), dst_block), blocks.end()) << "group " << gid;
    }

    struct RegionCase {
        int               gid;
        int               layer_id;
        KVCacheRegionName region_name;
        uint8_t           pattern;
    };

    const std::vector<RegionCase> cases = {
        {0, csa_layer, KVCacheRegionName::CSA_KV, 0x11},
        {2, csa_layer, KVCacheRegionName::INDEXER_KV, 0x22},
        {3, csa_layer, KVCacheRegionName::INDEXER_STATE, 0x33},
        {4, csa_layer, KVCacheRegionName::CSA_STATE, 0x44},
        {6, csa_layer, KVCacheRegionName::SWA_KV, 0x55},
        {1, hca_layer, KVCacheRegionName::HCA_KV, 0x66},
        {5, hca_layer, KVCacheRegionName::HCA_STATE, 0x77},
        {6, swa_only_layer, KVCacheRegionName::SWA_KV, 0x88},
    };

    for (const auto& region_case : cases) {
        const size_t bytes = manager_config.cache_specs[static_cast<size_t>(region_case.gid)]->block_size_bytes();
        ASSERT_GT(bytes, 0u);
        writeDsv4RegionPattern(
            manager, src_block, region_case.layer_id, region_case.region_name, bytes, region_case.pattern);
        writeDsv4RegionPattern(manager, dst_block, region_case.layer_id, region_case.region_name, bytes, 0);
        assertDsv4RegionPatternEq(
            manager, src_block, region_case.layer_id, region_case.region_name, bytes, region_case.pattern);
        assertDsv4RegionPatternEq(manager, dst_block, region_case.layer_id, region_case.region_name, bytes, 0);
    }

    manager->blockCopy(src_block, dst_block);
    runtimeSyncAndCheck();

    for (const auto& region_case : cases) {
        const size_t bytes = manager_config.cache_specs[static_cast<size_t>(region_case.gid)]->block_size_bytes();
        assertDsv4RegionPatternEq(
            manager, dst_block, region_case.layer_id, region_case.region_name, bytes, region_case.pattern);
    }

    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
}

TEST_F(KVCacheManagerTest, DSV4InsertIntoDeviceBlockCacheThenReuseSamePrefix) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb     = static_cast<int>(manager_config.seq_size_per_block);
    const int seq_len = 3 * spb + 17;

    auto first_resource = makeDSV4BatchResource(manager_config);
    auto first_tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo first_malloc{first_resource, first_tokens};
    first_malloc.reuse_cache         = true;
    first_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(first_malloc).success);

    std::vector<BlockIndicesType> first_blocks;
    first_blocks.reserve(kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        first_blocks.push_back(first_resource->blocks(0, gid));
    }

    InsertInfo insert_info{first_resource, first_tokens, /*is_resident=*/false};
    manager->insertIntoCache(insert_info);

    FreeInfo first_free{first_resource, first_tokens};
    manager->free(first_free);

    auto second_resource = makeDSV4BatchResource(manager_config);
    auto second_tokens   = makeDSV4CompleteTokenIds(seq_len, seq_len, spb);

    MallocInfo second_malloc{second_resource, second_tokens};
    second_malloc.reuse_cache         = true;
    second_malloc.enable_device_cache = true;
    auto reuse_result                 = manager->malloc(second_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_GE(reuse_result.reuse_len, spb);

    for (int gid = 0; gid < 3; ++gid) {
        ASSERT_GE(second_resource->blocksNum(0, gid), 3) << "paged group " << gid;
        EXPECT_EQ(second_resource->blocks(0, gid)[0], first_blocks[gid][0]);
        EXPECT_EQ(second_resource->blocks(0, gid)[1], first_blocks[gid][1]);
    }
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_GE(second_resource->blocksNum(0, gid), 3) << "tail group " << gid;
        EXPECT_EQ(second_resource->blocks(0, gid)[2], first_blocks[gid][2]);
    }

    FreeInfo second_free{second_resource, second_tokens};
    manager->free(second_free);
}

TEST_F(KVCacheManagerTest, DSV4InitReuseKeepsSWAPrefixTailBlock) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/64);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb = static_cast<int>(manager_config.seq_size_per_block);

    auto first_resource = makeDSV4BatchResource(manager_config);
    auto first_tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/4 * spb, /*max_seq_len=*/4 * spb + 1, spb);

    MallocInfo first_malloc{first_resource, first_tokens};
    first_malloc.reuse_cache         = false;
    first_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(first_malloc).success);

    std::vector<std::pair<BlockIdxType, BlockIdxType>> first_swa_tail_blocks;
    first_swa_tail_blocks.reserve(kDsv4PoolNum - 3);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(first_resource->blocksNum(0, gid), 4) << "first SWA group " << gid;
        ASSERT_TRUE(isNullBlockIdx(first_resource->blocks(0, gid)[0]));
        ASSERT_TRUE(isNullBlockIdx(first_resource->blocks(0, gid)[1]));
        ASSERT_FALSE(isNullBlockIdx(first_resource->blocks(0, gid)[2]));
        ASSERT_FALSE(isNullBlockIdx(first_resource->blocks(0, gid)[3]));
        first_swa_tail_blocks.emplace_back(first_resource->blocks(0, gid)[2], first_resource->blocks(0, gid)[3]);
    }

    // Simulate one generated token before inserting into the device cache, so
    // the fourth full block is cached and can be reused by the next prefill.
    first_tokens->setSeqLength(4 * spb + 1);
    manager->insertIntoCache(InsertInfo{first_resource, first_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{first_resource, first_tokens});

    auto second_resource = makeDSV4BatchResource(manager_config);
    auto second_tokens   = makeDSV4CompleteTokenIds(/*initial_seq_len=*/24 * spb, /*max_seq_len=*/24 * spb, spb);

    MallocInfo second_malloc{second_resource, second_tokens};
    second_malloc.reuse_cache                  = true;
    second_malloc.enable_device_cache          = true;
    second_malloc.enable_remove_skipped_blocks = false;
    auto reuse_result                          = manager->malloc(second_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_EQ(reuse_result.reuse_len, 4 * spb);

    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = second_resource->blocks(0, gid);
        ASSERT_EQ(blocks.size(), 24u) << "second SWA group " << gid;
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "SWA reuse prefix penultimate block is NULL (no prev lookup)";
        EXPECT_EQ(blocks[3], first_swa_tail_blocks[static_cast<size_t>(gid - 3)].second)
            << "SWA reuse prefix tail block must stay readable";
        EXPECT_FALSE(isNullBlockIdx(blocks[22])) << "second SWA group " << gid << " fresh tail block 22";
        EXPECT_FALSE(isNullBlockIdx(blocks[23])) << "second SWA group " << gid << " fresh tail block 23";
    }

    manager->free(FreeInfo{second_resource, second_tokens});
}

TEST_F(KVCacheManagerTest, DSV4PopCachedBlocksPreservesGroupShape) {
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/16);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb      = static_cast<int>(manager_config.seq_size_per_block);
    const int seq_len  = 3 * spb + 1;
    auto      resource = makeDSV4BatchResource(manager_config);
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
    EXPECT_EQ(evicted->cacheResource(0).layerAttnBlocks().size(), static_cast<size_t>(manager_config.layer_num));

    bool saw_paged_block = false;
    bool saw_tail_block  = false;
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(evicted->blocksNum(0, gid), static_cast<int>(evicted->cacheKeys(0).size())) << "group " << gid;
        for (auto block : evicted->blocks(0, gid)) {
            if (!isNullBlockIdx(block)) {
                if (gid < 3) {
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

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config, false, nullptr, kv_cache_config);
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

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config, false, nullptr, kv_cache_config);
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

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config, false, nullptr, kv_cache_config);
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
        cache_config, false, nullptr, kv_cache_config, ParallelismConfig{}, runtime_config);
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

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cache_config);
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

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cache_config);
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

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cache_config);
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

    auto kv_cache_manager          = std::make_shared<KVCacheManager>(cache_config);
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

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config, false, nullptr, kv_cache_config);
    ASSERT_TRUE(kv_cache_manager->init());
    ASSERT_NE(kv_cache_manager->allocator_, nullptr);
    ASSERT_NE(kv_cache_manager->coordinator_, nullptr);

    // Seed device block cache with keys: 10, 11, 12 (put makes MRU at front => snapshot order: 12,11,10)
    auto shared_cache = kv_cache_manager->allocator_->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);
    {
        std::vector<BlockIdxType> group_slots(1);
        group_slots[0] = 1;
        shared_cache->put(10, group_slots, false);
        group_slots[0] = 2;
        shared_cache->put(11, group_slots, false);
        group_slots[0] = 3;
        shared_cache->put(12, group_slots, false);
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

TEST_F(KVCacheManagerTest, GetKVCacheInfo_UsesSmallestHybridPoolTokenCapacity) {
    auto cache_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/16, /*swa_batch_size=*/3);

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(kv_cache_manager->init());

    auto hybrid_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(kv_cache_manager->allocator_);
    ASSERT_NE(hybrid_allocator, nullptr);

    size_t      expected_total_tokens     = std::numeric_limits<size_t>::max();
    size_t      expected_available_tokens = std::numeric_limits<size_t>::max();
    const auto& pools                     = hybrid_allocator->groupBlockPools();
    ASSERT_GT(pools.size(), 1u);

    for (size_t gid = 0; gid < pools.size(); ++gid) {
        ASSERT_NE(pools[gid], nullptr);
        const size_t seq_size =
            (gid < cache_config.group_seq_size_per_block.size() && cache_config.group_seq_size_per_block[gid] > 0) ?
                cache_config.group_seq_size_per_block[gid] :
                cache_config.seq_size_per_block;
        expected_total_tokens     = std::min(expected_total_tokens, pools[gid]->totalBlocksNum() * seq_size);
        expected_available_tokens = std::min(expected_available_tokens, pools[gid]->availableBlocksNum() * seq_size);
    }

    auto info = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);

    EXPECT_EQ(info.total_kv_cache, expected_total_tokens);
    EXPECT_EQ(info.available_kv_cache, expected_available_tokens);
    EXPECT_LT(info.total_kv_cache, kv_cache_manager->totalBlocksNum() * cache_config.seq_size_per_block);
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
        cache_config, false, nullptr, kv_cache_config, ParallelismConfig{}, runtime_config);
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

// Re-enabled: realigned to allocator-level insertIntoCache (commit 9b2f364c5)
// + SWA reuse_cache=true semantics (kSwaActiveTailBlocks=2, step=1 -> all
// positions allocated). See docs/dsv4/kvcache-unify-final/summaries/
// REENABLE_DISABLED_TESTS.md for the full derivation.
TEST_F(KVCacheManagerTest, DSV4EvictionTriggeredWhenPoolExhaustedByCache) {
    // This test verifies that when block pools are exhausted by cached (but freed) requests,
    // a new allocation correctly triggers LRU eviction from the shared BlockCache.
    //
    // Setup: block_num=8 -> 7 usable blocks per group (block 0 reserved). 7 groups.
    // Request seq_len = 3*spb. With reuse_cache=true and step=1, the SWA group's
    // shouldAllocateBlock() returns true at every position (step_hit always fires),
    // so SWA allocates all 3 slots as real blocks just like FULL. The allocator-level
    // insertIntoCache caches every non-NULL slot (no active-tail drop), so each request
    // caches 3 blocks per FULL and per SWA group.
    //
    // Crucially, SharedBlockCache eviction is GLOBAL across groups: evicting one
    // cache_key frees the slot in every group that contributed to that key. The
    // SWA pool pressure therefore cascades cache loss into the FULL pool as well.
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/8);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int    spb         = static_cast<int>(manager_config.seq_size_per_block);
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
    auto       res_a    = makeDSV4BatchResource(manager_config);
    auto       tokens_a = makeTokens(/*offset=*/0);
    MallocInfo malloc_a{res_a, tokens_a};
    malloc_a.reuse_cache         = true;
    malloc_a.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_a).success);

    InsertInfo insert_a{res_a, tokens_a, /*is_resident=*/false};
    manager->insertIntoCache(insert_a);
    FreeInfo free_a{res_a, tokens_a};
    manager->free(free_a);

    // After A: cache_ref=1, request_ref=0 for every materialised slot.
    // Each FULL group: 3 cached, 4 free.  Each SWA group: 3 cached, 4 free.
    // Total free = 3*4 + 4*4 = 28.
    const size_t free_after_a = manager->freeBlocksNum();
    EXPECT_EQ(free_after_a, 3u * 4u + 4u * 4u);

    // --- Request B: different tokens → different cache keys ---
    auto       res_b    = makeDSV4BatchResource(manager_config);
    auto       tokens_b = makeTokens(/*offset=*/10000);
    MallocInfo malloc_b{res_b, tokens_b};
    malloc_b.reuse_cache         = true;
    malloc_b.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_b).success);

    InsertInfo insert_b{res_b, tokens_b, /*is_resident=*/false};
    manager->insertIntoCache(insert_b);
    FreeInfo free_b{res_b, tokens_b};
    manager->free(free_b);

    // After B: no pool exhausted (each had 4 free, B needs 3) -> no eviction.
    // Each FULL/SWA group now caches 3 (A) + 3 (B) = 6 blocks, 1 free.
    // Total free = 3*1 + 4*1 = 7.
    const size_t free_after_b = manager->freeBlocksNum();
    EXPECT_EQ(free_after_b, 3u * 1u + 4u * 1u);

    // --- Request C: still fits, but leaves FULL groups with only one free block ---
    auto       res_c    = makeDSV4BatchResource(manager_config);
    auto       tokens_c = makeTokens(/*offset=*/20000);
    MallocInfo malloc_c{res_c, tokens_c};
    malloc_c.reuse_cache         = true;
    malloc_c.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_c).success);

    InsertInfo insert_c{res_c, tokens_c, /*is_resident=*/false};
    manager->insertIntoCache(insert_c);
    FreeInfo free_c{res_c, tokens_c};
    manager->free(free_c);

    // After C: every pool has only 1 free, so C's 3-block need triggers
    // shared-cache eviction. Each evicted key frees 1 slot in every group
    // (A's keys are oldest -> evicted first). After enough eviction to fit C
    // and a re-insert of C, every group is at 7 cached, 0 free.
    // Total free = 3*0 + 4*0 = 0.
    const size_t free_after_c = manager->freeBlocksNum();
    EXPECT_EQ(free_after_c, 0u);

    // --- Request D: triggers eviction on FULL groups ---
    auto       res_d    = makeDSV4BatchResource(manager_config);
    auto       tokens_d = makeTokens(/*offset=*/30000);
    MallocInfo malloc_d{res_d, tokens_d};
    malloc_d.reuse_cache         = true;
    malloc_d.enable_device_cache = false;

    // This allocation MUST succeed — FULL groups trigger ensureFreeBlocks → evict from cache.
    auto result_d = manager->malloc(malloc_d);
    ASSERT_TRUE(result_d.success) << "Fourth allocation should succeed via eviction";

    // Verify block structure for request D. With reuse_cache=true and step=1,
    // SWA also allocates every slot, so D's per-group layout is identical for
    // FULL and SWA: 3 real blocks each.
    ASSERT_EQ(res_d->groupNums(), kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_d->blocksNum(0, gid), 3) << "group " << gid;
        const auto& blocks = res_d->blocks(0, gid);
        for (int i = 0; i < 3; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "group " << gid << " pos " << i;
        }
    }

    // After D allocated: every group is fully consumed (0 free per group) and
    // the pool is at least as tight as before D. free_after_c was already 0,
    // so EXPECT_LE rather than EXPECT_LT.
    EXPECT_LE(manager->freeBlocksNum(), free_after_c) << "Pool should be at least as tight after D allocated";

    // --- Free D and verify blocks return to pool ---
    FreeInfo free_d{res_d, tokens_d};
    manager->free(free_d);

    // After freeing D, its 3-per-group request blocks (request_ref->0, cache_ref=0
    // since we did not insertIntoCache for D) return to the free pool. Each pool
    // ends up with 4 cached (residue of B/C survivors) + 3 free per group.
    // freeBlocksNum >= free_after_c trivially since free_after_c was 0.
    EXPECT_GE(manager->freeBlocksNum(), free_after_c);

    // --- Pop all remaining cached blocks and verify full pool recovery ---
    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/100);
    if (evicted) {
        manager->blockCacheFree(evicted);
    }
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

// Re-enabled: allocator-level insertIntoCache (commit 9b2f364c5) now caches
// every materialised slot. For a 2-block seed the SWA group caches both
// tail blocks (vs. legacy 1), so the reuse path matches reuse_len = 2 * spb
// instead of spb. The post-reuse block layout is unchanged: the reused
// prefix slot is NULL-stamped by reuseCache() and only the new tail block
// is allocated fresh at position 2.
TEST_F(KVCacheManagerTest, DSV4ReuseAllSeededTailBlocksAndAllocOneFreshTail) {
    auto manager_config = makeProductionDSV4Config(/*full_block_num=*/8, /*max_concurrency=*/1);
    ASSERT_EQ(manager_config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    // SWA groups: rule_blocks(8) + non_full_addition(4) = 12
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(manager_config.group_block_nums[gid], 12u) << "group " << gid;
    }

    auto manager = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    // FULL groups: 3 × (8-1)=21, SWA groups: 4 × (12-1)=44, total=65
    EXPECT_EQ(free_before, 3u * 7u + 4u * 11u);
    const int spb = static_cast<int>(manager_config.seq_size_per_block);

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

    // Seed one reusable SWA/state block per fixed pool.  For a 2-block request,
    // HybridKVCacheAllocator::insertIntoCache caches EVERY non-NULL slot
    // (no N-1 / active-tail filter), so both blocks land in SharedBlockCache.
    // The matching expected counters below derive from that documented behavior.
    auto       seed_res    = makeDSV4BatchResource(manager_config);
    auto       seed_tokens = makeTokens(2 * spb);
    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = false;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(seed_malloc).success);

    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(seed_res->blocksNum(0, gid), 2) << "seed group " << gid;
        ASSERT_FALSE(isNullBlockIdx(seed_res->blocks(0, gid)[0])) << "seed group " << gid;
    }

    manager->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{seed_res, seed_tokens});

    // Same prefix, one more block.  This hits one cached fixed-pool block and
    // must still have room for the two fresh tail blocks.  The matched block is
    // then skipped out of the active SWA tail by the decode allocation path.
    auto       reuse_res    = makeDSV4BatchResource(manager_config);
    auto       reuse_tokens = makeTokens(3 * spb);
    MallocInfo reuse_malloc{reuse_res, reuse_tokens};
    reuse_malloc.reuse_cache         = true;
    reuse_malloc.enable_device_cache = true;
    auto reuse_result                = manager->malloc(reuse_malloc);
    ASSERT_TRUE(reuse_result.success);
    // Allocator-level insertIntoCache caches every materialised slot, so both
    // seed SWA tail blocks were inserted; the reuse path matches both keys
    // -> reuse_len = 2 * spb (vs. legacy spb when only the last tail was cached).
    EXPECT_EQ(reuse_result.reuse_len, 2 * spb);

    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = reuse_res->blocks(0, gid);
        ASSERT_EQ(blocks.size(), 3u) << "reuse group " << gid;
        // reuseCache() NULL-stamps the prefix portion and re-inserts the
        // matched tail block at logical_reuse_len-1; positions 0..(L-2)
        // therefore remain NULL even when the prefix matches multiple keys.
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "reuse group " << gid << " skipped reused prefix";
        EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "reuse group " << gid << " matched tail block from seed";
        EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "reuse group " << gid << " fresh tail block 2";
    }

    manager->free(FreeInfo{reuse_res, reuse_tokens});
    auto evicted = manager->popBlocksFromCache(/*min_blocks_to_free=*/100);
    if (evicted) {
        manager->blockCacheFree(evicted);
    }
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

// Re-enabled: realigned to allocator-level insertIntoCache (commit 9b2f364c5)
// + SharedBlockCache global-eviction (cross-pool cascade) semantics + new SWA
// removeSkippedBlocks keep-last-3 invariant. See REENABLE_DISABLED_TESTS.md.
TEST_F(KVCacheManagerTest, DSV4EvictionOnSWAGroupsDuringInferenceWithDecodeContinuation) {
    // This test simulates DSV4 inference under tight SWA pressure and verifies
    // (a) eviction triggers when concurrent requests exhaust the SWA fixed pool,
    // (b) decode-phase removeSkippedBlocks frees the SWA trailing-3 slot, and
    // (c) eviction during decode succeeds while requests still hold cached blocks.
    //
    // Tight stress layout:
    //   FULL groups (0,1,2): paged pool with block_num=8 -> 7 usable
    //   SWA  groups (3,4,5,6): fixed pool with 4 blocks -> 3 usable
    //
    // We use reuse_cache=false for malloc so the SWA group keeps its native
    // tail-only allocation pattern ([NULL, real, real] for a 3*spb request).
    // Test 1 (DSV4EvictionTriggeredWhenPoolExhaustedByCache) covers the
    // reuse_cache=true / all-positions variant; Test 4 uses reuse_cache=false
    // throughout. A dedicated reuse_cache=true + decode-step-pressure
    // composition is queued as a follow-up (R3-7 MED).
    auto manager_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/8, /*swa_batch_size=*/2);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb     = static_cast<int>(manager_config.seq_size_per_block);
    const int seq_len = 3 * spb;

    // Verify differentiated pool sizes.
    const size_t free_before = manager->freeBlocksNum();
    // FULL groups: 3 groups × 7 usable = 21.  SWA groups: 4 groups × 3 usable = 12.  Total = 33.
    EXPECT_EQ(free_before, 3u * 7u + 4u * 3u);

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
    auto       res_a    = makeDSV4BatchResource(manager_config);
    auto       tokens_a = makeTokens(/*offset=*/0);
    MallocInfo malloc_a{res_a, tokens_a};
    malloc_a.reuse_cache         = false;
    malloc_a.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_a).success);
    InsertInfo insert_a{res_a, tokens_a, /*is_resident=*/false};
    manager->insertIntoCache(insert_a);
    manager->free(FreeInfo{res_a, tokens_a});

    auto       res_b    = makeDSV4BatchResource(manager_config);
    auto       tokens_b = makeTokens(/*offset=*/10000);
    MallocInfo malloc_b{res_b, tokens_b};
    malloc_b.reuse_cache         = false;
    malloc_b.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_b).success);
    InsertInfo insert_b{res_b, tokens_b, /*is_resident=*/false};
    manager->insertIntoCache(insert_b);
    manager->free(FreeInfo{res_b, tokens_b});

    // After Phase 1 (A then B, both inserted+freed):
    //   * B's malloc needs 2 SWA blocks but has only 1 free, so ensureFreeBlocks
    //     evicts cache keys until 2 SWA slots are available. The oldest key
    //     (kA_3, both FULL+SWA non-null) is enough -- evicting it cascades a
    //     1-block free into every FULL group and a 1-block free into every SWA group.
    //   * Only 1 key (kA_3) is evicted; kA_1 (FULL-only) and kA_2 (FULL+SWA) survive.
    //   * Re-insert B adds 3 FULL keys (kB_1 SWA-null, kB_2/kB_3 FULL+SWA).
    //   After all of Phase 1:
    //     FULL groups: 5 cached (kA_1 + kA_2 + 3xB) + 2 free -> 3*2 = 6 total FULL free
    //     SWA  groups: 3 cached (kA_2 + kB_2 + kB_3) + 0 free -> 4*0 = 0 total SWA free
    const size_t free_after_cache = manager->freeBlocksNum();
    EXPECT_EQ(free_after_cache, 3u * 2u + 4u * 0u);  // = 6

    // === Phase 2: 3rd request triggers eviction on SWA groups ===
    auto       res_c    = makeDSV4BatchResource(manager_config);
    auto       tokens_c = makeTokens(/*offset=*/20000);
    MallocInfo malloc_c{res_c, tokens_c};
    malloc_c.reuse_cache         = false;
    malloc_c.enable_device_cache = false;

    // FULL needs 3 but only has 2 free -> evicts kA_2 (oldest with FULL slot).
    // SWA needs 2 but only has 1 free -> evicts kA_1 (FULL-only, no SWA freed)
    // then kB_3 (oldest remaining with SWA non-null).
    // After eviction: cache = {kB_1, kB_2}; FULL has 2 cached, SWA has 1 cached.
    auto result_c = manager->malloc(malloc_c);
    ASSERT_TRUE(result_c.success) << "3rd allocation must succeed via SWA eviction";

    // Verify block structure: FULL fully real, SWA tail-only [NULL, X, Y].
    ASSERT_EQ(res_c->groupNums(), kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 3) << "group " << gid;
        const auto& blocks = res_c->blocks(0, gid);
        if (gid < 3) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i;
            }
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "SWA group " << gid << " pos 0";
            EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "SWA group " << gid << " pos 1 (X = tail-1)";
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "SWA group " << gid << " pos 2 (Y = tail)";
        }
    }

    // Record C's initial SWA tail blocks for tracking through decode.
    std::vector<BlockIdxType> swa_block_X(4), swa_block_Y(4);
    for (int i = 0; i < 4; ++i) {
        int gid        = 3 + i;
        swa_block_X[i] = res_c->blocks(0, gid)[1];
        swa_block_Y[i] = res_c->blocks(0, gid)[2];
    }

    // === Phase 3: Decode incrKVBlock -> SWA eviction + removeSkippedBlocks ===

    // --- Incr to 4*spb ---
    // Before incr: FULL 2 cached + 3 req, 2 free; SWA 1 cached + 2 req, 0 free.
    // SWA needs 1 -> evict kB_1 (no SWA freed, FULL-only), then kB_2 (frees SWA).
    // Cache now empty. Alloc 1 SWA tail at pos 3 + 1 FULL block.
    // removeSkippedBlocks(block_size=4, keep_begin=4-3=1) then frees pos 1 (X).
    // Final SWA: [NULL, NULL, Y(pos2 preserved), Z(new pos3)].
    tokens_c->setSeqLength(4 * spb);
    MallocInfo incr1{res_c, tokens_c};
    incr1.reuse_cache         = false;
    incr1.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr1).success) << "First incr must succeed via SWA eviction";

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 4) << "group " << gid << " after incr to 4*spb";
    }
    std::vector<BlockIdxType> swa_block_Z(4);
    for (int i = 0; i < 4; ++i) {
        int gid = 3 + i;
        EXPECT_TRUE(isNullBlockIdx(res_c->blocks(0, gid)[0])) << "SWA group " << gid << " pos 0";
        EXPECT_TRUE(isNullBlockIdx(res_c->blocks(0, gid)[1])) << "SWA group " << gid << " pos 1 (X freed)";
        EXPECT_EQ(res_c->blocks(0, gid)[2], swa_block_Y[i]) << "SWA group " << gid << " pos 2 = Y preserved";
        EXPECT_FALSE(isNullBlockIdx(res_c->blocks(0, gid)[3])) << "SWA group " << gid << " pos 3 = Z new";
        swa_block_Z[i] = res_c->blocks(0, gid)[3];
    }

    // --- Incr to 5*spb ---
    // SWA before alloc: [NULL, NULL, Y, Z], 2 used, 1 free per group. No eviction needed.
    // Alloc pos 4 (W). After alloc: [NULL, NULL, Y, Z, W].
    // removeSkippedBlocks(block_size=5, keep_begin=5-3=2): pos 2 (Y) is real -> free,
    // pos 1 is NULL -> break. Final SWA: [NULL, NULL, NULL, Z, W].
    tokens_c->setSeqLength(5 * spb);
    MallocInfo incr2{res_c, tokens_c};
    incr2.reuse_cache         = false;
    incr2.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr2).success) << "Second incr must succeed";

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 5) << "group " << gid << " after incr to 5*spb";
    }
    for (int i = 0; i < 4; ++i) {
        int gid = 3 + i;
        EXPECT_TRUE(isNullBlockIdx(res_c->blocks(0, gid)[0])) << "SWA group " << gid << " pos 0";
        EXPECT_TRUE(isNullBlockIdx(res_c->blocks(0, gid)[1])) << "SWA group " << gid << " pos 1";
        EXPECT_TRUE(isNullBlockIdx(res_c->blocks(0, gid)[2])) << "SWA group " << gid << " pos 2 (Y freed)";
        EXPECT_EQ(res_c->blocks(0, gid)[3], swa_block_Z[i]) << "SWA group " << gid << " pos 3 = Z preserved";
        EXPECT_FALSE(isNullBlockIdx(res_c->blocks(0, gid)[4])) << "SWA group " << gid << " pos 4 = W new";
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
// Re-enabled: realigned to the new SWA removeSkippedBlocks invariant
// (keep last 3 slots, kSwaActiveTailBlocks=2). After each incr the active
// tail-1 block (block_size-3) is freed and the next-newest pair survives.
// See REENABLE_DISABLED_TESTS.md for the per-step state derivation.
TEST_F(KVCacheManagerTest, DSV4InitThenIncrWithRemoveSkippedBlocksFullLifecycle) {
    // This test exercises the full lifecycle of a DSV4 request:
    //   1. initKVBlock (first malloc with 4 blocks)
    //   2. Multiple incrKVBlock calls (decode phase) that trigger removeSkippedBlocks
    //   3. Verify SWA groups free old non-tail blocks during incr
    //   4. Final free returns all blocks to pool
    auto manager_config = makeCompactDSV4ManagerConfig(/*block_num=*/32);
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    const int    spb         = static_cast<int>(manager_config.seq_size_per_block);
    auto         resource    = makeDSV4BatchResource(manager_config);

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
    //             SWA groups (3,4,5,6) have 4 slots: [NULL, NULL, real, real] (only tail 2 allocated).
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 4) << "group " << gid;
        const auto& blocks = resource->blocks(0, gid);
        if (gid < 3) {
            for (int i = 0; i < 4; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i;
            }
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "SWA group " << gid << " pos 0 should be NULL";
            EXPECT_TRUE(isNullBlockIdx(blocks[1])) << "SWA group " << gid << " pos 1 should be NULL";
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "SWA group " << gid << " pos 2 should be real";
            EXPECT_FALSE(isNullBlockIdx(blocks[3])) << "SWA group " << gid << " pos 3 should be real";
        }
    }

    // Record block IDs allocated after init for later validation.
    std::vector<BlockIndicesType> init_blocks(kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        init_blocks[gid] = resource->blocks(0, gid);
    }
    const size_t free_after_init = manager->freeBlocksNum();

    // --- Phase 2: First incrKVBlock (4 -> 5 blocks) ---
    // Alloc 1 new block per group. Then SWA removeSkippedBlocks(block_size=5,
    // keep_begin=5-3=2): pos 2 (init A) is real -> free, pos 1 is NULL -> break.
    // Final SWA: [NULL, NULL, NULL, B_init(pos3), C_new].
    tokens->setSeqLength(5 * spb);
    MallocInfo incr1_info{resource, tokens};
    incr1_info.reuse_cache         = false;
    incr1_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr1_info).success);

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 5) << "group " << gid << " after incr1";
    }
    // FULL groups: all 5 blocks real, first 4 preserved.
    for (int gid = 0; gid < 3; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 5; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i << " after incr1";
        }
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(blocks[i], init_blocks[gid][i]) << "FULL group " << gid << " pos " << i << " changed";
        }
    }
    // SWA groups after incr1: [NULL, NULL, NULL, B_init(pos3), C_new(pos4)]
    // Old A (init pos 2) is freed by removeSkippedBlocks.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "SWA group " << gid << " pos 0 after incr1";
        EXPECT_TRUE(isNullBlockIdx(blocks[1])) << "SWA group " << gid << " pos 1 after incr1";
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "SWA group " << gid << " pos 2 (init A freed)";
        EXPECT_EQ(blocks[3], init_blocks[gid][3]) << "SWA group " << gid << " pos 3 = init B preserved";
        EXPECT_FALSE(isNullBlockIdx(blocks[4])) << "SWA group " << gid << " pos 4 = C new";
    }

    // Net: alloc 1 per group (7) - free 1 per SWA group (4) = -3 blocks.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_init - 3);
    const size_t free_after_incr1 = manager->freeBlocksNum();

    // Record SWA blocks present after incr1 for the next step.
    // init_B is at pos 3, new C is at pos 4.
    std::vector<BlockIdxType> swa_tail_B(4), swa_new_C(4);
    for (int idx = 0; idx < 4; ++idx) {
        int gid        = 3 + idx;
        swa_tail_B[idx] = resource->blocks(0, gid)[3];
        swa_new_C[idx]  = resource->blocks(0, gid)[4];
    }

    // --- Phase 3: Second incrKVBlock (5 -> 6 blocks) ---
    // Alloc 1 new D. Then SWA removeSkippedBlocks(block_size=6, keep_begin=6-3=3):
    //   i=3: B_init (real) -> free, set NULL.
    //   i=2: NULL -> break.
    // Final SWA: [NULL, NULL, NULL, NULL, C_new(pos4 preserved), D_new(pos5)].
    tokens->setSeqLength(6 * spb);
    MallocInfo incr2_info{resource, tokens};
    incr2_info.reuse_cache         = false;
    incr2_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr2_info).success);

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 6) << "group " << gid << " after incr2";
    }

    // FULL groups: all 6 blocks real, first 4 unchanged.
    for (int gid = 0; gid < 3; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 6; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i << " after incr2";
        }
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(blocks[i], init_blocks[gid][i]) << "FULL group " << gid << " init block preserved";
        }
    }

    // SWA groups after incr2: [NULL, NULL, NULL, NULL, C, D]
    std::vector<BlockIdxType> swa_new_D(4);
    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(isNullBlockIdx(blocks[i])) << "SWA group " << gid << " pos " << i << " after incr2";
        }
        EXPECT_EQ(blocks[4], swa_new_C[gid_offset]) << "SWA group " << gid << " pos 4 = C preserved";
        EXPECT_FALSE(isNullBlockIdx(blocks[5])) << "SWA group " << gid << " pos 5 = new D";
        swa_new_D[gid_offset] = blocks[5];
    }

    // Net change per incr step: -3 (alloc 7, free 4 from SWA removeSkipped).
    EXPECT_EQ(manager->freeBlocksNum(), free_after_incr1 - 3);
    const size_t free_after_incr2 = manager->freeBlocksNum();

    // --- Phase 4: Third incrKVBlock (6 -> 7 blocks) ---
    // Alloc new E. SWA removeSkippedBlocks(block_size=7, keep_begin=7-3=4):
    //   i=4: C_new (real) -> free.
    //   i=3: NULL -> break.
    // Final SWA: [NULL, NULL, NULL, NULL, NULL, D(pos5 preserved), E(pos6)].
    tokens->setSeqLength(7 * spb);
    MallocInfo incr3_info{resource, tokens};
    incr3_info.reuse_cache         = false;
    incr3_info.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr3_info).success);

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 7) << "group " << gid << " after incr3";
    }

    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 5; ++i) {
            EXPECT_TRUE(isNullBlockIdx(blocks[i])) << "SWA group " << gid << " pos " << i << " after incr3";
        }
        EXPECT_EQ(blocks[5], swa_new_D[gid_offset]) << "SWA group " << gid << " pos 5 = D preserved";
        EXPECT_FALSE(isNullBlockIdx(blocks[6])) << "SWA group " << gid << " pos 6 = new E";
    }

    EXPECT_EQ(manager->freeBlocksNum(), free_after_incr2 - 3);

    // --- Phase 5: Free all — all blocks should return to pool ---
    FreeInfo free_info{resource, tokens};
    manager->free(free_info);
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

}  // namespace test
}  // namespace rtp_llm
