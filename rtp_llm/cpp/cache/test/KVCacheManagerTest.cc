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
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {
constexpr int kDsv4PoolNum = 7;
const std::vector<std::string> kDsv4Tags = {
    "csa_kv", "hca_kv", "indexer_kv", "indexer_state", "csa_state", "hca_state", "swa_kv"};
}

class KVCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                 = StaticConfig::user_ft_core_dump_on_exception;
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
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

static void setGroupBlockNumsForTest(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    std::vector<size_t> block_sizes;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    block_sizes.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(gid));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(gid));
        block_sizes.push_back(config.blockSizeBytesForGroup(gid));
    }
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides, block_sizes);
}

static CacheConfig makeCompactDSV4ManagerConfig(uint32_t block_num = 16) {
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block     = 128;
    kv_cache_config.dsv4_hca_state_pool_blocks = 0;
    auto              config = HybridPoolConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc, kv_cache_config, false, 0);
    config.block_num         = block_num;
    setGroupBlockNumsForTest(config, std::vector<uint32_t>(static_cast<size_t>(config.groupNums()), block_num));
    return config;
}

static bool isHcaStateGroup(const CacheConfig& config, int gid) {
    return gid >= 0 && static_cast<size_t>(gid) < static_cast<size_t>(config.groupNums())
           && config.tagForGroup(static_cast<size_t>(gid)) == "hca_state";
}

static int dsv4ActiveTailBlocks(const CacheConfig& config, int gid) {
    return isHcaStateGroup(config, gid) ? 1 : 2;
}

static void expectDsv4SwaAllocatedBlocks(const CacheConfig& config,
                                         const BlockIndicesType& blocks,
                                         int gid,
                                         const std::string& label,
                                         bool enable_reuse_cache = false) {
    const int active_tail_blocks = dsv4ActiveTailBlocks(config, gid);
    const int tail_begin         = std::max(static_cast<int>(blocks.size()) - active_tail_blocks, 0);
    const int linear_step        = std::max(1, config.linear_step);
    const bool effective_reuse   = enable_reuse_cache && !isHcaStateGroup(config, gid);
    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
        const bool should_allocate = i >= tail_begin || (effective_reuse && ((i + 1) % linear_step == 0));
        if (should_allocate) {
            EXPECT_FALSE(isNullBlockIdx(blocks[static_cast<size_t>(i)]))
                << label << " group " << gid << " pos " << i;
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[static_cast<size_t>(i)]))
                << label << " group " << gid << " pos " << i;
        }
    }
}

// Creates an intentionally tight DSV4 config for eviction stress tests: FULL
// groups use a large paged pool, while SWA groups use a small independent pool.
static CacheConfig makeDSV4ConfigWithConcurrencyPool(uint32_t full_block_num, uint32_t swa_batch_size) {
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block     = 128;
    kv_cache_config.dsv4_hca_state_pool_blocks = 0;
    auto              config = HybridPoolConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc, kv_cache_config, false, 0);
    config.block_num         = full_block_num;
    std::vector<uint32_t> block_nums(static_cast<size_t>(config.groupNums()), full_block_num);
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        block_nums[static_cast<size_t>(gid)] = (gid < 3) ? full_block_num : (2u * swa_batch_size);
    }
    setGroupBlockNumsForTest(config, block_nums);
    return config;
}

static CacheConfig
makeProductionDSV4Config(uint32_t full_block_num, uint32_t max_concurrency, uint32_t hca_state_pool_blocks = 4) {
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block                         = 128;
    kv_cache_config.test_block_num                              = full_block_num;
    kv_cache_config.dsv4_hca_state_pool_blocks                  = hca_state_pool_blocks;
    runtime_config.max_generate_batch_size                      = max_concurrency;
    runtime_config.fifo_scheduler_config.max_context_batch_size = max_concurrency;
    return CacheConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc, runtime_config, kv_cache_config);
}

static BatchKVCacheResourcePtr makeDSV4BatchResource(const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(1);
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.primaryLayerGroupIdsSnapshot(),
                    config.kernelBlocksPerKvBlock(),
                    config.groupTypesSnapshot(),
                    config.layerGroupIdsSnapshot());
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
                                   int                                    group_id,
                                   size_t                                 bytes,
                                   uint8_t                                pattern) {
    auto addr_info = manager->convertIndexToAddr(block_id, layer_id, group_id);
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
                                      int                                    group_id,
                                      size_t                                 bytes,
                                      uint8_t                                expected) {
    auto addr_info = manager->convertIndexToAddr(block_id, layer_id, group_id);
    ASSERT_NE(addr_info.kv_addr, nullptr);

    auto dev_t =
        torch::from_blob(addr_info.kv_addr, {(int64_t)bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host_t = dev_t.cpu();
    const auto* ptr    = host_t.data_ptr<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        ASSERT_EQ(ptr[i], expected) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id
                                    << " group=" << group_id;
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

TEST_F(KVCacheManagerTest, DSV4IndependentPoolsUseGpuBacking) {
    auto expect_pool_backing = [](RoleType role_type) {
        auto config = makeCompactDSV4ManagerConfig(/*block_num=*/8);

        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        KVCacheConfig kv_cache_config;
        auto          cache_manager = std::make_shared<KVCacheManager>(config,
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
        ASSERT_EQ(allocator->groupBlockPools().size(), static_cast<size_t>(config.groupNums()));

        for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
            const auto& tag = config.tagForGroup(gid);
            EXPECT_EQ(allocator->groupBlockPools()[gid]->where(), MemoryType::MEMORY_GPU)
                << "role=" << static_cast<int>(role_type) << " gid=" << gid << " tag=" << tag;
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

    auto&        spec    = cache_manager->cacheConfig().specForGroup(0);
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

    auto&        spec    = cache_manager->cacheConfig().specForGroup(0);
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
            expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "tail group");
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
    ASSERT_EQ(layout.group_tags.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(layout.group_tags, kDsv4Tags);
    ASSERT_EQ(layout.group_seq_size_per_block, manager_config.group_seq_size_per_block);
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_group.size(), static_cast<size_t>(manager_config.layer_num));

    const int csa_layer = manager_config.layerIdsForGroup(0)[0];
    const int hca_layer = manager_config.layerIdsForGroup(1)[0];
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 0)[0], csa_layer, 0).kv_addr, nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 2)[0], csa_layer, 2).kv_addr,
              nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 4)[2], csa_layer, 4).kv_addr, nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 6)[2], csa_layer, 6).kv_addr, nullptr);
    EXPECT_NE(manager->convertIndexToAddr(resource->blocks(0, 1)[0], hca_layer, 1).kv_addr, nullptr);
    EXPECT_ANY_THROW((void)manager->convertIndexToAddr(resource->blocks(0, 1)[0], csa_layer, 1));

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

    auto expectTagGroup = [&](int layer_id, const std::string& tag, int expected_gid) {
        EXPECT_EQ(manager_config.groupIdForLayerTag(layer_id, tag), expected_gid)
            << "layer=" << layer_id << " tag=" << tag;
        EXPECT_EQ(resource->groupId(/*batch_id=*/0, layer_id, expected_gid), expected_gid)
            << "layer=" << layer_id << " tag=" << tag;
        EXPECT_EQ(resource->blocks(/*batch_id=*/0, layer_id, expected_gid), resource->blocks(0, expected_gid))
            << "layer=" << layer_id << " tag=" << tag;
        EXPECT_EQ(resource->kernelBlocks(/*batch_id=*/0, layer_id, expected_gid),
                  resource->kernelBlocks(0, expected_gid))
            << "layer=" << layer_id << " tag=" << tag;
    };

    // Flash DSV4 layers 0/1 are SWA-only. Inference resolves typed block tables by semantic tag.
    expectTagGroup(/*layer_id=*/0, "swa_kv", /*expected_gid=*/6);
    EXPECT_THROW((void)manager_config.groupIdForLayerTag(/*layer_id=*/0, "csa_kv"), std::exception);
    EXPECT_THROW((void)manager_config.groupIdForLayerTag(/*layer_id=*/0, "hca_kv"), std::exception);

    // Layer 2 is CSA: CSA_KV + INDEXER_KV + INDEXER_STATE + CSA_STATE + SWA_KV.
    const int csa_layer = manager_config.layerIdsForGroup(0)[0];
    expectTagGroup(csa_layer, "csa_kv", /*expected_gid=*/0);
    expectTagGroup(csa_layer, "indexer_kv", /*expected_gid=*/2);
    expectTagGroup(csa_layer, "indexer_state", /*expected_gid=*/3);
    expectTagGroup(csa_layer, "csa_state", /*expected_gid=*/4);
    expectTagGroup(csa_layer, "swa_kv", /*expected_gid=*/6);
    EXPECT_THROW((void)manager_config.groupIdForLayerTag(csa_layer, "hca_kv"), std::exception);

    // Layer 3 is HCA: HCA_KV + HCA_STATE + SWA_KV.
    const int hca_layer = manager_config.layerIdsForGroup(1)[0];
    expectTagGroup(hca_layer, "hca_kv", /*expected_gid=*/1);
    expectTagGroup(hca_layer, "hca_state", /*expected_gid=*/5);
    expectTagGroup(hca_layer, "swa_kv", /*expected_gid=*/6);
    EXPECT_THROW((void)manager_config.groupIdForLayerTag(hca_layer, "csa_kv"), std::exception);

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
    const int csa_layer      = manager_config.layerIdsForGroup(0)[0];
    const int hca_layer      = manager_config.layerIdsForGroup(1)[0];
    const int swa_only_layer = 0;

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        EXPECT_NE(std::find(blocks.begin(), blocks.end(), src_block), blocks.end()) << "group " << gid;
        if (!isHcaStateGroup(manager_config, gid)) {
            EXPECT_NE(std::find(blocks.begin(), blocks.end(), dst_block), blocks.end()) << "group " << gid;
        }
    }

    struct RegionCase {
        int               gid;
        int               layer_id;
        uint8_t           pattern;
    };

    const std::vector<RegionCase> cases = {
        {0, csa_layer, 0x11},
        {2, csa_layer, 0x22},
        {3, csa_layer, 0x33},
        {4, csa_layer, 0x44},
        {6, csa_layer, 0x55},
        {1, hca_layer, 0x66},
        {5, hca_layer, 0x77},
        {6, swa_only_layer, 0x88},
    };

    for (const auto& region_case : cases) {
        const size_t bytes = manager_config.specForGroup(static_cast<size_t>(region_case.gid))->block_size_bytes();
        ASSERT_GT(bytes, 0u);
        writeDsv4RegionPattern(manager, src_block, region_case.layer_id, region_case.gid, bytes, region_case.pattern);
        writeDsv4RegionPattern(manager, dst_block, region_case.layer_id, region_case.gid, bytes, 0);
        assertDsv4RegionPatternEq(
            manager, src_block, region_case.layer_id, region_case.gid, bytes, region_case.pattern);
        assertDsv4RegionPatternEq(manager, dst_block, region_case.layer_id, region_case.gid, bytes, 0);
    }

    manager->blockCopy(src_block, dst_block);
    runtimeSyncAndCheck();

    for (const auto& region_case : cases) {
        const size_t bytes = manager_config.specForGroup(static_cast<size_t>(region_case.gid))->block_size_bytes();
        assertDsv4RegionPatternEq(manager, dst_block, region_case.layer_id, region_case.gid, bytes, region_case.pattern);
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
        if (manager_config.policyForGroup(static_cast<size_t>(gid)).reuse_policy == CacheReusePolicy::NON_REUSABLE) {
            continue;
        }
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

    std::vector<BlockIdxType> first_swa_tail_blocks(static_cast<size_t>(kDsv4PoolNum), NULL_BLOCK_IDX);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(first_resource->blocksNum(0, gid), 4) << "first SWA group " << gid;
        expectDsv4SwaAllocatedBlocks(manager_config, first_resource->blocks(0, gid), gid, "first SWA");
        first_swa_tail_blocks[static_cast<size_t>(gid)] = first_resource->blocks(0, gid)[3];
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
        if (manager_config.policyForGroup(static_cast<size_t>(gid)).reuse_policy == CacheReusePolicy::NON_REUSABLE) {
            continue;
        }
        const auto& blocks = second_resource->blocks(0, gid);
        ASSERT_EQ(blocks.size(), 24u) << "second SWA group " << gid;
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "SWA reuse prefix penultimate block is NULL (no prev lookup)";
        EXPECT_EQ(blocks[3], first_swa_tail_blocks[static_cast<size_t>(gid)])
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
    EXPECT_EQ(evicted->cacheResource(0).layerGroupBlocks().size(), static_cast<size_t>(manager_config.layer_num));

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

TEST_F(KVCacheManagerTest, MaxAvailableTokensNumUsesCPVirtualBlockSizeForHybridPoolFullGroups) {
    auto cache_config = makeDSV4ConfigWithConcurrencyPool(/*full_block_num=*/16, /*swa_batch_size=*/3);

    auto kv_cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(kv_cache_manager->init());

    auto hybrid_allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(kv_cache_manager->allocator_);
    ASSERT_NE(hybrid_allocator, nullptr);

    const size_t physical_capacity = hybrid_allocator->maxAvailableTokensNum();
    auto cp_slot_mapper =
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, static_cast<int>(cache_config.seq_size_per_block));
    kv_cache_manager->cp_slot_mapper_ = cp_slot_mapper;
    hybrid_allocator->setCPSlotMapper(cp_slot_mapper);

    size_t      expected_logical_capacity = std::numeric_limits<size_t>::max();
    const auto& pools                     = hybrid_allocator->groupBlockPools();
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        if (gid < static_cast<size_t>(cache_config.groupNums())
            && cache_config.typeForGroup(gid) != CacheGroupType::FULL) {
            continue;
        }
        expected_logical_capacity =
            std::min(expected_logical_capacity,
                     pools[gid]->totalBlocksNum() * static_cast<size_t>(cache_config.seq_size_per_block * 2));
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

    const size_t free_after_a = manager->freeBlocksNum();
    EXPECT_LT(free_after_a, free_before);

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

    const size_t free_after_b = manager->freeBlocksNum();
    EXPECT_LT(free_after_b, free_after_a);

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

    const size_t free_after_c = manager->freeBlocksNum();
    EXPECT_LE(free_after_c, free_after_b);

    // --- Request D: triggers eviction on FULL groups ---
    auto       res_d    = makeDSV4BatchResource(manager_config);
    auto       tokens_d = makeTokens(/*offset=*/30000);
    MallocInfo malloc_d{res_d, tokens_d};
    malloc_d.reuse_cache         = true;
    malloc_d.enable_device_cache = false;

    // This allocation MUST succeed — FULL groups trigger ensureFreeBlocks → evict from cache.
    auto result_d = manager->malloc(malloc_d);
    ASSERT_TRUE(result_d.success) << "Fourth allocation should succeed via eviction";

    // Verify block structure for request D.
    ASSERT_EQ(res_d->groupNums(), kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_d->blocksNum(0, gid), 3) << "group " << gid;
        const auto& blocks = res_d->blocks(0, gid);
        if (gid < 3) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "fixed group", /*enable_reuse_cache=*/true);
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
    ASSERT_EQ(manager_config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        const uint32_t expected = gid == 5 ? 12u : 8u;
        ASSERT_EQ(manager_config.blockNumForGroup(static_cast<size_t>(gid)), expected) << "group " << gid;
    }

    auto manager = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const size_t free_before = manager->freeBlocksNum();
    EXPECT_EQ(free_before, 6u * 7u + 11u);
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

    // Seed one reusable SWA/state block per independent pool. For a 2-block request,
    // insertIntoCache keeps only the first full block; the active tail is not cached.
    auto       seed_res    = makeDSV4BatchResource(manager_config);
    auto       seed_tokens = makeTokens(2 * spb);
    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = false;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(seed_malloc).success);

    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(seed_res->blocksNum(0, gid), 2) << "seed group " << gid;
        expectDsv4SwaAllocatedBlocks(manager_config, seed_res->blocks(0, gid), gid, "seed group");
    }

    manager->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false});
    manager->free(FreeInfo{seed_res, seed_tokens});

    // Same prefix, one more block. This hits one cached independent-pool block and
    // must still have room for the two fresh tail blocks.  The matched block is
    // then skipped out of the active SWA tail by the decode allocation path.
    auto       reuse_res    = makeDSV4BatchResource(manager_config);
    auto       reuse_tokens = makeTokens(3 * spb);
    MallocInfo reuse_malloc{reuse_res, reuse_tokens};
    reuse_malloc.reuse_cache         = true;
    reuse_malloc.enable_device_cache = true;
    auto reuse_result                = manager->malloc(reuse_malloc);
    ASSERT_TRUE(reuse_result.success);
    EXPECT_EQ(reuse_result.reuse_len, 2 * spb);

    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        if (manager_config.policyForGroup(static_cast<size_t>(gid)).reuse_policy == CacheReusePolicy::NON_REUSABLE) {
            continue;
        }
        const auto& blocks = reuse_res->blocks(0, gid);
        ASSERT_EQ(blocks.size(), 3u) << "reuse group " << gid;
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "reuse group " << gid << " skipped reused prefix";
        EXPECT_FALSE(isNullBlockIdx(blocks[1])) << "reuse group " << gid << " tail block 1";
        EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "reuse group " << gid << " tail block 2";
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
    auto manager        = std::make_shared<KVCacheManager>(manager_config, /*warmup=*/false);
    ASSERT_TRUE(manager->init());

    const int spb     = static_cast<int>(manager_config.seq_size_per_block);
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
    auto       res_a    = makeDSV4BatchResource(manager_config);
    auto       tokens_a = makeTokens(/*offset=*/0);
    MallocInfo malloc_a{res_a, tokens_a};
    malloc_a.reuse_cache         = true;
    malloc_a.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(malloc_a).success);
    InsertInfo insert_a{res_a, tokens_a, /*is_resident=*/false};
    manager->insertIntoCache(insert_a);
    manager->free(FreeInfo{res_a, tokens_a});

    auto       res_b    = makeDSV4BatchResource(manager_config);
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
    auto       res_c    = makeDSV4BatchResource(manager_config);
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
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 3) << "group " << gid;
        const auto& blocks = res_c->blocks(0, gid);
        if (gid < 3) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "SWA group", /*enable_reuse_cache=*/true);
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

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 4) << "group " << gid << " after incr to 4*spb";
    }
    // SWA/state fixed groups retain the current tail window.
    for (int i = 0; i < 4; ++i) {
        int gid = 3 + i;
        expectDsv4SwaAllocatedBlocks(manager_config, res_c->blocks(0, gid), gid, "SWA group");
    }

    // --- Incr to 5*spb ---
    // Non-HCA SWA removes blocks before the active two-block tail; HCA_STATE keeps a one-block tail.
    // SWA pools may need another eviction if no free block remains.
    tokens_c->setSeqLength(5 * spb);
    MallocInfo incr2{res_c, tokens_c};
    incr2.reuse_cache         = false;
    incr2.enable_device_cache = false;
    ASSERT_TRUE(manager->malloc(incr2).success) << "Second incr must succeed (removeSkipped frees block)";

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(res_c->blocksNum(0, gid), 5) << "group " << gid << " after incr to 5*spb";
    }
    // SWA/state fixed groups keep only the active tail window.
    for (int i = 0; i < 4; ++i) {
        int gid = 3 + i;
        expectDsv4SwaAllocatedBlocks(manager_config, res_c->blocks(0, gid), gid, "SWA group");
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
    //             SWA groups keep the active tail window; HCA_STATE keeps a one-block tail.
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 4) << "group " << gid;
        const auto& blocks = resource->blocks(0, gid);
        if (gid < 3) {
            for (int i = 0; i < 4; ++i) {
                EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i;
            }
        } else {
            expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "SWA group");
        }
    }

    // Record block IDs allocated after init for later validation.
    std::vector<BlockIndicesType> init_blocks(kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        init_blocks[gid] = resource->blocks(0, gid);
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

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 5) << "group " << gid << " after incr1";
    }
    // FULL groups: all 5 blocks should be real.
    for (int gid = 0; gid < 3; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 5; ++i) {
            EXPECT_FALSE(isNullBlockIdx(blocks[i])) << "FULL group " << gid << " pos " << i << " after incr1";
        }
        // Original init blocks should be preserved.
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(blocks[i], init_blocks[gid][i]) << "FULL group " << gid << " pos " << i << " changed";
        }
    }
    // SWA/state fixed groups keep the current tail window.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "SWA group after incr1");
        if (!isHcaStateGroup(manager_config, gid)) {
            EXPECT_EQ(blocks[3], init_blocks[gid][3]) << "SWA group " << gid << " old tail pos 3";
        }
    }

    // Four fixed groups freed one stale block and all seven groups allocated one new block.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_init - 7 + 4);
    const size_t free_after_incr1 = manager->freeBlocksNum();

    // Record SWA tail blocks after incr1 for the next step.
    std::vector<BlockIdxType> swa_new_C(4);
    for (int idx = 0; idx < 4; ++idx) {
        int gid         = 3 + idx;
        swa_new_C[idx]  = resource->blocks(0, gid)[4];
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

    // SWA/state fixed groups after incr2 keep their configured active tail window.
    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "SWA group after incr2");
        if (!isHcaStateGroup(manager_config, gid)) {
            EXPECT_EQ(blocks[4], swa_new_C[gid_offset]) << "SWA group " << gid << " pos 4 = old C";
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

    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 7) << "group " << gid << " after incr3";
    }

    // SWA/state fixed groups after incr3 keep their configured active tail window.
    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        expectDsv4SwaAllocatedBlocks(manager_config, blocks, gid, "SWA group after incr3");
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
