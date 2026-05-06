#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <algorithm>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"
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
    auto              config = DSV4ConfigCreator::createConfig(makeDSV4ManagerFlashModelConfig(), pc);
    config.block_num         = block_num;
    config.group_block_nums.assign(config.groupNums(), block_num);
    return config;
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
    ASSERT_EQ(resource->groupNums(), DSV4_NUM_POOLS);

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
        EXPECT_EQ(resource->blocksNum(0, gid), 4) << "group " << gid;
    }

    auto layout = manager->getMainModelCacheLayerLayout();
    ASSERT_EQ(layout.group_region_names.size(), static_cast<size_t>(DSV4_NUM_POOLS));
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

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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
    first_blocks.reserve(DSV4_NUM_POOLS);
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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
    for (int gid = 3; gid < DSV4_NUM_POOLS; ++gid) {
        ASSERT_GE(second_resource->blocksNum(0, gid), 3) << "tail group " << gid;
        EXPECT_EQ(second_resource->blocks(0, gid)[2], first_blocks[gid][2]);
    }

    FreeInfo second_free{second_resource, second_tokens};
    manager->free(second_free);
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
    EXPECT_EQ(evicted->groupNums(), DSV4_NUM_POOLS);
    EXPECT_EQ(evicted->cacheResource(0).layerAttnBlocks().size(), static_cast<size_t>(manager_config.layer_num));

    bool saw_paged_block = false;
    bool saw_tail_block  = false;
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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
    auto block_cache = kv_cache_manager->allocator_->getBlockPool()->blockCache();
    ASSERT_NE(block_cache, nullptr);
    {
        BlockCache::CacheItem item;
        item.group_id    = 0;
        item.is_resident = false;
        item.cache_key   = 10;
        item.block_index = 1;
        ASSERT_TRUE(block_cache->put(item));
        item.cache_key   = 11;
        item.block_index = 2;
        ASSERT_TRUE(block_cache->put(item));
        item.cache_key   = 12;
        item.block_index = 3;
        ASSERT_TRUE(block_cache->put(item));
    }

    // Inject a lightweight memory connector with a MemoryBlockCache snapshot:
    // put 11 then 13 => MRU order: 13,11 (11 duplicates device key)
    auto mem_connector = std::make_shared<KVCacheMemoryConnector>(
        cache_config, kv_cache_config, kv_cache_manager->allocator_, std::vector<std::string>{});
    mem_connector->block_cache_ = std::make_shared<MemoryBlockCache>();
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

    // latest_version=-1 forces BlockCache snapshot to return all current keys.
    auto info = kv_cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true);

    // Current implementation uses unordered_set -> assign, so order is not stable.
    // Only validate de-dup and set-equality.
    std::vector<CacheKeyType> got = info.cached_keys;
    std::sort(got.begin(), got.end());
    std::vector<CacheKeyType> expected = {10, 11, 12, 13};
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(got, expected);
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
    ASSERT_EQ(resource->groupNums(), DSV4_NUM_POOLS);

    // After init: FULL groups (0,1,2) have 4 real blocks each.
    //             SWA groups (3,4,5,6) have 4 slots: [NULL, NULL, real, real] (only tail 2 allocated).
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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
    std::vector<BlockIndicesType> init_blocks(DSV4_NUM_POOLS);
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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
    // SWA groups after incr1: [NULL, NULL, A, B, C] — position 2,3 are old tail, 4 is new.
    for (int gid = 3; gid < DSV4_NUM_POOLS; ++gid) {
        const auto& blocks = resource->blocks(0, gid);
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "SWA group " << gid << " pos 0 after incr1";
        EXPECT_TRUE(isNullBlockIdx(blocks[1])) << "SWA group " << gid << " pos 1 after incr1";
        // Old tail blocks preserved at positions 2,3.
        EXPECT_EQ(blocks[2], init_blocks[gid][2]) << "SWA group " << gid << " old tail pos 2";
        EXPECT_EQ(blocks[3], init_blocks[gid][3]) << "SWA group " << gid << " old tail pos 3";
        EXPECT_FALSE(isNullBlockIdx(blocks[4])) << "SWA group " << gid << " new block at pos 4";
    }

    // SWA groups did not free any blocks in this step (positions 0,1 were already NULL).
    // Each group allocated 1 new block → 7 new blocks consumed.
    EXPECT_EQ(manager->freeBlocksNum(), free_after_init - 7);
    const size_t free_after_incr1 = manager->freeBlocksNum();

    // Record SWA tail blocks after incr1 for the next step.
    std::vector<BlockIdxType> swa_tail_A(4), swa_tail_B(4), swa_new_C(4);
    for (int idx = 0; idx < 4; ++idx) {
        int gid         = 3 + idx;
        swa_tail_A[idx] = resource->blocks(0, gid)[2];
        swa_tail_B[idx] = resource->blocks(0, gid)[3];
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

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
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

    // SWA groups after incr2: [NULL, NULL, NULL, B, C, D]
    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        // Positions 0,1,2 should all be NULL (position 2 was freed by removeSkippedBlocks).
        EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "SWA group " << gid << " pos 0 after incr2";
        EXPECT_TRUE(isNullBlockIdx(blocks[1])) << "SWA group " << gid << " pos 1 after incr2";
        EXPECT_TRUE(isNullBlockIdx(blocks[2])) << "SWA group " << gid << " pos 2 should be freed by removeSkipped";
        // Position 3 = old B (tail-1), position 4 = old C (tail), position 5 = new D.
        EXPECT_EQ(blocks[3], swa_tail_B[gid_offset]) << "SWA group " << gid << " pos 3 = old B";
        EXPECT_EQ(blocks[4], swa_new_C[gid_offset]) << "SWA group " << gid << " pos 4 = old C";
        EXPECT_FALSE(isNullBlockIdx(blocks[5])) << "SWA group " << gid << " pos 5 = new D";
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

    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
        ASSERT_EQ(resource->blocksNum(0, gid), 7) << "group " << gid << " after incr3";
    }

    // SWA groups after incr3: [NULL, NULL, NULL, NULL, C, D, E]
    for (int gid_offset = 0; gid_offset < 4; ++gid_offset) {
        int         gid    = 3 + gid_offset;
        const auto& blocks = resource->blocks(0, gid);
        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(isNullBlockIdx(blocks[i])) << "SWA group " << gid << " pos " << i << " after incr3";
        }
        // Position 4 = old C, position 5 = old D (from incr2), position 6 = new E.
        EXPECT_EQ(blocks[4], swa_new_C[gid_offset]) << "SWA group " << gid << " pos 4 = C";
        EXPECT_FALSE(isNullBlockIdx(blocks[5])) << "SWA group " << gid << " pos 5";
        EXPECT_FALSE(isNullBlockIdx(blocks[6])) << "SWA group " << gid << " pos 6 = new E";
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
