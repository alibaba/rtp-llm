#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

#include <memory>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

// Build the device pool config from the lightweight cache-config test helpers
// (cache_config_test_utils + DeviceBlockPoolConfigHelper) rather than BlockPoolTestHelper,
// which drags in the heavy //rtp_llm/cpp/cache monolith (and its CUDA-13 prebuilt
// remote_kv_cache_manager dependency that does not link in the CUDA-12 toolchain).
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

static_assert(std::is_same_v<decltype(DeviceBlockPoolConfigHelper::createConfig(std::declval<const CacheConfig&>())),
                             DeviceBlockPoolConfig>);

std::shared_ptr<DeviceBlockPoolConfig> makeConfig() {
    constexpr int        kLayerNum       = 4;
    constexpr int        kBlockNum       = 10;
    constexpr size_t     kTokensPerBlock = 1;
    rtp_llm::CacheConfig cache_config    = rtp_llm::test::makeSimpleMhaCacheConfig(kLayerNum,
                                                                                kBlockNum,
                                                                                kTokensPerBlock,
                                                                                rtp_llm::TYPE_FP16,
                                                                                /*local_head_num_kv=*/1,
                                                                                /*size_per_head=*/64);
    auto config = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfig(cache_config));
    config->pool_name               = "device";
    config->use_cuda_malloc_backing = false;
    return config;
}

std::shared_ptr<DeviceBlockPoolConfig> makeMixedScaleConfig() {
    constexpr int    kBlockNum       = 8;
    constexpr size_t kTokensPerBlock = 1;

    rtp_llm::CacheConfig scaled_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(2, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_INT8, 1, 64);
    rtp_llm::CacheConfig plain_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(3, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_FP16, 1, 64);

    DeviceBlockPoolConfig scaled_pool = DeviceBlockPoolConfigHelper::createConfig(scaled_cfg);
    DeviceBlockPoolConfig plain_pool  = DeviceBlockPoolConfigHelper::createConfig(plain_cfg);

    MemoryLayoutConfig l0    = scaled_pool.memory_layouts[0];
    MemoryLayoutConfig l1    = plain_pool.memory_layouts[0];
    l1.kv_cache_offset_bytes = l0.total_size_bytes + l1.kv_cache_offset_bytes;
    l1.kv_scale_offset_bytes = l0.total_size_bytes + l1.kv_scale_offset_bytes;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = "mixed_scale_device";
    config->physical_block_count    = l0.block_num;
    config->total_size_bytes        = l0.total_size_bytes + l1.total_size_bytes;
    config->memory_layouts          = {l0, l1};
    config->use_cuda_malloc_backing = false;
    return config;
}

CacheConfig makeMtpCacheConfig() {
    CacheConfig main = rtp_llm::test::makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/3, /*seq_size_per_block=*/1, TYPE_FP16, /*local_head_num_kv=*/1, 64);
    main.mtp_sub_configs = {
        std::make_shared<CacheConfig>(rtp_llm::test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/4, /*seq_size_per_block=*/1, TYPE_FP16, /*local_head_num_kv=*/1, 64)),
        std::make_shared<CacheConfig>(rtp_llm::test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/5, /*seq_size_per_block=*/1, TYPE_FP16, /*local_head_num_kv=*/1, 64)),
    };
    return main;
}

}  // namespace

TEST(DeviceBlockPoolTest, InitKeepsBlockZeroInvalid) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_EQ(pool.totalBlocksNum(), config->physical_block_count - 1);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(DeviceBlockPoolTest, ConstructorRejectsInvalidConfigMatrix) {
    std::shared_ptr<const DeviceBlockPoolConfig> null_config;
    EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(null_config)));

    {
        auto config       = makeConfig();
        config->pool_type = BlockPoolType::HOST;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        config->memory_layouts.clear();
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        config->memory_layouts.push_back(config->memory_layouts.front());
        --config->memory_layouts.back().block_num;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                              = makeConfig();
        config->memory_layouts.front().layer_num = 0;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                                             = makeConfig();
        config->memory_layouts.front().kv_block_pool_size_bytes = 0;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        ++config->physical_block_count;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                                             = makeConfig();
        config->physical_block_count                            = 1;
        config->memory_layouts.front().block_num                = 1;
        config->memory_layouts.front().total_size_bytes         = config->memory_layouts.front().kv_block_stride_bytes;
        config->memory_layouts.front().kv_block_pool_size_bytes = config->memory_layouts.front().kv_block_stride_bytes;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
}

TEST(DeviceBlockPoolTest, MultiLayoutMtpConfigUsesMainBlockCountAndGlobalLayerMapping) {
    CacheConfig cache_config = makeMtpCacheConfig();
    ASSERT_EQ(cache_config.block_num, 3u);
    ASSERT_EQ(cache_config.mtp_sub_configs.size(), 2u);
    ASSERT_EQ(cache_config.mtp_sub_configs[0]->block_num, 4u);
    ASSERT_EQ(cache_config.mtp_sub_configs[1]->block_num, 5u);

    auto config = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfig(cache_config));
    ASSERT_EQ(config->physical_block_count, 3u);
    ASSERT_EQ(config->memory_layouts.size(), 3u);
    EXPECT_EQ(config->memory_layouts[0].layer_num, 2u);
    EXPECT_EQ(config->memory_layouts[1].layer_num, 1u);
    EXPECT_EQ(config->memory_layouts[2].layer_num, 1u);
    for (const auto& layout : config->memory_layouts) {
        EXPECT_EQ(layout.block_num, 3u);
    }

    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());
    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    const auto main_addr = pool.convertIndexToAddr(/*global_layer_id=*/0, *block);
    const auto mtp1_addr = pool.convertIndexToAddr(/*global_layer_id=*/2, *block);
    const auto mtp2_addr = pool.convertIndexToAddr(/*global_layer_id=*/3, *block);
    ASSERT_NE(main_addr.kv_addr, nullptr);
    ASSERT_NE(mtp1_addr.kv_addr, nullptr);
    ASSERT_NE(mtp2_addr.kv_addr, nullptr);
    EXPECT_NE(main_addr.kv_addr, mtp1_addr.kv_addr);
    EXPECT_NE(mtp1_addr.kv_addr, mtp2_addr.kv_addr);
}

TEST(DeviceBlockPoolTest, ConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);

    auto buffers = pool.convertIndexToBuffer(0, *block);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, PartitionedConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    auto buffers = pool.convertIndexToBuffer(0, *block, 1, 0);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, InvalidLayerAndBlockDoNotCorruptPoolState) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    pool.incRef(*block);
    const size_t free_before = pool.freeBlocksNum();
    const auto   refs_before = pool.refCount(*block);

    EXPECT_ANY_THROW(pool.convertIndexToBuffer(static_cast<int>(config->memory_layouts.front().layer_num), *block));
    EXPECT_ANY_THROW(pool.convertIndexToAddr(/*layer_id=*/0, static_cast<BlockIdxType>(config->physical_block_count)));

    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), refs_before);
    EXPECT_EQ(pool.freeBlocksNum(), free_before);
    pool.decRef(*block);
}

TEST(DeviceBlockPoolTest, LifecycleStartsAllocatedBlockWithZeroRefCount) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, LifecycleUsesIBlockPoolSemantics) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, ExposesAllocatorFacingLayerTensorsAndBuffers) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto cache_tensors = pool.allLayerCacheBase();
    ASSERT_FALSE(cache_tensors.empty());
    EXPECT_TRUE(cache_tensors[0].defined());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    auto addr = pool.convertIndexToAddr(/*layer_id=*/0, *block);
    EXPECT_NE(addr.kv_addr, nullptr);

    auto infos = pool.convertIndexToBuffer(/*layer_id=*/0, *block);
    ASSERT_FALSE(infos.empty());
    EXPECT_NE(infos[0].addr, nullptr);
    EXPECT_GT(infos[0].size_bytes, 0u);
    EXPECT_TRUE(infos[0].is_cuda);
}

TEST(DeviceBlockPoolTest, RegUserMrWithoutCacheStoreIsNoOp) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    // No cache store wired: MR registration must be a no-op, not a crash.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);

    // Idempotent / safe to call again and to deregister when nothing was registered.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    pool.deregUserMr();
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);
}

TEST(DeviceBlockPoolTest, AllLayerScaleCacheBaseStaysAlignedWithPartialScale) {
    auto            config = makeMixedScaleConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    const auto kv_bases    = pool.allLayerCacheBase();
    const auto scale_bases = pool.allLayerScaleCacheBase();

    ASSERT_EQ(kv_bases.size(), 5u);
    ASSERT_EQ(scale_bases.size(), 5u);

    for (const auto& kv : kv_bases) {
        EXPECT_TRUE(kv.defined());
        EXPECT_GT(kv.numel(), 0);
    }

    EXPECT_TRUE(scale_bases[0].defined());
    EXPECT_GT(scale_bases[0].numel(), 0);
    EXPECT_TRUE(scale_bases[1].defined());
    EXPECT_GT(scale_bases[1].numel(), 0);
    EXPECT_FALSE(scale_bases[2].defined() && scale_bases[2].numel() > 0);
    EXPECT_FALSE(scale_bases[3].defined() && scale_bases[3].numel() > 0);
    EXPECT_FALSE(scale_bases[4].defined() && scale_bases[4].numel() > 0);
}

TEST(DeviceBlockPoolTest, MallocAllocatesLowestBlockFirst) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto b1 = pool.malloc();
    auto b2 = pool.malloc();
    auto b3 = pool.malloc();
    ASSERT_TRUE(b1.has_value() && b2.has_value() && b3.has_value());
    EXPECT_EQ(*b1, 1);
    EXPECT_EQ(*b2, 2);
    EXPECT_EQ(*b3, 3);
}

}  // namespace rtp_llm
