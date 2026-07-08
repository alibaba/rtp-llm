#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

#include <memory>

#include "gtest/gtest.h"

// Build the device pool config from the lightweight cache-config test helpers
// (cache_config_test_utils + BlockPoolConfigHelper) rather than BlockPoolTestHelper,
// which drags in the heavy //rtp_llm/cpp/cache monolith (and its CUDA-13 prebuilt
// remote_kv_cache_manager dependency that does not link in the CUDA-12 toolchain).
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

std::shared_ptr<DeviceBlockPoolConfig> makeConfig() {
    constexpr int    kLayerNum       = 4;
    constexpr int    kBlockNum       = 10;
    constexpr size_t kTokensPerBlock = 1;
    rtp_llm::CacheConfig cache_config = rtp_llm::test::makeSimpleMhaCacheConfig(kLayerNum,
                                                                               kBlockNum,
                                                                               kTokensPerBlock,
                                                                               rtp_llm::TYPE_FP16,
                                                                               /*local_head_num_kv=*/1,
                                                                               /*size_per_head=*/64);
    rtp_llm::BlockPoolConfig old_cfg = rtp_llm::BlockPoolConfigHelper::createConfig(cache_config);

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = "device";
    config->physical_block_count    = old_cfg.block_num;
    config->free_block_order_policy = FreeBlockOrderPolicy::ANY_ORDER;
    config->total_size_bytes        = old_cfg.total_size_bytes;
    config->memory_layouts          = old_cfg.memory_layouts;
    config->allocation_type         = AllocationType::DEVICE;
    config->use_pinned_cpu_backing  = false;
    config->use_cuda_malloc_backing = false;
    return config;
}

}  // namespace

TEST(DeviceBlockPoolTest, InitKeepsBlockZeroInvalid) {
    auto      config = makeConfig();
    DeviceBlockPool pool(config);

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_EQ(pool.totalBlocksNum(), config->physical_block_count - 1);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(DeviceBlockPoolTest, BlockBuffersCarryBlockIdxAndBytes) {
    auto      config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);

    auto buffers = pool.blockBuffers(0, *block);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_EQ(buffer.block, *block);
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, PartitionedBlockBuffersCarryBlockIdx) {
    auto      config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    auto buffers = pool.blockBuffers(0, *block, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_EQ(buffer.block, *block);
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, LifecycleStartsAllocatedBlockWithZeroRefCount) {
    auto config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    pool.decRef(*block);
    EXPECT_EQ(pool.refCount(*block), 0u);
    EXPECT_TRUE(pool.isAllocated(*block));

    pool.free(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, LifecycleUsesIBlockPoolSemantics) {
    auto      config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    pool.decRef(*block);
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.free(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, ExposesAllocatorFacingLayerTensorsAndDeviceBlockViews) {
    auto config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto cache_tensors = pool.allLayerCacheBase();
    ASSERT_FALSE(cache_tensors.empty());
    EXPECT_TRUE(cache_tensors[0].defined());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    auto addr = pool.convertIndexToAddr(/*layer_id=*/0, *block);
    EXPECT_NE(addr.kv_addr, nullptr);

    auto buffers = pool.blockBuffers(/*layer_id=*/0, *block);
    ASSERT_FALSE(buffers.empty());
    EXPECT_NE(buffers[0].addr, nullptr);
    EXPECT_GT(buffers[0].bytes, 0u);
    EXPECT_EQ(buffers[0].block, *block);

    auto infos = pool.convertIndexToBuffer(/*layer_id=*/0, *block);
    ASSERT_EQ(infos.size(), buffers.size());
    EXPECT_EQ(infos[0].addr, buffers[0].addr);
    EXPECT_EQ(infos[0].size_bytes, buffers[0].bytes);
    EXPECT_TRUE(infos[0].is_cuda);
}

TEST(DeviceBlockPoolTest, RejectsPinnedCpuBacking) {
    auto config                    = makeConfig();
    config->use_pinned_cpu_backing = true;

    // RTP_LLM_CHECK aborts instead of throwing when core-dump-on-exception is enabled
    // (the default in this test env); flip it so the rejection is observable as a throw.
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;

    DeviceBlockPool pool(config);
    EXPECT_ANY_THROW((void)pool.init());

    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;
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

}  // namespace rtp_llm
