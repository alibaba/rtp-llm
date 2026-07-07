#include "rtp_llm/cpp/cache/block_tree_cache/BlockPool.h"

#include <memory>

#include "gtest/gtest.h"

// Build the device pool config from the lightweight cache-config test helpers
// (cache_config_test_utils + BlockPoolConfigHelper) rather than BlockPoolTestHelper,
// which drags in the heavy //rtp_llm/cpp/cache monolith (and its CUDA-13 prebuilt
// remote_kv_cache_manager dependency that does not link in the CUDA-12 toolchain).
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::block_tree_cache {
namespace {

std::shared_ptr<BlockPoolConfig> makeConfig() {
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

    auto config                     = std::make_shared<BlockPoolConfig>();
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

TEST(BlockPoolTest, InitKeepsBlockZeroInvalid) {
    auto      config = makeConfig();
    BlockPool pool(config);

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_EQ(pool.totalBlocksNum(), config->physical_block_count - 1);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(BlockPoolTest, BlockBuffersCarryBlockIdxAndBytes) {
    auto      config = makeConfig();
    BlockPool pool(config);
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

TEST(BlockPoolTest, PartitionedBlockBuffersCarryBlockIdx) {
    auto      config = makeConfig();
    BlockPool pool(config);
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

TEST(BlockPoolTest, LifecycleUsesIBlockPoolSemantics) {
    auto      config = makeConfig();
    BlockPool pool(config);
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

}  // namespace rtp_llm::block_tree_cache
